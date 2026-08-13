from __future__ import annotations

import json
import sys
from collections.abc import Callable
from pathlib import Path

import pytest

from stropha.adapters.retrieval.hybrid_rrf import HybridRrfConfig, HybridRrfRetrieval
from stropha.meta.reranker_proxy import (
    MAX_RERANKER_HITS,
    MAX_RERANKER_QUERY_BYTES,
    SandboxedRerankerProxy,
)
from stropha.meta.sandbox import SandboxError, SandboxResult
from stropha.models import SearchHit
from stropha.pipeline.base import StageHealth
from stropha.stages.reranker import RerankerStage


def _hit(chunk_id: str, *, rank: int = 1) -> SearchHit:
    return SearchHit(
        rank=rank,
        score=float(10 - rank),
        rel_path=f"src/{chunk_id}.py",
        language="python",
        kind="function",
        symbol=chunk_id,
        start_line=1,
        end_line=2,
        snippet=f"def {chunk_id}(): pass",
        chunk_id=chunk_id,
    )


def _candidate(tmp_path: Path, rerank_body: str) -> Path:
    candidate_dir = tmp_path / "candidate"
    artifact = candidate_dir / "adapters" / "reranker" / "generated.py"
    artifact.parent.mkdir(parents=True)
    artifact.write_text(
        f"""from pydantic import BaseModel
from stropha.models import SearchHit
from stropha.pipeline.base import StageHealth
from stropha.pipeline.registry import register_adapter

class Config(BaseModel):
    pass

@register_adapter(stage="reranker", name="generated")
class GeneratedReranker:
    Config = Config
    def __init__(self, config=None): self.config = config or Config()
    @property
    def stage_name(self): return "reranker"
    @property
    def adapter_name(self): return "generated"
    @property
    def adapter_id(self): return "reranker:generated"
    @property
    def config_schema(self): return Config
    def health(self): return StageHealth(status="ready", message="generated")
    def rerank(self, query: str, hits: list[SearchHit], *, top_k: int = 10):
        {rerank_body}
""",
        encoding="utf-8",
    )
    (candidate_dir / "config.yaml").write_text(
        "reranker:\n  adapter: generated\n  config: {}\n",
        encoding="utf-8",
    )
    return candidate_dir


class _FakeSandbox:
    def __init__(self, result: SandboxResult, requests: list[dict[str, object]]) -> None:
        self._result = result
        self._requests = requests

    def run(
        self,
        candidate_dir: Path,
        *,
        job: str,
        input_data: str,
    ) -> SandboxResult:
        self._requests.append(
            {
                "candidate_dir": candidate_dir,
                "job": job,
                "payload": json.loads(input_data),
            }
        )
        return self._result


def _factory(
    result: SandboxResult, requests: list[dict[str, object]]
) -> Callable[[], _FakeSandbox]:
    return lambda: _FakeSandbox(result, requests)


def _result(ids: object) -> SandboxResult:
    return SandboxResult(
        succeeded=True,
        returncode=0,
        payload={
            "ok": True,
            "job": "reranker_apply",
            "returned_chunk_ids": ids,
        },
    )


def test_proxy_implements_protocol_and_reorders_original_hits(tmp_path: Path) -> None:
    requests: list[dict[str, object]] = []
    proxy = SandboxedRerankerProxy(
        tmp_path,
        adapter_name="generated",
        sandbox_factory=_factory(_result(["b", "a"]), requests),  # type: ignore[arg-type]
    )
    hits = [_hit("a"), _hit("b", rank=2)]

    output = proxy.rerank("query", hits, top_k=2)

    assert isinstance(proxy, RerankerStage)
    assert output == [hits[1], hits[0]]
    assert output[0] is hits[1]
    assert requests[0]["job"] == "reranker_apply"
    payload = requests[0]["payload"]
    assert isinstance(payload, dict)
    assert payload["query"] == "query"
    assert payload["top_k"] == 2
    assert all("repo" not in hit for hit in payload["hits"])  # type: ignore[union-attr]


def test_proxy_creates_a_fresh_sandbox_for_each_call(tmp_path: Path) -> None:
    created = 0

    def factory() -> _FakeSandbox:
        nonlocal created
        created += 1
        return _FakeSandbox(_result(["a"]), [])

    proxy = SandboxedRerankerProxy(
        tmp_path,
        adapter_name="generated",
        sandbox_factory=factory,  # type: ignore[arg-type]
    )
    proxy.rerank("one", [_hit("a")], top_k=1)
    proxy.rerank("two", [_hit("a")], top_k=1)
    assert created == 2


@pytest.mark.parametrize(
    "result,match",
    [
        (SandboxResult(succeeded=True, returncode=0, payload={}), "malformed"),
        (_result("a"), "list of chunk IDs"),
        (_result(["a", "a"]), "duplicate"),
        (_result(["unknown"]), "unknown"),
        (_result(["a", "b"]), "exceeds top_k"),
        (
            SandboxResult(
                succeeded=False,
                returncode=0,
                error="stdout exceeded sandbox limit",
            ),
            "stdout exceeded",
        ),
    ],
)
def test_proxy_rejects_invalid_child_output(
    tmp_path: Path, result: SandboxResult, match: str
) -> None:
    proxy = SandboxedRerankerProxy(
        tmp_path,
        adapter_name="generated",
        sandbox_factory=_factory(result, []),  # type: ignore[arg-type]
    )
    with pytest.raises(SandboxError, match=match):
        proxy.rerank("query", [_hit("a"), _hit("b", rank=2)], top_k=1)


def test_proxy_rejects_oversized_input_before_spawning(tmp_path: Path) -> None:
    calls = 0

    def factory() -> _FakeSandbox:
        nonlocal calls
        calls += 1
        return _FakeSandbox(_result([]), [])

    proxy = SandboxedRerankerProxy(
        tmp_path,
        adapter_name="generated",
        sandbox_factory=factory,  # type: ignore[arg-type]
    )
    with pytest.raises(SandboxError, match="query exceeds"):
        proxy.rerank("x" * (MAX_RERANKER_QUERY_BYTES + 1), [_hit("a")])
    with pytest.raises(SandboxError, match="hits"):
        proxy.rerank(
            "query",
            [_hit(str(index)) for index in range(MAX_RERANKER_HITS + 1)],
        )
    assert calls == 0


def test_real_child_job_reranks_without_importing_candidate_in_parent(tmp_path: Path) -> None:
    candidate_dir = _candidate(tmp_path, "return list(reversed(hits))[:top_k]")
    before_modules = set(sys.modules)
    hits = [_hit("a"), _hit("b", rank=2)]

    output = SandboxedRerankerProxy(candidate_dir, adapter_name="generated").rerank(
        "query", hits, top_k=2
    )

    assert output == [hits[1], hits[0]]
    assert not any(
        name.startswith("_candidate_reranker_apply_generated")
        for name in set(sys.modules) - before_modules
    )


def test_child_job_rejects_candidate_returning_unknown_id(tmp_path: Path) -> None:
    candidate_dir = _candidate(
        tmp_path,
        "return [hits[0].model_copy(update={'chunk_id': 'unknown'})]",
    )
    proxy = SandboxedRerankerProxy(candidate_dir, adapter_name="generated")
    with pytest.raises(SandboxError, match="unknown chunk ID"):
        proxy.rerank("query", [_hit("a")], top_k=1)


class _Storage:
    def search_dense(self, vec: list[float], k: int) -> list[SearchHit]:
        return [_hit("a"), _hit("b", rank=2)]

    def search_bm25(self, query: str, k: int) -> list[SearchHit]:
        return []

    def search_symbol_tokens(self, query: str, k: int) -> list[SearchHit]:
        return []


class _Embedder:
    @property
    def adapter_name(self) -> str:
        return "stub"

    def embed_query(self, query: str) -> list[float]:
        return [1.0]


class _InjectedReranker:
    Config = SandboxedRerankerProxy.Config
    stage_name = "reranker"
    adapter_name = "injected"
    adapter_id = "reranker:injected"
    config_schema = SandboxedRerankerProxy.Config

    def health(self) -> StageHealth:
        return StageHealth(status="ready", message="injected")

    def rerank(self, query: str, hits: list[SearchHit], *, top_k: int = 10) -> list[SearchHit]:
        return list(reversed(hits))[:top_k]


def test_hybrid_rrf_public_reranker_injection_api() -> None:
    retrieval = HybridRrfRetrieval(
        HybridRrfConfig(
            top_k=2,
            streams={"sparse": None, "symbol": None},
            learned_weights_enabled=False,
        ),
        storage=_Storage(),  # type: ignore[arg-type]
        embedder=_Embedder(),  # type: ignore[arg-type]
    )
    retrieval.set_reranker(_InjectedReranker())
    assert [hit.chunk_id for hit in retrieval.search("query")] == ["b", "a"]
