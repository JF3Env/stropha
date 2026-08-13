from __future__ import annotations

from pathlib import Path

import pytest

from stropha.meta import PytestGate, SubprocessSandbox, _sandbox_runner
from stropha.meta.code_candidate import (
    MAX_CODE_ARTIFACT_BYTES,
    CodeCandidateError,
    validate_code_candidate,
    validate_code_candidate_directory,
)
from stropha.meta.store import CODE_ARTIFACTS_KEY, ArtifactStore, ArtifactStoreError


def _source(name: str = "candidate") -> str:
    return f'''from pydantic import BaseModel
from stropha.models import SearchHit
from stropha.pipeline.base import StageHealth
from stropha.pipeline.registry import register_adapter

class Config(BaseModel):
    pass

@register_adapter(stage="reranker", name="{name}")
class CandidateReranker:
    Config = Config
    def __init__(self, config=None): self.config = config or Config()
    @property
    def stage_name(self): return "reranker"
    @property
    def adapter_name(self): return "{name}"
    @property
    def adapter_id(self): return "reranker:{name}"
    @property
    def config_schema(self): return Config
    def health(self): return StageHealth(status="ready", message="candidate")
    def rerank(self, query: str, hits: list[SearchHit], *, top_k: int = 10):
        return hits[:top_k]
'''


def test_valid_candidate_requires_path_decorator_and_config_agreement() -> None:
    spec = validate_code_candidate(
        {"reranker": {"adapter": "candidate", "config": {}}},
        {"adapters/reranker/candidate.py": _source()},
    )
    assert spec.name == "candidate"
    assert spec.adapter_config == {}


@pytest.mark.parametrize(
    "config,path,source,match",
    [
        ({"reranker": "candidate"}, "adapters/enricher/candidate.py", _source(), "path"),
        ({"reranker": "other"}, "adapters/reranker/candidate.py", _source(), "candidate reranker"),
        (
            {"reranker": "candidate"},
            "adapters/reranker/candidate.py",
            _source("other"),
            "path name",
        ),
        (
            {"reranker": "candidate"},
            "adapters/reranker/candidate.py",
            _source().replace('stage="reranker"', 'stage="enricher"'),
            "stage",
        ),
        (
            {"reranker": "candidate"},
            "adapters/reranker/candidate.py",
            "import socket\n" + _source(),
            "prohibited import",
        ),
        (
            {"reranker": "candidate"},
            "adapters/reranker/candidate.py",
            "open('owned', 'w')\n" + _source(),
            "prohibited",
        ),
    ],
)
def test_static_validation_rejects_invalid_candidate(
    config: dict[str, object], path: str, source: str, match: str
) -> None:
    with pytest.raises(CodeCandidateError, match=match):
        validate_code_candidate(config, {path: source})


def test_store_enforces_artifact_count_size_and_types(tmp_path: Path) -> None:
    store = ArtifactStore(tmp_path)
    with pytest.raises(ArtifactStoreError, match="exactly one"):
        store.register_candidate(
            {
                "reranker": "candidate",
                CODE_ARTIFACTS_KEY: {
                    "adapters/reranker/a.py": _source("a"),
                    "adapters/reranker/b.py": _source("b"),
                },
            }
        )
    with pytest.raises(ArtifactStoreError, match="strings"):
        store.register_candidate(
            {"reranker": "candidate", CODE_ARTIFACTS_KEY: {1: _source()}}
        )
    oversized = _source() + "#" * MAX_CODE_ARTIFACT_BYTES
    with pytest.raises(ArtifactStoreError, match="exceeds"):
        store.register_candidate(
            {
                "reranker": "candidate",
                CODE_ARTIFACTS_KEY: {"adapters/reranker/candidate.py": oversized},
            }
        )


@pytest.mark.parametrize("candidate_id", ["../escape", "a/b", ".", "", "a\\b"])
def test_candidate_ids_cannot_traverse(tmp_path: Path, candidate_id: str) -> None:
    store = ArtifactStore(tmp_path)
    with pytest.raises(ArtifactStoreError, match="candidate id"):
        store.register_candidate({}, candidate_id=candidate_id)


def test_directory_validation_rejects_extra_non_python_artifact(tmp_path: Path) -> None:
    store = ArtifactStore(tmp_path)
    record = store.register_candidate(
        {
            "reranker": "candidate",
            CODE_ARTIFACTS_KEY: {"adapters/reranker/candidate.py": _source()},
        }
    )
    assert record.base_dir is not None
    (record.base_dir / "adapters" / "README.md").write_text("extra", encoding="utf-8")
    with pytest.raises(CodeCandidateError, match="exactly one"):
        validate_code_candidate_directory(record.base_dir)


def test_reranker_smoke_checks_protocol_and_candidate_ids(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    store = ArtifactStore(tmp_path)
    record = store.register_candidate(
        {
            "reranker": "candidate",
            CODE_ARTIFACTS_KEY: {"adapters/reranker/candidate.py": _source()},
        }
    )
    assert record.base_dir is not None
    assert _sandbox_runner.main(["reranker_smoke", str(record.base_dir)]) == 0
    assert '"returned_candidate_ids": ["candidate-1"]' in capsys.readouterr().out

    artifact = record.base_dir / "adapters" / "reranker" / "candidate.py"
    artifact.write_text(
        _source().replace(
            "return hits[:top_k]",
            "return [hits[0].model_copy(update={'chunk_id': 'unknown'})]",
        ),
        encoding="utf-8",
    )
    assert _sandbox_runner.main(["reranker_smoke", str(record.base_dir)]) == 1
    assert "candidate IDs" in capsys.readouterr().out


def test_sandbox_sanitizes_environment_and_runs_real_smoke(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "do-not-inherit")
    store = ArtifactStore(tmp_path)
    record = store.register_candidate(
        {
            "reranker": "candidate",
            CODE_ARTIFACTS_KEY: {"adapters/reranker/candidate.py": _source()},
        }
    )
    assert record.base_dir is not None
    result = SubprocessSandbox(timeout_s=30).run(record.base_dir, job="reranker_smoke")
    assert result.succeeded is True, result.to_dict()
    assert result.payload["adapter"] == "candidate"


def test_pytest_gate_runs_candidate_smoke_before_production_tests(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = ArtifactStore(tmp_path)
    record = store.register_candidate(
        {
            "reranker": "candidate",
            CODE_ARTIFACTS_KEY: {"adapters/reranker/candidate.py": _source()},
        }
    )
    assert record.base_dir is not None

    class FailingSandbox:
        def run(self, candidate_dir: Path, *, job: str):
            from stropha.meta import SandboxResult

            assert candidate_dir == record.base_dir
            assert job == "reranker_smoke"
            return SandboxResult(succeeded=False, returncode=1, stdout="smoke failed")

    monkeypatch.setattr(
        "stropha.meta.gate.subprocess.run",
        lambda *args, **kwargs: pytest.fail("production pytest ran before smoke passed"),
    )
    result = PytestGate(uv_binary="uv", sandbox=FailingSandbox())(record.base_dir)  # type: ignore[arg-type]
    assert result.passed is False
    assert "smoke" in (result.error or "")
