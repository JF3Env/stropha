"""Child-side runner for the Phase-5 subprocess sandbox (WIN-198).

Executed by :class:`stropha.meta.sandbox.SubprocessSandbox` in a fresh
Python interpreter — NEVER imported into the parent loop process. The
parent communicates by argv + stdin, the runner replies by printing
one JSON line to stdout + an exit code.

Usage::

    python -m stropha.meta._sandbox_runner <job> <candidate_dir> [args...]

Jobs
----

``import_check <candidate_dir>``
    Try to import every ``.py`` file under
    ``<candidate_dir>/adapters/``. Print one JSON line to stdout::

        {"ok": true|false, "imported": [...], "failed": [{path, error, traceback}]}

    Exit 0 on full success, 1 when at least one import failed.

Design constraints
------------------

- **Zero stropha dependencies.** The runner MUST NOT ``import stropha``
  except for narrow helpers — that would defeat the isolation purpose
  (a candidate that crashes stropha at import time would also crash
  the runner). Standard library only.
- **Deterministic stdout shape.** The parent parses the printed line
  with :func:`json.loads`; the runner must emit valid JSON in every
  branch, including the error paths.
- **No registry side effects propagate.** Each ``.py`` is loaded with
  a synthesised module name (``_candidate_<sanitised_path>``) so the
  in-process ``stropha.pipeline.registry._REGISTRY`` (which doesn't
  exist in this fresh interpreter anyway, by design) cannot be
  corrupted. Even if the candidate imports the registry, the child
  has its own copy — when the subprocess exits, every registration
  evaporates with it.
"""

from __future__ import annotations

import importlib.util
import json
import sys
import traceback
from pathlib import Path
from typing import Any


def _job_import_check(candidate_dir: Path) -> int:
    """Try to import every Python file under ``<candidate>/adapters/``.

    Emits one JSON line on stdout and returns the exit code (0 ok,
    1 partial / full failure).
    """
    adapters_dir = candidate_dir / "adapters"
    imported: list[str] = []
    failed: list[dict[str, Any]] = []

    if not adapters_dir.is_dir():
        print(
            json.dumps(
                {
                    "ok": True,
                    "imported": [],
                    "failed": [],
                    "reason": "no_adapters_dir",
                }
            )
        )
        return 0

    for py_path in sorted(adapters_dir.rglob("*.py")):
        rel = py_path.relative_to(adapters_dir)
        mod_name = "_candidate_" + str(rel).replace("/", "_").replace(".py", "")
        try:
            spec = importlib.util.spec_from_file_location(mod_name, py_path)
            if spec is None or spec.loader is None:
                raise ImportError(f"no module spec for {py_path}")
            module = importlib.util.module_from_spec(spec)
            sys.modules[mod_name] = module
            spec.loader.exec_module(module)
            imported.append(str(rel))
        except BaseException as exc:
            # BaseException catches SystemExit too — a candidate that
            # calls ``sys.exit()`` at module level would otherwise tear
            # down the runner mid-loop.
            failed.append(
                {
                    "path": str(rel),
                    "error": f"{type(exc).__name__}: {exc}",
                    "traceback": traceback.format_exc(),
                }
            )

    ok = not failed
    print(
        json.dumps(
            {
                "ok": ok,
                "imported": imported,
                "failed": failed,
            }
        )
    )
    return 0 if ok else 1


def _job_reranker_smoke(candidate_dir: Path) -> int:
    """Load the generated reranker and exercise its public protocol."""
    from stropha.meta.code_candidate import validate_code_candidate_directory
    from stropha.models import SearchHit
    from stropha.pipeline.registry import drop_namespace, namespace_scope
    from stropha.stages.reranker import RerankerStage

    namespace: str | None = None
    try:
        spec = validate_code_candidate_directory(candidate_dir)
        artifact = candidate_dir / spec.rel_path
        namespace = f"candidate-smoke-{candidate_dir.name}"
        with namespace_scope(namespace) as registry:
            before_modules = set(sys.modules)
            module_spec = importlib.util.spec_from_file_location(
                f"_candidate_reranker_{spec.name}", artifact
            )
            if module_spec is None or module_spec.loader is None:
                raise ImportError(f"no module spec for {artifact}")
            module = importlib.util.module_from_spec(module_spec)
            sys.dont_write_bytecode = True
            module_spec.loader.exec_module(module)
            newly_loaded = set(sys.modules) - before_modules
            forbidden_loaded = sorted(
                name
                for name in newly_loaded
                if name.split(".", 1)[0]
                in {"http", "multiprocessing", "requests", "socket", "subprocess", "urllib"}
            )
            if forbidden_loaded:
                raise ImportError(
                    "candidate loaded prohibited runtime modules: " + ", ".join(forbidden_loaded)
                )
            adapter_cls = registry.get(("reranker", spec.name))
            if adapter_cls is None:
                raise TypeError(f"artifact did not register reranker {spec.name!r}")
            config = adapter_cls.Config(**spec.adapter_config)
            adapter = adapter_cls(config)
            if not isinstance(adapter, RerankerStage):
                raise TypeError("generated adapter does not implement RerankerStage")
            hits = [
                SearchHit(
                    rank=index,
                    score=float(3 - index),
                    rel_path=f"src/example_{index}.py",
                    language="python",
                    kind="function",
                    symbol=f"example_{index}",
                    start_line=1,
                    end_line=2,
                    snippet=f"def example_{index}(): pass",
                    chunk_id=f"candidate-{index}",
                )
                for index in (1, 2)
            ]
            output = adapter.rerank("example query", hits, top_k=1)
            if not isinstance(output, list) or len(output) > 1:
                raise TypeError("rerank must return a list bounded by top_k")
            input_ids = {hit.chunk_id for hit in hits}
            output_ids = [hit.chunk_id for hit in output if isinstance(hit, SearchHit)]
            if len(output_ids) != len(output):
                raise TypeError("rerank must return only SearchHit instances")
            if len(output_ids) != len(set(output_ids)) or not set(output_ids) <= input_ids:
                raise TypeError("rerank returned duplicate or unknown candidate IDs")
    except BaseException as exc:
        print(
            json.dumps(
                {
                    "ok": False,
                    "job": "reranker_smoke",
                    "error": f"{type(exc).__name__}: {exc}",
                    "traceback": traceback.format_exc(),
                }
            )
        )
        return 1
    finally:
        if namespace is not None:
            drop_namespace(namespace)

    print(
        json.dumps(
            {
                "ok": True,
                "job": "reranker_smoke",
                "adapter": spec.name,
                "returned_candidate_ids": output_ids,
            }
        )
    )
    return 0


def _job_reranker_apply(candidate_dir: Path) -> int:
    """Apply the generated reranker to one bounded JSON request from stdin."""
    from pydantic import ValidationError

    from stropha.meta.code_candidate import validate_code_candidate_directory
    from stropha.meta.reranker_proxy import (
        MAX_RERANKER_HITS,
        MAX_RERANKER_QUERY_BYTES,
        MAX_RERANKER_TOP_K,
    )
    from stropha.meta.sandbox import MAX_INPUT_BYTES
    from stropha.models import SearchHit
    from stropha.pipeline.registry import drop_namespace, namespace_scope
    from stropha.stages.reranker import RerankerStage

    namespace: str | None = None
    try:
        raw = sys.stdin.buffer.read(MAX_INPUT_BYTES + 1)
        if len(raw) > MAX_INPUT_BYTES:
            raise ValueError(f"request exceeds {MAX_INPUT_BYTES} bytes")
        request = json.loads(raw)
        if not isinstance(request, dict) or set(request) != {"query", "top_k", "hits"}:
            raise TypeError("request must contain exactly query, top_k, and hits")

        query = request["query"]
        top_k = request["top_k"]
        raw_hits = request["hits"]
        if not isinstance(query, str):
            raise TypeError("query must be a string")
        if len(query.encode("utf-8")) > MAX_RERANKER_QUERY_BYTES:
            raise ValueError(f"query exceeds {MAX_RERANKER_QUERY_BYTES} bytes")
        if (
            isinstance(top_k, bool)
            or not isinstance(top_k, int)
            or not 1 <= top_k <= MAX_RERANKER_TOP_K
        ):
            raise ValueError(f"top_k must be between 1 and {MAX_RERANKER_TOP_K}")
        if not isinstance(raw_hits, list) or len(raw_hits) > MAX_RERANKER_HITS:
            raise ValueError(f"hits must be a list bounded by {MAX_RERANKER_HITS}")
        hit_fields = {
            "rank",
            "score",
            "rel_path",
            "language",
            "kind",
            "symbol",
            "start_line",
            "end_line",
            "snippet",
            "chunk_id",
            "streams",
        }
        if any(not isinstance(hit, dict) or set(hit) != hit_fields for hit in raw_hits):
            raise TypeError("each hit must contain exactly the allowed serialized fields")
        try:
            hits = [SearchHit.model_validate(hit) for hit in raw_hits]
        except ValidationError as exc:
            raise TypeError(f"invalid hit: {exc}") from exc
        input_ids = [hit.chunk_id for hit in hits]
        if len(input_ids) != len(set(input_ids)):
            raise ValueError("input contains duplicate chunk IDs")

        spec = validate_code_candidate_directory(candidate_dir)
        artifact = candidate_dir / spec.rel_path
        namespace = f"candidate-apply-{candidate_dir.name}"
        with namespace_scope(namespace) as registry:
            module_spec = importlib.util.spec_from_file_location(
                f"_candidate_reranker_apply_{spec.name}", artifact
            )
            if module_spec is None or module_spec.loader is None:
                raise ImportError(f"no module spec for {artifact}")
            module = importlib.util.module_from_spec(module_spec)
            sys.dont_write_bytecode = True
            module_spec.loader.exec_module(module)
            adapter_cls = registry.get(("reranker", spec.name))
            if adapter_cls is None:
                raise TypeError(f"artifact did not register reranker {spec.name!r}")
            adapter = adapter_cls(adapter_cls.Config(**spec.adapter_config))
            if not isinstance(adapter, RerankerStage):
                raise TypeError("generated adapter does not implement RerankerStage")
            output = adapter.rerank(query, hits, top_k=top_k)

        if not isinstance(output, list):
            raise TypeError("rerank must return a list")
        if len(output) > top_k:
            raise ValueError("rerank output exceeds top_k")
        if any(not isinstance(hit, SearchHit) for hit in output):
            raise TypeError("rerank must return only SearchHit instances")
        output_ids = [hit.chunk_id for hit in output]
        if len(output_ids) != len(set(output_ids)):
            raise ValueError("rerank returned duplicate chunk IDs")
        unknown = [chunk_id for chunk_id in output_ids if chunk_id not in set(input_ids)]
        if unknown:
            raise ValueError(f"rerank returned unknown chunk ID: {unknown[0]!r}")
    except BaseException as exc:
        print(
            json.dumps(
                {
                    "ok": False,
                    "job": "reranker_apply",
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
        )
        return 1
    finally:
        if namespace is not None:
            drop_namespace(namespace)

    print(
        json.dumps(
            {
                "ok": True,
                "job": "reranker_apply",
                "returned_chunk_ids": output_ids,
            }
        )
    )
    return 0


# Job dispatch table. Add new jobs as a new entry; each job takes the
# candidate_dir as its first positional and returns an exit code.
_JOBS = {
    "import_check": _job_import_check,
    "reranker_apply": _job_reranker_apply,
    "reranker_smoke": _job_reranker_smoke,
}


def main(argv: list[str] | None = None) -> int:
    """Entry point. ``argv`` defaults to ``sys.argv`` minus argv[0]."""
    args = argv if argv is not None else sys.argv[1:]
    if len(args) < 2:
        print(
            json.dumps(
                {
                    "ok": False,
                    "error": "usage: <job> <candidate_dir> [extra...]",
                }
            ),
            file=sys.stderr,
        )
        return 2
    job_name = args[0]
    candidate_dir = Path(args[1])
    job_fn = _JOBS.get(job_name)
    if job_fn is None:
        print(
            json.dumps(
                {
                    "ok": False,
                    "error": f"unknown job {job_name!r}; available: {sorted(_JOBS)}",
                }
            ),
            file=sys.stderr,
        )
        return 2
    return job_fn(candidate_dir)


if __name__ == "__main__":  # pragma: no cover - subprocess entry point
    sys.exit(main())
