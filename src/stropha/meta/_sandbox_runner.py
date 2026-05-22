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


# Job dispatch table. Add new jobs as a new entry; each job takes the
# candidate_dir as its first positional and returns an exit code.
_JOBS = {
    "import_check": _job_import_check,
}


def main(argv: list[str] | None = None) -> int:
    """Entry point. ``argv`` defaults to ``sys.argv`` minus argv[0]."""
    args = (argv if argv is not None else sys.argv[1:])
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
