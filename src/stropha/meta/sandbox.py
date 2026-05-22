"""Subprocess sandbox for Phase-5 code-search candidates (WIN-198).

A candidate adapter is untrusted Python. A typo in its module-level
imports, an infinite loop in ``__init__``, or a wild segfault in a C
extension dependency would corrupt or kill the Meta-Harness outer
loop if the adapter were imported in-process.

The sandbox solves that by spawning a *fresh* Python interpreter via
``subprocess.run`` with the candidate's ``adapters/`` directory
prepended to ``PYTHONPATH``. Any module-level side effect of the
candidate's code (`@register_adapter` registration, top-level imports,
class definitions) runs in that child process. From the parent loop's
perspective, the child is just a returncode + stdout JSON + stderr
tail. A crash on the child side becomes data on the parent side.

What this module DOES
---------------------

- ``SubprocessSandbox.run(candidate_dir, ...)`` — spawn the child with
  candidate adapters on the path, capture stdout / stderr, return a
  :class:`SandboxResult`.
- ``stropha.meta._sandbox_runner`` (sibling module) — the canonical
  child-side entry point. Currently ships one job, ``import_check``,
  which tries to import every ``.py`` under ``<candidate>/adapters/``
  and reports per-file success / failure.

What this module DOES NOT do
----------------------------

- It does NOT score candidates. Scoring requires WIN-199's namespaced
  registry to load candidate adapters alongside production without
  colliding. The sandbox + runner is the runtime-isolation layer that
  the scoring path will sit on top of.
- It does NOT run pytest. WIN-197's :class:`stropha.meta.gate.PytestGate`
  is the regression-guard layer; the sandbox runs AFTER the gate has
  cleared, when the parent loop wants to perform a more targeted
  candidate-loading check.

Why a subprocess and not multiprocessing
-----------------------------------------

``multiprocessing`` shares the parent's import state via ``fork``
(on POSIX). A candidate that registers a global side effect at import
time would leak into the parent's registry. ``subprocess.run`` spawns
a fully fresh interpreter — no shared state, no leakage. The cost is
one Python startup per call (~50-100 ms on Apple Silicon); acceptable
because Phase-5 candidates are expensive anyway and the import_check
is a small fraction of total per-candidate wall time.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..errors import StrophaError
from ..logging import get_logger

log = get_logger(__name__)


class SandboxError(StrophaError):
    """Raised when the sandbox runner itself cannot execute.

    Distinct from :class:`SandboxResult` ``succeeded=False`` — the
    latter means "the subprocess ran but reported a failure"; this
    exception means "we could not even spawn the subprocess".
    """


# Trim noisy stderr to a manageable tail. Stdout is preserved in full
# because the runner uses it as its JSON return channel.
_TAIL_BYTES = 4_000

DEFAULT_TIMEOUT_S: float = 60.0
"""Default per-call sandbox wall-clock. ``import_check`` typically
finishes in <500 ms; 60 s catches hangs while leaving room for slow
filesystems / cold-disk imports."""

RUNNER_MODULE = "stropha.meta._sandbox_runner"
"""Canonical child-side entry-point module. Callers may override via
``SubprocessSandbox.run(module=...)`` for tests."""


@dataclass
class SandboxResult:
    """Outcome of one ``SubprocessSandbox`` invocation."""

    succeeded: bool
    returncode: int
    timed_out: bool = False
    crashed: bool = False
    """True when the child returned a negative return code (POSIX
    signal — segfault, SIGKILL, …). ``succeeded`` is False in this
    case too; ``crashed`` lets the caller distinguish a clean
    non-zero exit from an abnormal death."""
    duration_s: float = 0.0
    command: list[str] = field(default_factory=list)
    stdout: str = ""
    """Full stdout — the runner uses this as its JSON return channel,
    so trimming would lose information."""
    stderr_tail: str = ""
    error: str | None = None
    payload: dict[str, Any] = field(default_factory=dict)
    """Parsed JSON the runner printed to stdout. Populated on
    success and best-effort on failure. When the stdout is not
    valid JSON, ``payload`` falls back to ``{"raw": stdout}`` so the
    caller still has something to inspect."""

    def to_dict(self) -> dict[str, Any]:
        return {
            "succeeded": self.succeeded,
            "returncode": self.returncode,
            "timed_out": self.timed_out,
            "crashed": self.crashed,
            "duration_s": round(self.duration_s, 3),
            "command": list(self.command),
            "stdout": self.stdout,
            "stderr_tail": self.stderr_tail,
            "error": self.error,
            "payload": dict(self.payload),
        }


@dataclass
class SubprocessSandbox:
    """Spawn a fresh Python interpreter with candidate adapters on PYTHONPATH.

    Args:
        python_binary: path to the Python interpreter. ``None`` =
            current ``sys.executable`` (same Python the loop runs in).
        timeout_s: subprocess timeout. ``None`` disables — only use
            for unit tests.
        project_root: cwd for the child. ``None`` = stropha repo root
            (resolved from this module's filesystem location).
        extra_env: additional environment variables merged on top of
            the parent's ``os.environ``.
    """

    python_binary: str | None = None
    timeout_s: float | None = DEFAULT_TIMEOUT_S
    project_root: Path | None = None
    extra_env: dict[str, str] | None = None

    def run(
        self,
        candidate_dir: Path,
        *,
        module: str = RUNNER_MODULE,
        job: str = "import_check",
        extra_args: tuple[str, ...] = (),
    ) -> SandboxResult:
        """Spawn the child + return a :class:`SandboxResult`.

        When ``<candidate_dir>/adapters/`` does not exist the sandbox
        short-circuits with ``SandboxResult(succeeded=True, payload={
        "skipped": True, "reason": "no_adapters_dir"})`` — there is
        nothing to isolate.
        """
        adapters_dir = candidate_dir / "adapters"
        if not adapters_dir.is_dir():
            log.info(
                "sandbox.skipped",
                candidate=str(candidate_dir),
                reason="no_adapters_dir",
            )
            return SandboxResult(
                succeeded=True,
                returncode=0,
                payload={"skipped": True, "reason": "no_adapters_dir"},
            )

        py = self.python_binary or sys.executable
        cmd = [py, "-m", module, job, str(candidate_dir), *extra_args]
        cwd = self.project_root or _stropha_project_root()

        env = dict(os.environ)
        if self.extra_env:
            env.update(self.extra_env)
        # Prepend candidate adapters to PYTHONPATH so import-by-module
        # resolves to candidate code first.
        existing_pp = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = (
            f"{adapters_dir}{os.pathsep}{existing_pp}"
            if existing_pp
            else str(adapters_dir)
        )

        log.info(
            "sandbox.start",
            candidate=str(candidate_dir),
            module=module,
            job=job,
            cwd=str(cwd),
            timeout_s=self.timeout_s,
        )

        t0 = time.monotonic()
        try:
            proc = subprocess.run(
                cmd,
                cwd=str(cwd),
                env=env,
                capture_output=True,
                text=True,
                timeout=self.timeout_s,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            duration = time.monotonic() - t0
            log.warning(
                "sandbox.timeout",
                candidate=str(candidate_dir),
                timeout_s=self.timeout_s,
                duration_s=round(duration, 3),
            )
            return SandboxResult(
                succeeded=False,
                returncode=-1,
                timed_out=True,
                duration_s=duration,
                command=cmd,
                stdout=_tail_str(exc.stdout, _TAIL_BYTES),
                stderr_tail=_tail_str(exc.stderr, _TAIL_BYTES),
                error=f"timeout after {self.timeout_s}s",
            )
        except OSError as exc:
            raise SandboxError(
                f"failed to spawn sandbox subprocess: {exc}"
            ) from exc

        duration = time.monotonic() - t0
        crashed = proc.returncode < 0
        succeeded = proc.returncode == 0
        payload = _parse_payload(proc.stdout)

        log.info(
            "sandbox.done",
            candidate=str(candidate_dir),
            succeeded=succeeded,
            crashed=crashed,
            returncode=proc.returncode,
            duration_s=round(duration, 3),
        )
        return SandboxResult(
            succeeded=succeeded,
            returncode=proc.returncode,
            crashed=crashed,
            duration_s=duration,
            command=cmd,
            stdout=proc.stdout or "",
            stderr_tail=_tail_str(proc.stderr, _TAIL_BYTES),
            payload=payload,
        )


def _parse_payload(stdout: str | None) -> dict[str, Any]:
    """Best-effort parse of the runner's stdout into a JSON dict.

    Returns the parsed dict on success. When the stdout is not valid
    JSON OR is JSON but not a dict, falls back to ``{"raw": stdout}``
    so callers can still inspect the output without surprise
    KeyError.
    """
    if not stdout:
        return {}
    try:
        obj = json.loads(stdout)
    except (json.JSONDecodeError, ValueError):
        return {"raw": stdout}
    if isinstance(obj, dict):
        return obj
    return {"raw": stdout}


def _tail_str(text: str | bytes | None, n: int) -> str:
    """Return the last ``n`` bytes of ``text``, decoded best-effort."""
    if text is None:
        return ""
    if isinstance(text, bytes):
        text = text.decode("utf-8", errors="replace")
    if len(text) > n:
        return text[-n:]
    return text


def _stropha_project_root() -> Path:
    """``src/stropha/meta/sandbox.py`` → ``stropha-repo/``. Three parents."""
    return Path(__file__).resolve().parents[3]


__all__ = [
    "DEFAULT_TIMEOUT_S",
    "RUNNER_MODULE",
    "SandboxError",
    "SandboxResult",
    "SubprocessSandbox",
]
