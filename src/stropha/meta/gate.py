"""Pytest gate for Phase-5 code-search candidates (WIN-197).

A candidate that ships ``code_artifacts`` (WIN-196) brings new adapter
source files to the table. Before the candidate is scored, the gate
runs ``uv run pytest -q``: if the suite goes red, the candidate is
rejected outright — no ScoreCard is computed, no ``scores.json`` is
written, the loop logs the failure and moves on.

Why pytest at all
-----------------

Per Lee et al. §5 + RFC §Phase-5 risk table: a proposed adapter that
breaks the suite is a candidate whose code is broken on its face.
There is no point burning the expensive end-to-end search eval (~30s
per candidate at HEAD) on a harness whose imports already fail.

The gate is the cheap regression guard. WIN-198 will sandbox the
candidate's *scoring* path in a subprocess so even an adapter that
imports cleanly cannot crash the loop process at retrieval time.
WIN-199 will namespace the registry so concurrent candidates with
same-named adapters do not collide. Together those three turn the
Phase-5 plain pipe (proposer → score) into a hardened
(proposer → gate → sandbox → score) chain.

When the gate is a no-op
-------------------------

The gate runs IFF the candidate's ``adapters/`` directory exists AND
contains at least one ``.py`` file. Plain config-flip candidates
(no ``code_artifacts``) skip the gate entirely — running the full
suite for a flag toggle is pure waste.

GateError vs GateResult(passed=False)
-------------------------------------

- :class:`GateError` is raised when the runner itself is broken: no
  ``uv`` on PATH, subprocess could not spawn, etc. This is a loop-fatal
  condition; the caller decides whether to retry or abort.
- :class:`GateResult` with ``passed=False`` is the *normal* gate
  failure path: pytest ran, exit code was non-zero. The candidate is
  rejected, ``gate.json`` is persisted for inspection, and the loop
  continues with the next iteration.
"""

from __future__ import annotations

import shutil
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..errors import StrophaError
from ..logging import get_logger
from .code_candidate import CodeCandidateError, validate_code_candidate_directory
from .sandbox import SandboxError, SubprocessSandbox

log = get_logger(__name__)


class GateError(StrophaError):
    """Raised when the gate runner itself cannot execute.

    A failed pytest suite is NOT a GateError — that path returns
    ``GateResult(passed=False, ...)``. GateError is reserved for
    "we could not even run pytest" conditions (no uv binary,
    subprocess could not spawn, …).
    """


# Trim stdout / stderr to a manageable size in ``gate.json``. Long
# tracebacks aren't useful in the archive — the human reading the
# archive needs the last few KB to see the failure summary.
_TAIL_BYTES = 4_000

# Spec ``uv run pytest -q`` — see RFC §Phase-5 deliverables. ``--no-header``
# saves three lines of noise per invocation.
DEFAULT_PYTEST_ARGS: tuple[str, ...] = ("-q", "--no-header")

# Hard upper bound on gate wall-clock. The full suite at HEAD runs in
# ~22s; 600s leaves headroom for slow CI and forces termination on hangs.
DEFAULT_TIMEOUT_S: float = 600.0


@dataclass
class GateResult:
    """Outcome of one gate invocation.

    ``passed`` is the single source of truth the loop reads. The other
    fields are for the human (or a future agentic proposer) digging
    into ``gate.json`` to understand why a candidate was rejected.
    """

    passed: bool
    skipped: bool = False
    """True when no artifacts were present and the gate ran no
    subprocess. ``passed`` is True in this case too."""
    returncode: int = 0
    duration_s: float = 0.0
    command: list[str] = field(default_factory=list)
    stdout_tail: str = ""
    stderr_tail: str = ""
    error: str | None = None
    """Populated when the gate ran but the runner reported an
    abnormal state (timeout, signal, …). ``None`` on the happy path
    and on plain pytest failure."""

    def to_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "skipped": self.skipped,
            "returncode": self.returncode,
            "duration_s": round(self.duration_s, 3),
            "command": list(self.command),
            "stdout_tail": self.stdout_tail,
            "stderr_tail": self.stderr_tail,
            "error": self.error,
        }


@dataclass
class PytestGate:
    """Run ``uv run pytest`` from a project root before scoring.

    The gate is conditional: a candidate with no ``adapters/`` content
    skips the subprocess entirely and returns ``GateResult(passed=True,
    skipped=True)``. Only candidates carrying at least one ``.py``
    artifact pay the suite-run cost.

    Args:
        project_root: directory ``uv run pytest`` runs from. Defaults
            to the stropha repo root (resolved from this module's
            filesystem location).
        pytest_args: arguments appended after ``uv run pytest``.
        timeout_s: subprocess timeout. ``None`` disables the timeout —
            useful for tests but never recommended in production.
        uv_binary: optional explicit ``uv`` path. ``None`` =
            ``shutil.which("uv")`` at call time.
    """

    project_root: Path | None = None
    pytest_args: tuple[str, ...] = DEFAULT_PYTEST_ARGS
    timeout_s: float | None = DEFAULT_TIMEOUT_S
    uv_binary: str | None = None
    sandbox: SubprocessSandbox | None = None

    def __call__(self, candidate_dir: Path) -> GateResult:
        adapters_dir = candidate_dir / "adapters"
        # Skip cheap candidates: no adapters/ dir OR adapters/ has no
        # Python source. Both are common ("plain config flip" candidates).
        if not adapters_dir.is_dir() or not any(adapters_dir.rglob("*.py")):
            log.info(
                "gate.skipped",
                candidate=str(candidate_dir),
                reason="no_artifacts",
            )
            return GateResult(passed=True, skipped=True)

        try:
            validate_code_candidate_directory(candidate_dir)
        except CodeCandidateError as exc:
            return GateResult(
                passed=False,
                returncode=1,
                error=f"candidate code validation failed: {exc}",
            )

        try:
            smoke = (self.sandbox or SubprocessSandbox()).run(
                candidate_dir, job="reranker_smoke"
            )
        except SandboxError as exc:
            raise GateError(f"failed to run candidate smoke subprocess: {exc}") from exc
        if not smoke.succeeded:
            return GateResult(
                passed=False,
                returncode=smoke.returncode,
                duration_s=smoke.duration_s,
                command=smoke.command,
                stdout_tail=_tail(smoke.stdout),
                stderr_tail=smoke.stderr_tail,
                error=smoke.error or "candidate reranker import/protocol smoke failed",
            )

        uv = self.uv_binary or shutil.which("uv")
        if uv is None:
            raise GateError(
                "uv binary not found on PATH; install uv or set "
                "PytestGate(uv_binary=...) explicitly"
            )

        cmd = [uv, "run", "pytest", *self.pytest_args]
        cwd = self.project_root or _stropha_project_root()

        log.info(
            "gate.start",
            candidate=str(candidate_dir),
            command=cmd,
            cwd=str(cwd),
            timeout_s=self.timeout_s,
        )

        t0 = time.monotonic()
        try:
            proc = subprocess.run(
                cmd,
                cwd=str(cwd),
                capture_output=True,
                text=True,
                timeout=self.timeout_s,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            duration = time.monotonic() - t0
            log.warning(
                "gate.timeout",
                candidate=str(candidate_dir),
                timeout_s=self.timeout_s,
                duration_s=round(duration, 3),
            )
            return GateResult(
                passed=False,
                returncode=-1,
                duration_s=duration,
                command=cmd,
                stdout_tail=_tail(exc.stdout),
                stderr_tail=_tail(exc.stderr),
                error=f"timeout after {self.timeout_s}s",
            )
        except OSError as exc:
            # subprocess couldn't even spawn (e.g. PATH manipulated,
            # uv binary unreadable). Surface as GateError so the loop
            # can decide whether to abort.
            raise GateError(f"failed to spawn gate subprocess: {exc}") from exc

        duration = time.monotonic() - t0
        passed = proc.returncode == 0
        log.info(
            "gate.done",
            candidate=str(candidate_dir),
            passed=passed,
            returncode=proc.returncode,
            duration_s=round(duration, 3),
        )
        return GateResult(
            passed=passed,
            returncode=proc.returncode,
            duration_s=duration,
            command=cmd,
            stdout_tail=_tail(proc.stdout),
            stderr_tail=_tail(proc.stderr),
        )


def _stropha_project_root() -> Path:
    """Resolve the stropha repo root for the default subprocess cwd.

    ``src/stropha/meta/gate.py`` -> ``stropha-repo/``. Three parents.
    """
    return Path(__file__).resolve().parents[3]


def _tail(text: str | bytes | None) -> str:
    """Return the last ``_TAIL_BYTES`` of ``text``, decoded best-effort.

    Bytes input is decoded as UTF-8 with ``errors="replace"`` so the
    tail never raises on a malformed byte sequence — that would
    sabotage the archive for what is essentially a diagnostic field.
    """
    if text is None:
        return ""
    if isinstance(text, bytes):
        text = text.decode("utf-8", errors="replace")
    if len(text) > _TAIL_BYTES:
        return text[-_TAIL_BYTES:]
    return text


__all__ = [
    "DEFAULT_PYTEST_ARGS",
    "DEFAULT_TIMEOUT_S",
    "GateError",
    "GateResult",
    "PytestGate",
]
