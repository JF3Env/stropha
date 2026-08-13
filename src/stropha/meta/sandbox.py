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

import contextlib
import json
import os
import resource
import signal
import subprocess
import sys
import tempfile
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
MAX_OUTPUT_BYTES = 64 * 1024
MAX_INPUT_BYTES = 1024 * 1024
DEFAULT_CPU_S = 10
DEFAULT_MEMORY_BYTES = 512 * 1024 * 1024
DEFAULT_FILE_BYTES = 1024 * 1024
DEFAULT_PROCESSES = 8
_SAFE_ENV_KEYS = ("LANG", "LC_ALL", "PATH", "PYTHONHASHSEED", "PYTHONPATH", "TZ")

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
    max_output_bytes: int = MAX_OUTPUT_BYTES
    max_input_bytes: int = MAX_INPUT_BYTES
    cpu_limit_s: int = DEFAULT_CPU_S
    memory_limit_bytes: int = DEFAULT_MEMORY_BYTES
    file_limit_bytes: int = DEFAULT_FILE_BYTES
    process_limit: int = DEFAULT_PROCESSES

    def run(
        self,
        candidate_dir: Path,
        *,
        module: str = RUNNER_MODULE,
        job: str = "import_check",
        extra_args: tuple[str, ...] = (),
        input_data: str | bytes | None = None,
    ) -> SandboxResult:
        """Spawn the child + return a :class:`SandboxResult`.

        When ``<candidate_dir>/adapters/`` does not exist the sandbox
        short-circuits with ``SandboxResult(succeeded=True, payload={
        "skipped": True, "reason": "no_adapters_dir"})`` — there is
        nothing to isolate.
        """
        encoded_input = (
            input_data.encode("utf-8") if isinstance(input_data, str) else input_data or b""
        )
        if len(encoded_input) > self.max_input_bytes:
            raise SandboxError(f"sandbox input exceeds {self.max_input_bytes} bytes")

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

        env = {key: os.environ[key] for key in _SAFE_ENV_KEYS if key in os.environ}
        if self.extra_env:
            for key, value in self.extra_env.items():
                if key not in _SAFE_ENV_KEYS:
                    raise SandboxError(f"sandbox environment key is not allowed: {key}")
                env[key] = value
        # Prepend candidate adapters to PYTHONPATH so import-by-module
        # resolves to candidate code first.
        existing_pp = env.get("PYTHONPATH", "")
        python_paths = [str(adapters_dir), str(cwd / "src")]
        if existing_pp:
            python_paths.append(existing_pp)
        env["PYTHONPATH"] = os.pathsep.join(python_paths)

        log.info(
            "sandbox.start",
            candidate=str(candidate_dir),
            module=module,
            job=job,
            cwd=str(cwd),
            timeout_s=self.timeout_s,
        )

        proc: subprocess.Popen[bytes] | None = None
        t0 = time.monotonic()
        with tempfile.TemporaryDirectory(prefix="stropha-candidate-") as home:
            env.update(
                {
                    "HOME": home,
                    "TMPDIR": home,
                    "PYTHONDONTWRITEBYTECODE": "1",
                    "PYTHONNOUSERSITE": "1",
                }
            )
            with (
                tempfile.TemporaryFile() as stdin_file,
                tempfile.TemporaryFile() as stdout_file,
                tempfile.TemporaryFile() as stderr_file,
            ):
                stdin_file.write(encoded_input)
                stdin_file.seek(0)
                try:
                    proc = subprocess.Popen(
                        cmd,
                        cwd=str(candidate_dir),
                        env=env,
                        stdin=stdin_file,
                        stdout=stdout_file,
                        stderr=stderr_file,
                        start_new_session=True,
                        preexec_fn=(self._apply_resource_limits if os.name == "posix" else None),
                    )
                    try:
                        proc.wait(timeout=self.timeout_s)
                    except subprocess.TimeoutExpired:
                        if os.name == "posix":
                            os.killpg(proc.pid, signal.SIGKILL)
                        else:  # pragma: no cover - Windows fallback
                            proc.kill()
                        proc.wait()
                        stdout, _ = _read_bounded(stdout_file, self.max_output_bytes)
                        stderr, _ = _read_bounded(stderr_file, self.max_output_bytes)
                        duration = time.monotonic() - t0
                        return SandboxResult(
                            succeeded=False,
                            returncode=-1,
                            timed_out=True,
                            duration_s=duration,
                            command=cmd,
                            stdout=_tail_str(stdout, self.max_output_bytes),
                            stderr_tail=_tail_str(stderr, _TAIL_BYTES),
                            error=f"timeout after {self.timeout_s}s",
                        )
                    stdout, stdout_limited = _read_bounded(stdout_file, self.max_output_bytes)
                    stderr, _ = _read_bounded(stderr_file, self.max_output_bytes)
                except OSError as exc:
                    raise SandboxError(f"failed to spawn sandbox subprocess: {exc}") from exc

        duration = time.monotonic() - t0
        assert proc is not None
        crashed = proc.returncode < 0
        stdout = _tail_str(stdout, self.max_output_bytes)
        succeeded = proc.returncode == 0 and not stdout_limited
        payload = _parse_payload(stdout)

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
            stdout=stdout,
            stderr_tail=_tail_str(stderr, _TAIL_BYTES),
            error="stdout exceeded sandbox limit" if stdout_limited else None,
            payload=payload,
        )

    def _apply_resource_limits(self) -> None:
        """Set best-effort POSIX limits in the child immediately before exec."""
        limits = (
            (resource.RLIMIT_CPU, self.cpu_limit_s),
            (resource.RLIMIT_AS, self.memory_limit_bytes),
            (resource.RLIMIT_FSIZE, min(self.file_limit_bytes, self.max_output_bytes)),
            (resource.RLIMIT_NPROC, self.process_limit),
        )
        for limit, value in limits:
            with contextlib.suppress(OSError, ValueError):
                resource.setrlimit(limit, (value, value))


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


def _read_bounded(stream: Any, limit: int) -> tuple[str, bool]:
    stream.seek(0)
    data = stream.read(limit + 1)
    return data[:limit].decode("utf-8", errors="replace"), len(data) > limit


def _stropha_project_root() -> Path:
    """``src/stropha/meta/sandbox.py`` → ``stropha-repo/``. Three parents."""
    return Path(__file__).resolve().parents[3]


__all__ = [
    "DEFAULT_TIMEOUT_S",
    "MAX_INPUT_BYTES",
    "RUNNER_MODULE",
    "SandboxError",
    "SandboxResult",
    "SubprocessSandbox",
]
