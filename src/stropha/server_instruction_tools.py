"""Repository-scoped saved-worktree instruction resolution."""

from __future__ import annotations

import hashlib
import os
import stat
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from .errors import InstructionResolutionError, ObservationError
from .models import MAX_AGENT_PATH_CHARS
from .server_memory_tools import _resolve_repo
from .server_models import InstructionContextResponse, InstructionSourceEntry
from .server_observation_tools import _canonical_relative_path
from .server_runtime import AppContext

INSTRUCTION_FILENAMES = ("AGENTS.md", "CLAUDE.md")
DEFAULT_INSTRUCTION_CHARS = 16_000
MAX_INSTRUCTION_CHARS = 32_768
MAX_INSTRUCTION_FILE_BYTES = 256 * 1024
MAX_INSTRUCTION_TOTAL_BYTES = 1024 * 1024
MAX_INSTRUCTION_SOURCES = 64

_BASE_OPEN_FLAGS = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
_FILE_OPEN_FLAGS = _BASE_OPEN_FLAGS | getattr(os, "O_NONBLOCK", 0)
_DIRECTORY_OPEN_FLAGS = _BASE_OPEN_FLAGS | getattr(os, "O_DIRECTORY", 0)
_SECURE_DIR_FD_SUPPORTED = bool(
    getattr(os, "O_NOFOLLOW", 0) and getattr(os, "O_DIRECTORY", 0) and os.open in os.supports_dir_fd
)


@dataclass(frozen=True)
class _ResolvedInstruction:
    path: str
    scope: str
    precedence: int
    sha256: str
    content: str
    byte_count: int


def _indexed_repository(app: AppContext, repo: str | None) -> tuple[str, Path]:
    repo_id, _ = _resolve_repo(app, repo)
    if repo_id is None:
        raise InstructionResolutionError("Current repository is not indexed")
    try:
        repositories = app.storage.list_repos()
    except Exception as exc:
        raise InstructionResolutionError("Could not resolve indexed repositories") from exc

    for repository in repositories:
        key = getattr(repository, "normalized_key", None)
        if not isinstance(key, str) or app.storage.lookup_repo_id(key) != repo_id:
            continue
        local_path = getattr(repository, "local_path", None)
        if not isinstance(local_path, str) or not local_path:
            break
        root = Path(local_path).expanduser()
        if not root.is_absolute():
            break
        try:
            root = root.resolve()
        except (OSError, RuntimeError) as exc:
            raise InstructionResolutionError(
                f"Repository root could not be resolved: {key}"
            ) from exc
        if not root.is_dir():
            break
        return key, root

    raise InstructionResolutionError("Repository has no accessible indexed root")


def _canonical_path(root: Path, path: str) -> str:
    try:
        return _canonical_relative_path(root, path)
    except ObservationError as exc:
        raise InstructionResolutionError(str(exc)) from exc


def _session_paths(
    app: AppContext,
    repository: str,
    explicit_path: str | None,
) -> tuple[str | None, list[str]]:
    snapshot = app.agent_state
    if snapshot is not None and snapshot.repository == repository:
        observed_active_path = snapshot.active_file
        changed_paths = list(snapshot.changed_paths)
    else:
        observed_active_path = None
        changed_paths = []
    active_path = explicit_path
    if active_path is None:
        active_path = observed_active_path
    if active_path is not None and not active_path.strip():
        raise InstructionResolutionError("Active path must not be empty")
    if active_path is not None and len(active_path) > MAX_AGENT_PATH_CHARS:
        raise InstructionResolutionError(
            f"Active path exceeds the {MAX_AGENT_PATH_CHARS}-character limit"
        )

    return active_path, changed_paths


def _scope_directories(root: Path, active_path: str | None) -> tuple[str | None, list[Path]]:
    if active_path is None:
        return None, [root]

    canonical = _canonical_path(root, active_path)
    target = root.joinpath(*PurePosixPath(canonical).parts)
    scope = target if target.is_dir() else target.parent
    try:
        scope = scope.resolve()
    except (OSError, RuntimeError) as exc:
        raise InstructionResolutionError(
            f"Active path could not be resolved safely: {active_path!r}"
        ) from exc
    if not scope.is_relative_to(root):
        raise InstructionResolutionError(f"Path escapes repository root: {active_path!r}")

    directories = [root]
    current = root
    for part in scope.relative_to(root).parts:
        current /= part
        directories.append(current)
    return canonical, directories


def _close_descriptor(descriptor: int) -> None:
    # Retrying a failed close can close an unrelated reused descriptor.
    with suppress(OSError):
        os.close(descriptor)


def _open_repository_root(root: Path) -> int | None:
    if os.name == "nt":
        return None
    if not _SECURE_DIR_FD_SUPPORTED:
        raise InstructionResolutionError(
            "Secure saved-worktree instruction reads are unsupported on this platform"
        )
    try:
        descriptor = os.open(root.anchor, _DIRECTORY_OPEN_FLAGS)
        for part in root.parts[1:]:
            next_descriptor = os.open(
                part,
                _DIRECTORY_OPEN_FLAGS,
                dir_fd=descriptor,
            )
            _close_descriptor(descriptor)
            descriptor = next_descriptor
    except OSError as exc:
        if "descriptor" in locals():
            _close_descriptor(descriptor)
        raise InstructionResolutionError("Repository root could not be opened safely") from exc
    try:
        metadata = os.fstat(descriptor)
    except OSError as exc:
        _close_descriptor(descriptor)
        raise InstructionResolutionError("Repository root could not be inspected safely") from exc
    if not stat.S_ISDIR(metadata.st_mode):
        _close_descriptor(descriptor)
        raise InstructionResolutionError("Repository root is not a directory")
    return descriptor


def _windows_final_path_from_handle(handle: int) -> str:
    import ctypes
    from ctypes import wintypes

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)  # type: ignore[attr-defined]
    get_final_path = kernel32.GetFinalPathNameByHandleW
    get_final_path.argtypes = [
        wintypes.HANDLE,
        wintypes.LPWSTR,
        wintypes.DWORD,
        wintypes.DWORD,
    ]
    get_final_path.restype = wintypes.DWORD
    volume_name_nt = 0x2
    length = get_final_path(handle, None, 0, volume_name_nt)
    if not length:
        raise OSError(
            ctypes.get_last_error(),  # type: ignore[attr-defined]
            "Could not resolve an open file handle",
        )
    buffer = ctypes.create_unicode_buffer(length + 1)
    written = get_final_path(handle, buffer, len(buffer), volume_name_nt)
    if not written or written >= len(buffer):
        raise OSError(
            ctypes.get_last_error(),  # type: ignore[attr-defined]
            "Could not resolve an open file handle",
        )
    return buffer.value.rstrip("\\")


def _windows_descriptor_final_path(descriptor: int) -> str:
    import msvcrt  # type: ignore[import-not-found]

    handle = msvcrt.get_osfhandle(descriptor)  # type: ignore[attr-defined]
    return _windows_final_path_from_handle(handle)


def _windows_directory_final_path(root: Path) -> str:
    import ctypes
    from ctypes import wintypes

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)  # type: ignore[attr-defined]
    create_file = kernel32.CreateFileW
    create_file.argtypes = [
        wintypes.LPCWSTR,
        wintypes.DWORD,
        wintypes.DWORD,
        wintypes.LPVOID,
        wintypes.DWORD,
        wintypes.DWORD,
        wintypes.HANDLE,
    ]
    create_file.restype = wintypes.HANDLE
    close_handle = kernel32.CloseHandle
    close_handle.argtypes = [wintypes.HANDLE]
    close_handle.restype = wintypes.BOOL
    share_read_write_delete = 0x1 | 0x2 | 0x4
    open_existing = 3
    backup_semantics = 0x02000000
    handle = create_file(
        str(root),
        0,
        share_read_write_delete,
        None,
        open_existing,
        backup_semantics,
        None,
    )
    if handle in (None, ctypes.c_void_p(-1).value):
        raise OSError(
            ctypes.get_last_error(),  # type: ignore[attr-defined]
            "Could not open the repository root handle",
        )
    try:
        return _windows_final_path_from_handle(handle)
    finally:
        close_handle(handle)


def _windows_path_is_contained(root: str, candidate: str) -> bool:
    return candidate == root or candidate.startswith(root + "\\")


def _open_windows_contained_file(root: Path, relative: Path, display_path: str) -> int:
    flags = _FILE_OPEN_FLAGS | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOINHERIT", 0)
    try:
        final_root = _windows_directory_final_path(root)
        descriptor = os.open(root.joinpath(*relative.parts), flags)
        final_path = _windows_descriptor_final_path(descriptor)
    except OSError as exc:
        if "descriptor" in locals():
            _close_descriptor(descriptor)
        raise InstructionResolutionError(
            f"Instruction source could not be opened safely: {display_path}"
        ) from exc
    if not _windows_path_is_contained(final_root, final_path):
        _close_descriptor(descriptor)
        raise InstructionResolutionError(
            f"Instruction source escapes repository root: {display_path}"
        )
    return descriptor


def _open_contained_file(
    root: Path,
    root_descriptor: int | None,
    relative: Path,
    display_path: str,
) -> int:
    parts = relative.parts
    if not parts:
        raise InstructionResolutionError(f"Invalid instruction source path: {display_path}")
    if root_descriptor is None:
        return _open_windows_contained_file(root, relative, display_path)

    try:
        directory_descriptor = os.dup(root_descriptor)
        for part in parts[:-1]:
            next_descriptor = os.open(
                part,
                _DIRECTORY_OPEN_FLAGS,
                dir_fd=directory_descriptor,
            )
            _close_descriptor(directory_descriptor)
            directory_descriptor = next_descriptor
        return os.open(parts[-1], _FILE_OPEN_FLAGS, dir_fd=directory_descriptor)
    except OSError as exc:
        raise InstructionResolutionError(
            f"Instruction source could not be opened safely: {display_path}"
        ) from exc
    finally:
        if "directory_descriptor" in locals():
            _close_descriptor(directory_descriptor)


def _read_bounded_file(
    root: Path,
    root_descriptor: int | None,
    relative: Path,
    display_path: str,
) -> bytes | None:
    descriptor = _open_contained_file(root, root_descriptor, relative, display_path)
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode):
            return None
        if metadata.st_size > MAX_INSTRUCTION_FILE_BYTES:
            raise InstructionResolutionError(
                f"Instruction source exceeds {MAX_INSTRUCTION_FILE_BYTES} bytes: {display_path}"
            )

        chunks: list[bytes] = []
        bytes_read = 0
        while bytes_read <= MAX_INSTRUCTION_FILE_BYTES:
            chunk = os.read(
                descriptor,
                min(64 * 1024, MAX_INSTRUCTION_FILE_BYTES + 1 - bytes_read),
            )
            if not chunk:
                break
            chunks.append(chunk)
            bytes_read += len(chunk)
        if bytes_read > MAX_INSTRUCTION_FILE_BYTES:
            raise InstructionResolutionError(
                f"Instruction source exceeds {MAX_INSTRUCTION_FILE_BYTES} bytes: {display_path}"
            )
        return b"".join(chunks)
    except OSError as exc:
        raise InstructionResolutionError(
            f"Instruction source could not be read: {display_path}"
        ) from exc
    finally:
        _close_descriptor(descriptor)


def _read_instruction(
    root: Path,
    root_descriptor: int | None,
    directory: Path,
    filename: str,
    depth: int,
) -> _ResolvedInstruction | None:
    candidate = directory / filename
    if not candidate.exists() and not candidate.is_symlink():
        return None
    display_path = candidate.relative_to(root).as_posix()
    try:
        resolved = candidate.resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise InstructionResolutionError(
            f"Instruction source could not be resolved safely: {display_path}"
        ) from exc
    if not resolved.is_relative_to(root):
        raise InstructionResolutionError(
            f"Instruction source escapes repository root: {display_path}"
        )
    raw = _read_bounded_file(root, root_descriptor, resolved.relative_to(root), display_path)
    if raw is None:
        return None
    try:
        content = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise InstructionResolutionError(
            f"Instruction source must be UTF-8: {display_path}"
        ) from exc

    scope = directory.relative_to(root).as_posix() or "."
    return _ResolvedInstruction(
        path=display_path,
        scope=scope,
        precedence=depth,
        sha256=hashlib.sha256(raw).hexdigest(),
        content=content,
        byte_count=len(raw),
    )


def _instruction_sources(root: Path, directories: list[Path]) -> list[_ResolvedInstruction]:
    sources: list[_ResolvedInstruction] = []
    total_bytes = 0
    root_descriptor = _open_repository_root(root)
    try:
        for depth, directory in enumerate(directories):
            for filename in INSTRUCTION_FILENAMES:
                source = _read_instruction(
                    root,
                    root_descriptor,
                    directory,
                    filename,
                    depth,
                )
                if source is None:
                    continue
                if len(sources) >= MAX_INSTRUCTION_SOURCES:
                    raise InstructionResolutionError(
                        "Applicable instruction sources exceed the "
                        f"{MAX_INSTRUCTION_SOURCES}-source limit"
                    )
                total_bytes += source.byte_count
                if total_bytes > MAX_INSTRUCTION_TOTAL_BYTES:
                    raise InstructionResolutionError(
                        "Applicable instruction sources exceed the "
                        f"{MAX_INSTRUCTION_TOTAL_BYTES}-byte aggregate limit"
                    )
                sources.append(source)
    finally:
        if root_descriptor is not None:
            _close_descriptor(root_descriptor)
    return sources


def _fair_allowances(lengths: list[int], budget: int) -> list[int]:
    allowances = [0] * len(lengths)
    pending = list(range(len(lengths)))
    remaining = budget
    while pending and remaining:
        share = max(1, remaining // len(pending))
        next_pending: list[int] = []
        for index in pending:
            needed = lengths[index] - allowances[index]
            take = min(needed, share, remaining)
            allowances[index] += take
            remaining -= take
            if allowances[index] < lengths[index]:
                next_pending.append(index)
            if remaining == 0:
                break
        pending = next_pending
    return allowances


def resolve_agent_instructions_impl(
    *,
    app: AppContext,
    path: str | None,
    repo: str | None,
    max_chars: int,
) -> InstructionContextResponse:
    """Resolve bounded saved-worktree instructions for one repository path."""
    if not 1 <= max_chars <= MAX_INSTRUCTION_CHARS:
        raise InstructionResolutionError(f"max_chars must be between 1 and {MAX_INSTRUCTION_CHARS}")

    repository, root = _indexed_repository(app, repo)
    active_path, changed_paths = _session_paths(app, repository, path)
    active_path, directories = _scope_directories(root, active_path)
    changed_paths = [_canonical_path(root, changed_path) for changed_path in changed_paths]
    resolved = _instruction_sources(root, directories)
    lengths = [len(source.content) for source in resolved]
    allowances = _fair_allowances(lengths, max_chars)
    sources = [
        InstructionSourceEntry(
            path=source.path,
            scope=source.scope,
            precedence=source.precedence,
            sha256=source.sha256,
            content=source.content[:allowance],
            full_chars=len(source.content),
            included_chars=allowance,
            truncated=allowance < len(source.content),
        )
        for source, allowance in zip(resolved, allowances, strict=True)
    ]
    total_chars = sum(lengths)
    included_chars = sum(allowances)
    return InstructionContextResponse(
        repository=repository,
        active_path=active_path,
        changed_paths=changed_paths,
        sources=sources,
        max_chars=max_chars,
        total_chars=total_chars,
        included_chars=included_chars,
        truncated=included_chars < total_chars,
    )


__all__ = [
    "DEFAULT_INSTRUCTION_CHARS",
    "MAX_INSTRUCTION_CHARS",
    "resolve_agent_instructions_impl",
]
