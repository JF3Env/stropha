"""Nested-git walker — discovers nested Git repositories under a root.

Use for monorepos that vendor multiple git submodules / vendored deps,
or for an `~/sources/` umbrella directory containing many independent
repos. Each nested git repo is processed by the standard ``git-ls-files``
walker, so per-repo `.gitignore` is honoured automatically.

Output ``SourceFile.rel_path`` is *relative to the umbrella root* so the
caller still sees a single coherent file tree. The actual repo identity
is preserved in the chunk's ``repo_id`` foreign key (the pipeline calls
``git_meta.detect`` on each nested repo independently).
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

from pydantic import BaseModel, Field

from ...ingest.walker import Walker
from ...logging import get_logger
from ...models import SourceFile
from ...pipeline.base import StageHealth
from ...pipeline.registry import register_adapter
from ...stages.walker import WalkerStage

log = get_logger(__name__)

_SKIP_DIRS = frozenset({".venv", "node_modules", "target", "build", "dist"})


class NestedGitWalkerConfig(BaseModel):
    max_file_bytes: int = Field(default=524_288, ge=1024)
    max_depth: int = Field(
        default=4,
        ge=1,
        le=10,
        description="Maximum directory depth to scan for nested Git repositories.",
    )
    follow_root_first: bool = Field(
        default=True,
        description=(
            "When the umbrella root is itself a git repo, walk it before "
            "descending into nested submodules."
        ),
    )


@register_adapter(stage="walker", name="nested-git")
class NestedGitWalker(WalkerStage):
    """Discover every nested Git repository and walk each as its own repo."""

    Config = NestedGitWalkerConfig

    def __init__(self, config: NestedGitWalkerConfig | None = None) -> None:
        self._config = config or NestedGitWalkerConfig()

    @property
    def stage_name(self) -> str:
        return "walker"

    @property
    def adapter_name(self) -> str:
        return "nested-git"

    @property
    def adapter_id(self) -> str:
        root_first = str(self._config.follow_root_first).lower()
        return (
            f"nested-git:depth={self._config.max_depth}:"
            f"max={self._config.max_file_bytes}:root-first={root_first}"
        )

    @property
    def config_schema(self) -> type[BaseModel]:
        return NestedGitWalkerConfig

    def health(self) -> StageHealth:
        return StageHealth(
            status="ready",
            message=f"nested-git walker (max depth={self._config.max_depth})",
        )

    def discover(self, repo: Path) -> Iterable[SourceFile]:
        seen: set[Path] = set()
        nested = list(self._find_nested_repos(repo, max_depth=self._config.max_depth))
        log.info(
            "nested_git_walker.discovered",
            root=str(repo),
            nested_count=len(nested),
            paths=[str(p.relative_to(repo)) for p in nested[:10]],
        )

        # Optionally walk the umbrella root first
        if self._config.follow_root_first and self._has_git_marker(repo):
            yield from self._walk_one(repo, root=repo, seen=seen)

        for sub in nested:
            yield from self._walk_one(sub, root=repo, seen=seen)

    def discover_paths(self, repo: Path, rel_paths: Iterable[str]) -> Iterable[SourceFile]:
        root = repo.resolve()
        seen: set[Path] = set()
        for rel_path in rel_paths:
            candidate = Path(rel_path)
            if candidate.is_absolute() or ".." in candidate.parts:
                continue
            absolute = (root / candidate).absolute()
            try:
                absolute.relative_to(root)
            except ValueError:
                continue
            if not self._is_within_root(root, absolute):
                continue
            sub_repo = self._nearest_repo_root(absolute.parent, root)
            if sub_repo is None:
                continue
            if not self._repo_is_admitted(root, sub_repo, max_depth=self._config.max_depth):
                continue
            nested_rel = absolute.relative_to(sub_repo).as_posix()
            walker = Walker(sub_repo, max_file_bytes=self._config.max_file_bytes)
            for source in walker.discover_paths([nested_rel]):
                rebased = self._rebase_source(source, sub_repo=sub_repo, root=root)
                if rebased.path in seen:
                    continue
                seen.add(rebased.path)
                yield rebased

    # ------------------------------------------------------------------ helpers

    def _find_nested_repos(self, root: Path, *, max_depth: int) -> Iterable[Path]:
        """BFS for nested Git work trees under ``root``.

        Skips the root's own ``.git`` (handled by ``follow_root_first``).
        """
        from collections import deque

        queue: deque[tuple[Path, int]] = deque([(root, 0)])
        while queue:
            d, depth = queue.popleft()
            if d != root and self._repo_is_admitted(root, d, max_depth=max_depth):
                yield d
            if depth >= max_depth:
                continue
            try:
                children = list(d.iterdir())
            except (OSError, PermissionError):
                continue
            for child in children:
                if child.name == ".git":
                    continue
                if not child.is_dir():
                    continue
                if not self._is_within_root(root, child):
                    continue
                if child.name in _SKIP_DIRS:
                    continue
                queue.append((child, depth + 1))

    def _repo_is_admitted(self, root: Path, repo: Path, *, max_depth: int) -> bool:
        if not self._is_within_root(root, repo):
            return False
        if not self._has_git_marker(repo):
            return False
        if repo == root:
            return self._config.follow_root_first
        relative = repo.relative_to(root)
        return len(relative.parts) <= max_depth and not (_SKIP_DIRS & set(relative.parts))

    @staticmethod
    def _is_within_root(root: Path, candidate: Path) -> bool:
        try:
            candidate.resolve().relative_to(root.resolve())
        except (OSError, RuntimeError, ValueError):
            return False
        return True

    @staticmethod
    def _has_git_marker(repo: Path) -> bool:
        return (repo / ".git").exists()

    @staticmethod
    def _nearest_repo_root(start: Path, root: Path) -> Path | None:
        current = start
        while current.is_relative_to(root):
            if NestedGitWalker._has_git_marker(current):
                return current
            if current == root:
                break
            current = current.parent
        return None

    def _walk_one(
        self,
        sub_repo: Path,
        *,
        root: Path,
        seen: set[Path],
    ) -> Iterable[SourceFile]:
        """Walk one nested repo, rebasing rel_path against the umbrella root."""
        try:
            walker = Walker(sub_repo, max_file_bytes=self._config.max_file_bytes)
        except Exception as exc:  # pragma: no cover — defensive
            log.warning("nested_git_walker.walker_init_failed", sub=str(sub_repo), error=str(exc))
            return
        for sf in walker.discover():
            rebased = self._rebase_source(sf, sub_repo=sub_repo, root=root)
            if rebased.path in seen:
                continue
            seen.add(rebased.path)
            yield rebased

    @staticmethod
    def _rebase_source(source: SourceFile, *, sub_repo: Path, root: Path) -> SourceFile:
        absolute = (sub_repo / source.rel_path).resolve()
        try:
            rel_path = absolute.relative_to(root.resolve()).as_posix()
        except ValueError:
            rel_path = Path(source.rel_path).as_posix()
        return source.model_copy(update={"rel_path": rel_path, "path": absolute})
