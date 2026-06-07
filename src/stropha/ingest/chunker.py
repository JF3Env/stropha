"""Chunker dispatcher.

For each `SourceFile`, picks a language-specific chunker and yields `Chunk`s.
Languages without an AST chunker fall back to file-level chunking
(`FallbackChunker`).
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator

from ..errors import ChunkerError
from ..logging import get_logger
from ..models import Chunk, SourceFile
from .chunkers import FallbackChunker, LanguageChunker
from .chunkers.ast_generic import AstGenericChunker
from .chunkers.base import make_chunk_id
from .chunkers.gherkin import GherkinChunker
from .chunkers.markdown import MarkdownChunker
from .chunkers.vue import VueChunker

log = get_logger(__name__)


def _build_registry() -> dict[str, LanguageChunker]:
    return {
        # Languages with tree-sitter-language-pack structure extraction:
        "java": AstGenericChunker("java"),
        "typescript": AstGenericChunker("typescript"),
        "javascript": AstGenericChunker("javascript"),
        "python": AstGenericChunker("python"),
        "rust": AstGenericChunker("rust"),
        "go": AstGenericChunker("go"),
        "kotlin": AstGenericChunker("kotlin"),
        # Custom chunkers:
        "vue": VueChunker(),
        "markdown": MarkdownChunker(),
        "gherkin": GherkinChunker(),
    }


class Chunker:
    """Public entry point. Replaces Phase 0's FileChunker."""

    def __init__(self) -> None:
        self._registry = _build_registry()
        self._fallback = FallbackChunker()

    def chunk(
        self,
        files: Iterable[SourceFile],
        *,
        repo_key: str | None = None,
    ) -> Iterator[Chunk]:
        """Chunk every file. When ``repo_key`` is provided, every emitted
        chunk's ``chunk_id`` is re-derived to include the repo discriminator
        so two repos with identical files do not collide on the global
        ``chunks.chunk_id`` UNIQUE constraint.
        """
        for sf in files:
            try:
                content = sf.path.read_text(encoding="utf-8", errors="replace")
            except OSError as exc:
                raise ChunkerError(f"Cannot read {sf.rel_path}: {exc}") from exc
            if not content.strip():
                continue
            chunker = self._registry.get(sf.language, self._fallback)
            try:
                emitted = chunker.chunk(sf, content)
            except Exception as exc:
                # Per the spec's "graceful failure" principle: log + fallback,
                # never poison the whole index because of one weird file.
                log.warning(
                    "chunker.error_fallback",
                    path=sf.rel_path,
                    language=sf.language,
                    error=str(exc),
                )
                emitted = self._fallback.chunk(sf, content)
            yield from self._apply_repo_key(list(emitted), repo_key)

    @staticmethod
    def _apply_repo_key(
        emitted: list[Chunk], repo_key: str | None
    ) -> Iterator[Chunk]:
        """Namespace a file's chunks with ``repo_key``, rewriting BOTH
        ``chunk_id`` and ``parent_chunk_id``.

        Sub-chunkers build ``parent_chunk_id`` from the un-namespaced
        ``make_chunk_id``. Rewriting only ``chunk_id`` (the old behaviour)
        left every child's ``parent_chunk_id`` dangling, silently disabling
        the hierarchical enricher and recursive-merge parent promotion.
        Parents live in the same file, so the per-file id map is complete.
        """
        if not repo_key:
            yield from emitted
            return
        id_map: dict[str, str] = {
            c.chunk_id: make_chunk_id(
                c.rel_path, c.start_line, c.end_line, c.content_hash,
                repo_key=repo_key,
            )
            for c in emitted
        }
        for c in emitted:
            new_id = id_map[c.chunk_id]
            new_parent = (
                id_map.get(c.parent_chunk_id, c.parent_chunk_id)
                if c.parent_chunk_id
                else None
            )
            if new_id == c.chunk_id and new_parent == c.parent_chunk_id:
                yield c
            else:
                yield c.model_copy(
                    update={"chunk_id": new_id, "parent_chunk_id": new_parent}
                )
