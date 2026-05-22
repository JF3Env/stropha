"""EnricherStage protocol — transform a chunk before embedding.

Phase 1 introduces two adapters:

- ``noop``         — identity (default; preserves current behavior).
- ``hierarchical`` — prepends a small skeleton of the parent chunk
                     (class header for a method, file header for a class, …).

Future adapters (Phase 2/3): ``ollama``, ``anthropic``, ``mlx``, ``openai`` —
LLM-generated contextual prefixes per the Anthropic "Contextual Retrieval"
technique (``docs/architecture/stropha-system.md`` §3.4).

Drift is detected via the ``adapter_id`` stored on each chunk row
(``chunks.enricher_id``). When the active enricher's id differs from the
stored id, the pipeline re-embeds the chunk transparently — no
``--rebuild`` required.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol, runtime_checkable

from pydantic import BaseModel

from ..models import Chunk
from ..pipeline.base import StageContext, StageHealth


@runtime_checkable
class EnricherStage(Protocol):
    """Returns the text that should be fed to the embedder for ``chunk``.

    The original ``chunk.content`` is preserved on the chunk row (so a
    ``Read`` of a search hit always shows source-truth). Only the
    embedded representation changes.
    """

    # ----- Stage contract --------------------------------------------------

    @property
    def stage_name(self) -> str: ...

    @property
    def adapter_name(self) -> str: ...

    @property
    def adapter_id(self) -> str: ...

    @property
    def config_schema(self) -> type[BaseModel]: ...

    def health(self) -> StageHealth: ...

    # ----- Enricher-specific surface --------------------------------------

    def enrich(self, chunk: Chunk, ctx: StageContext) -> str:
        """Return the text to embed.

        MUST be deterministic for a given ``(chunk, ctx, adapter_config)``
        triplet — non-determinism breaks the drift-detection cache.

        MAY consult ``ctx.parent_chunk`` and ``ctx.file_content``. MUST
        tolerate both being ``None``.
        """


def enrich_with_status(
    enricher: "EnricherStage", chunk: Chunk, ctx: StageContext
) -> tuple[str, bool]:
    """Run ``enricher.enrich`` and return ``(text, success)``.

    Adapters that can *fail gracefully* (network LLMs, rate-limited cloud
    backends) override this by defining an ``enrich_with_status`` method
    on the adapter — the helper dispatches to that override when present.

    The default contract is "always succeeded" because the legacy
    enrichers (``noop``, ``hierarchical``, ``graph-aware``) are pure
    transformations with no failure mode.

    The pipeline uses the ``success`` bit to decide whether to write the
    result into the ``enrichments`` cache. ``success=False`` means the
    returned text is a *fallback* (typically raw content) and caching it
    would poison future re-index runs by marking failed chunks as already
    enriched.
    """
    impl = getattr(enricher, "enrich_with_status", None)
    if callable(impl):
        return impl(chunk, ctx)
    return enricher.enrich(chunk, ctx), True


def enrich_batch_with_status(
    enricher: "EnricherStage",
    chunks: Sequence[Chunk],
    contexts: Sequence[StageContext],
) -> list[tuple[str, bool]]:
    """Run a batched enrichment if the adapter supports it; fall back per-chunk.

    Adapters that benefit from batching (typically LLM-backed enrichers
    that can pack N chunks into a single prompt + structured JSON response)
    implement ``enrich_batch(chunks, contexts) -> list[tuple[str, bool]]``.
    The helper dispatches to that override when present; otherwise it
    loops over :func:`enrich_with_status` so callers can use one code
    path regardless of adapter capability.

    Returns a list aligned 1:1 with ``chunks``.
    """
    if len(chunks) != len(contexts):
        raise ValueError(
            f"chunks ({len(chunks)}) and contexts ({len(contexts)}) "
            "must have the same length"
        )
    impl = getattr(enricher, "enrich_batch", None)
    if callable(impl):
        result = impl(chunks, contexts)
        # Defensive: validate shape so a buggy adapter can't silently
        # corrupt the embedding pipeline.
        if len(result) != len(chunks):
            raise ValueError(
                f"{enricher.adapter_name}.enrich_batch returned "
                f"{len(result)} results for {len(chunks)} chunks"
            )
        return list(result)
    return [
        enrich_with_status(enricher, chunk, ctx)
        for chunk, ctx in zip(chunks, contexts, strict=True)
    ]
