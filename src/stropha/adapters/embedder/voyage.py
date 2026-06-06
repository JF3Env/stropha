"""Voyage AI embedder adapter.

Active when a Voyage API key is available via explicit override,
``~/.stropha/config.yaml``, or ``VOYAGE_API_KEY``. The default model is
``voyage-4-large`` — Voyage's latest large-form embedder. Earlier
generations (``voyage-3-large``, ``voyage-code-3``) remain available via
``embedder.config.model: <id>`` in ``stropha.yaml`` or
``STROPHA_VOYAGE_EMBED_MODEL=<id>``. All emit Matryoshka vectors which we
truncate to ``dim`` (default 512).

Per-batch token budget
----------------------
Voyage caps each ``embed`` request at ~120K tokens *across* the whole
batch (after per-text truncation). The pipeline batches by chunk count
(``batch_size``), but real-world code chunks vary from ~50 to ~16K
tokens, so a 128-chunk batch routinely overshoots the limit. We split
incoming batches into sub-batches whose estimated token sum stays below
``max_tokens_per_batch`` and issue one API call per sub-batch.

Token estimation uses a cheap ``len(text) / chars_per_token`` heuristic
(no network, no extra deps). The default ``chars_per_token=3.5`` is
slightly conservative for English+code (Voyage docs quote ~4); combined
with a default budget of 96K (vs the 120K hard ceiling) it leaves ~20%
slack so multilingual / heavy-punctuation chunks don't tip us over.

These knobs do NOT change embedding output — only request shape — so
they are intentionally absent from ``adapter_id`` (no drift on tuning).
"""

from __future__ import annotations

import os
from collections.abc import Iterator, Sequence
from typing import Any

from pydantic import BaseModel, Field

from ...errors import EmbeddingError
from ...logging import get_logger
from ...pipeline.base import StageHealth
from ...pipeline.registry import register_adapter
from ...stages.embedder import EmbedderStage

log = get_logger(__name__)


class VoyageEmbedderConfig(BaseModel):
    """Config schema for the Voyage adapter."""

    model: str = Field(default="voyage-4-large")
    dim: int = Field(default=512, ge=64, le=4096)
    batch_size: int = Field(default=128, ge=1, le=512)
    api_key_env: str = Field(
        default="VOYAGE_API_KEY",
        description="Name of the env var that holds the API key.",
    )
    # Token-aware sub-batching. Voyage's documented per-request limit is
    # 120K tokens (after per-text truncation) for the v3 family. We
    # default to 96K to leave ~20% slack for the cheap chars/token
    # heuristic. NOT baked into ``adapter_id`` — request shape, not
    # output shape.
    max_tokens_per_batch: int = Field(
        default=96_000,
        ge=1024,
        le=120_000,
        description=(
            "Maximum estimated tokens per API call. Incoming batches "
            "above this threshold are split into sub-batches."
        ),
    )
    chars_per_token: float = Field(
        default=3.5,
        ge=1.0,
        le=10.0,
        description=(
            "Heuristic: average characters per token used to estimate "
            "batch size. Lower = more conservative (more API calls)."
        ),
    )


@register_adapter(stage="embedder", name="voyage")
class VoyageEmbedder(EmbedderStage):
    """Wraps the official ``voyageai`` SDK."""

    Config = VoyageEmbedderConfig

    def __init__(
        self,
        config: VoyageEmbedderConfig | None = None,
        *,
        api_key: str | None = None,
        model: str | None = None,
        dim: int | None = None,
    ) -> None:
        # Back-compat positional form: ``VoyageEmbedder(api_key=..., model=..., dim=...)``.
        if config is None:
            config = VoyageEmbedderConfig(
                model=model or VoyageEmbedderConfig.model_fields["model"].default,
                dim=dim if dim is not None else VoyageEmbedderConfig.model_fields["dim"].default,
            )
        elif api_key is not None or model is not None or dim is not None:
            raise TypeError("Pass either config= or legacy kwargs, not both")
        self._config = config
        self._api_key_override = api_key

        try:
            import voyageai  # type: ignore[import-untyped]
        except ImportError as exc:  # pragma: no cover
            raise EmbeddingError(
                "voyageai package is not installed. Run `uv sync`."
            ) from exc
        self._voyageai = voyageai
        self._client: object | None = None

    def _get_client(self) -> Any:
        # Lazy: the MCP server constructs every adapter at startup, but the
        # retrieval-only query path may never touch the embedder. Deferring
        # the API-key check + Client construction lets stropha-mcp start
        # without an active Voyage key when only serving an existing index.
        if self._client is not None:
            return self._client
        resolved_key = self._api_key_override or os.environ.get(self._config.api_key_env, "")
        if not resolved_key:
            try:
                from ...config import get_config

                resolved_key = get_config().secrets.voyage_api_key or ""
            except Exception:
                resolved_key = ""
        if not resolved_key:
            raise EmbeddingError(
                "Voyage API key is empty (no explicit key, config secret, "
                f"or env var {self._config.api_key_env!r} set)."
            )
        self._client = self._voyageai.Client(api_key=resolved_key)
        return self._client

    # ----- Stage contract -------------------------------------------------

    @property
    def stage_name(self) -> str:
        return "embedder"

    @property
    def adapter_name(self) -> str:
        return "voyage"

    @property
    def adapter_id(self) -> str:
        return f"voyage:{self._config.model}:{self._config.dim}"

    @property
    def config_schema(self) -> type[BaseModel]:
        return VoyageEmbedderConfig

    def health(self) -> StageHealth:
        # No-op probe. The API key is validated lazily on first embed call,
        # so we cannot promise the client will start — only that the adapter
        # is configured.
        return StageHealth(
            status="ready",
            message=f"voyageai client configured ({self._config.model}, dim={self._config.dim})",
            detail={"model": self._config.model, "dim": str(self._config.dim)},
        )

    # ----- Embedder surface ----------------------------------------------

    @property
    def model_name(self) -> str:
        return self._config.model

    @property
    def dim(self) -> int:
        return self._config.dim

    @property
    def batch_size(self) -> int:
        return self._config.batch_size

    def _estimate_tokens(self, text: str) -> int:
        """Cheap token estimate via ``len(text) / chars_per_token``.

        Always returns at least 1 so empty strings still consume a slot
        in the sub-batch (Voyage bills per item, not per token, for the
        empty case — and we want one-per-text monotonicity).
        """
        return max(1, int(len(text) / self._config.chars_per_token))

    def _iter_sub_batches(self, texts: Sequence[str]) -> Iterator[list[str]]:
        """Greedy first-fit packer.

        Yields lists of texts whose estimated token sum stays at or below
        ``max_tokens_per_batch``. A single text that exceeds the cap is
        emitted solo — Voyage's server-side per-text truncation will
        crop it before counting against the batch budget, so a solo
        oversized item is the right escape hatch (never silently drop).
        """
        cap = self._config.max_tokens_per_batch
        batch: list[str] = []
        batch_tokens = 0
        for text in texts:
            est = self._estimate_tokens(text)
            # When the current item alone is already at or above the cap,
            # flush whatever we have and ship the oversized item solo.
            if est >= cap:
                if batch:
                    yield batch
                    batch = []
                    batch_tokens = 0
                yield [text]
                continue
            if batch and batch_tokens + est > cap:
                yield batch
                batch = [text]
                batch_tokens = est
            else:
                batch.append(text)
                batch_tokens += est
        if batch:
            yield batch

    def _embed(self, texts: Sequence[str], input_type: str) -> list[list[float]]:
        client = self._get_client()
        embeddings: list[list[float]] = []
        for sub_batch in self._iter_sub_batches(texts):
            try:
                result = client.embed(
                    sub_batch,
                    model=self._config.model,
                    input_type=input_type,
                    output_dimension=self._config.dim,
                )
            except Exception as exc:
                raise EmbeddingError(f"Voyage embed failed: {exc}") from exc
            embeddings.extend(result.embeddings)
        if embeddings and len(embeddings[0]) != self._config.dim:
            log.warning(
                "voyage.dim_mismatch",
                expected=self._config.dim,
                actual=len(embeddings[0]),
                model=self._config.model,
            )
        return embeddings

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        return self._embed(texts, input_type="document")

    def embed_query(self, text: str) -> list[float]:
        # Single query is always a single sub-batch (well under any cap).
        return self._embed([text], input_type="query")[0]
