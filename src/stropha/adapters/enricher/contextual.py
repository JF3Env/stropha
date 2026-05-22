"""Contextual Prefix Enricher — LLM-generated semantic description prepended
to chunks before embedding.

Based on Anthropic's "Contextual Retrieval" technique (Sept 2024) which showed
+35% recall improvement by prepending context descriptions to chunks.

Unlike the simpler `ollama` enricher (one-line summary), this enricher generates
a richer contextual description that includes:
- What this code does
- Where it fits in the codebase architecture
- Related concepts and components

The description is prepended to the embedding_text, not the stored content,
so the original code is preserved but the embedding captures richer semantics.

Example:
    Original chunk:
        def submit_answer(self, exercise_id: ExerciseId, answer: Answer): ...

    After enrichment:
        [Context: This method belongs to StudyService, the application service
        layer for study sessions. It handles answer submission in the Phase 1
        acquisition flow, updating streaks and triggering FSRS calculation
        when mastery threshold is reached.]

        def submit_answer(self, exercise_id: ExerciseId, answer: Answer): ...

Backend selection (automatic):
- MLX (Apple Silicon): fastest, no daemon needed
- Ollama: cross-platform fallback

Failure semantics: Any error falls back to raw content (same as ollama enricher).
The index keeps building; warnings appear in logs. ``enrich_with_status``
returns ``success=False`` on the fallback path so the pipeline does NOT cache
the failed result — the next index run will retry the chunk.

Batched enrichment (Fix 3): :meth:`enrich_batch` packs N chunks into a single
LLM call by asking the model to return a JSON array of descriptions. Effective
throughput multiplies by N under the same rate-limit budget. Enabled via
``inference.enricher_batch_size: N`` in ``stropha.yaml`` (default 1 = per-chunk).
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Sequence

from pydantic import BaseModel, Field

from ...logging import get_logger
from ...models import Chunk
from ...pipeline.base import StageContext, StageHealth
from ...pipeline.registry import register_adapter
from ...stages.enricher import EnricherStage

log = get_logger(__name__)

_DEFAULT_PROMPT = """\
Write a dense 2-3 sentence semantic description of this code chunk for retrieval.
Cover:
- What this code does (functionality)
- Where it fits (layer, module, responsibility)
- Related concepts or components it interacts with

Keep it concrete. Avoid filler words and meta-commentary.

Code:
```{language}
{content}
```

File: {rel_path}
Symbol: {symbol}

Output: ONLY the description, starting with "This...". No code, no preamble, no markdown.
"""
# Prompt v2 (2026-05): tightened wording for brevity + output discipline. The
# wording change also forces the adapter_id (prompt_hash) to flip, which
# invalidates any chunks whose enrichment was cached with raw content during
# the v1 era of DeepSeek timeouts (before retry+backoff landed). On the next
# `stropha index`, drift detection re-enriches every chunk against the new
# adapter_id — failed v1 chunks get a clean second attempt.

_CONTEXT_PREFIX = "[Context: {description}]\n\n"

_BATCH_PROMPT_HEADER = """\
Write dense 2-3 sentence semantic descriptions for the {n} code chunks below.

OUTPUT FORMAT — STRICT JSON ONLY:
A JSON array of objects with exactly two keys: "id" (integer matching the
CHUNK number) and "description" (string, starts with "This"). Example:

[{{"id":1,"description":"This..."}},{{"id":2,"description":"This..."}}]

Rules:
- One entry per chunk, in order, IDs 1..{n}.
- Each description covers what the code does, where it fits, and related concepts.
- No code, no markdown, no preamble, no code fences. JSON ONLY.

"""

_BATCH_CHUNK_FORMAT = """\
CHUNK {id}
File: {rel_path}
Symbol: {symbol}
Language: {language}
```{language}
{content}
```

"""

# Token allowance per chunk on top of the structured-JSON envelope.
# ``max_tokens`` is multiplied by batch size + this constant so the model
# has enough budget for N descriptions plus the JSON wrapping.
_BATCH_ENVELOPE_TOKENS = 64


class ContextualEnricherConfig(BaseModel):
    prompt_template: str = Field(
        default=_DEFAULT_PROMPT,
        description=(
            "Format string with placeholders: {content}, {language}, "
            "{rel_path}, {symbol}"
        ),
    )
    max_content_chars: int = Field(
        default=4000,
        ge=500,
        le=16000,
        description="Max chars of content to send to LLM (truncates long chunks).",
    )
    temperature: float = Field(
        default=0.0,
        ge=0.0,
        le=2.0,
        description="LLM temperature. 0 for deterministic output.",
    )
    max_tokens: int = Field(
        default=256,
        ge=64,
        le=1024,
        description="Max tokens for the generated description.",
    )
    max_description_chars: int = Field(
        default=500,
        ge=100,
        le=2000,
        description="Max chars for the generated description.",
    )


@register_adapter(stage="enricher", name="contextual")
class ContextualEnricher(EnricherStage):
    """LLM-generated contextual descriptions prepended to chunks.

    Implements Anthropic's Contextual Retrieval technique for improved
    embedding quality. The description provides semantic context that
    helps the embedding model understand the chunk's role in the codebase.

    Uses the unified inference backend (MLX on Apple Silicon, Ollama fallback).
    """

    Config = ContextualEnricherConfig

    def __init__(self, config: ContextualEnricherConfig | None = None) -> None:
        self._config = config or ContextualEnricherConfig()

    @property
    def stage_name(self) -> str:
        return "enricher"

    @property
    def adapter_name(self) -> str:
        return "contextual"

    @property
    def adapter_id(self) -> str:
        # Hash the prompt so edits invalidate the enrichment cache
        prompt_hash = hashlib.sha256(
            self._config.prompt_template.encode()
        ).hexdigest()[:8]
        return (
            f"contextual"
            f":t={self._config.temperature}"
            f":p={prompt_hash}"
        )

    @property
    def config_schema(self) -> type[BaseModel]:
        return ContextualEnricherConfig

    def health(self) -> StageHealth:
        """Check inference backend availability."""
        from ...inference import get_backend

        backend = get_backend()
        is_healthy, message = backend.health_check()

        if is_healthy:
            return StageHealth(
                status="ready",
                message=f"contextual enricher using {backend.name} backend",
            )
        return StageHealth(
            status="warning",
            message=f"Inference backend issue: {message}",
        )

    def enrich(self, chunk: Chunk, ctx: StageContext) -> str:
        """Generate contextual description and prepend to content.

        Returns the original content if LLM call fails (graceful fallback).

        Pipeline integration: prefer :meth:`enrich_with_status` over this
        method — it surfaces the success bit the pipeline needs to avoid
        caching failed enrichments.
        """
        return self.enrich_with_status(chunk, ctx)[0]

    def enrich_with_status(
        self, chunk: Chunk, ctx: StageContext
    ) -> tuple[str, bool]:
        """Generate contextual description and report success.

        Returns:
            ``(embed_text, success)``. ``success=False`` means the LLM
            call failed (rate limit, network, malformed response) and the
            text is a raw-content fallback. The pipeline MUST NOT cache
            this result — otherwise the next index run would treat the
            failed chunk as already enriched and skip retry.
        """
        description = self._generate_context(chunk)
        if not description:
            return chunk.content, False
        prefix = _CONTEXT_PREFIX.format(description=description)
        return f"{prefix}{chunk.content}", True

    def enrich_batch(
        self,
        chunks: Sequence[Chunk],
        contexts: Sequence[StageContext],
    ) -> list[tuple[str, bool]]:
        """Enrich ``N`` chunks in a single LLM call via structured JSON.

        Drives down the per-chunk request count by N. The model receives
        a numbered list of chunks and must return a JSON array of
        ``{"id": int, "description": str}`` objects. Each chunk gets its
        own ``(text, success)`` tuple back in input order.

        Failure modes (each handled gracefully → ``success=False``):

        - Whole call fails (LLM error / rate limit exhausted /
          timeout): all N chunks marked failed, fall back to raw content.
        - Malformed JSON (no array found, unparseable): same as above.
        - Partial response (some IDs missing from the array): only the
          missing chunks marked failed; the rest succeed.
        - Empty/whitespace description for a given ID: marked failed.

        Single-chunk batches bypass the structured-JSON envelope and use
        the per-chunk path — same wire shape as before, no behaviour
        change for ``batch_size=1``.
        """
        if not chunks:
            return []
        if len(chunks) != len(contexts):
            raise ValueError(
                f"chunks ({len(chunks)}) and contexts ({len(contexts)}) "
                "must have the same length"
            )
        if len(chunks) == 1:
            # Avoid the batched-prompt overhead for a single chunk —
            # also keeps the wire format identical to the per-chunk path
            # for backward-compat assertions in existing tests.
            return [self.enrich_with_status(chunks[0], contexts[0])]

        prompt = self._build_batch_prompt(chunks)
        # Token budget: per-chunk allowance + JSON envelope overhead.
        max_tokens = (
            self._config.max_tokens * len(chunks) + _BATCH_ENVELOPE_TOKENS
        )

        from ...inference import generate as _generate

        raw = _generate(
            prompt,
            max_tokens=max_tokens,
            temperature=self._config.temperature,
        )

        if not raw:
            # LLM failed entirely (rate limit, timeout, missing key) —
            # fail every chunk so the pipeline retries next run.
            log.warning(
                "contextual.enrich_batch_failed",
                reason="empty response",
                batch_size=len(chunks),
            )
            return [(c.content, False) for c in chunks]

        descriptions_by_id = self._parse_batch_response(raw, len(chunks))
        if not descriptions_by_id:
            log.warning(
                "contextual.enrich_batch_failed",
                reason="no JSON array found in response",
                batch_size=len(chunks),
                preview=raw[:200],
            )
            return [(c.content, False) for c in chunks]

        results: list[tuple[str, bool]] = []
        for idx, chunk in enumerate(chunks):
            # IDs in the prompt are 1-indexed (1..N).
            raw_desc = descriptions_by_id.get(idx + 1)
            cleaned = self._clean_description(raw_desc) if raw_desc else None
            if not cleaned:
                results.append((chunk.content, False))
                continue
            prefix = _CONTEXT_PREFIX.format(description=cleaned)
            results.append((f"{prefix}{chunk.content}", True))
        return results

    # ----- internals --------------------------------------------------------

    def _build_batch_prompt(self, chunks: Sequence[Chunk]) -> str:
        """Render the structured-JSON batch prompt for N chunks."""
        parts = [_BATCH_PROMPT_HEADER.format(n=len(chunks))]
        for idx, chunk in enumerate(chunks, start=1):
            content = chunk.content[: self._config.max_content_chars]
            parts.append(
                _BATCH_CHUNK_FORMAT.format(
                    id=idx,
                    rel_path=chunk.rel_path,
                    symbol=chunk.symbol or "(anonymous)",
                    language=chunk.language or "unknown",
                    content=content,
                )
            )
        return "".join(parts)

    @staticmethod
    def _parse_batch_response(raw: str, expected: int) -> dict[int, str]:
        """Parse the LLM's JSON array into ``{id: description}``.

        Tolerates common output drift: markdown code fences, leading
        prose, trailing prose. Returns an empty dict when no array can
        be recovered — caller treats that as "whole batch failed".
        """
        if not raw:
            return {}
        # Strip markdown code fences if present.
        text = raw.strip()
        # Slice from the first '[' to the last ']'. JSON parsers reject
        # surrounding prose, but the array body itself usually parses.
        first = text.find("[")
        last = text.rfind("]")
        if first == -1 or last == -1 or last <= first:
            return {}
        candidate = text[first : last + 1]
        try:
            arr = json.loads(candidate)
        except json.JSONDecodeError:
            # One retry: try to repair trailing commas (common LLM tic).
            repaired = re.sub(r",\s*([}\]])", r"\1", candidate)
            try:
                arr = json.loads(repaired)
            except json.JSONDecodeError:
                return {}

        if not isinstance(arr, list):
            return {}

        result: dict[int, str] = {}
        for entry in arr:
            if not isinstance(entry, dict):
                continue
            entry_id = entry.get("id")
            entry_desc = entry.get("description")
            if not isinstance(entry_id, int):
                continue
            if not isinstance(entry_desc, str):
                continue
            # Drop IDs outside the expected 1..N range to defend against
            # hallucinated extras.
            if entry_id < 1 or entry_id > expected:
                continue
            result[entry_id] = entry_desc
        return result

    def _clean_description(self, raw: str | None) -> str | None:
        """Normalize an LLM-generated description.

        Strips code fences, takes the first paragraph, normalizes
        whitespace, and truncates to ``max_description_chars``. Returns
        ``None`` when the result is empty.
        """
        if not raw:
            return None
        cleaned = raw.replace("```", "").strip()
        if "\n\n" in cleaned:
            cleaned = cleaned.split("\n\n")[0]
        cleaned = " ".join(cleaned.split())
        if not cleaned:
            return None
        if len(cleaned) > self._config.max_description_chars:
            cleaned = cleaned[: self._config.max_description_chars - 3] + "..."
        return cleaned

    def _generate_context(self, chunk: Chunk) -> str | None:
        """Generate contextual description using the inference backend.

        Uses MLX on Apple Silicon, falls back to Ollama on other platforms.
        Returns None on any failure (graceful).
        """
        # Build prompt with chunk metadata
        content = chunk.content[: self._config.max_content_chars]
        prompt = self._config.prompt_template.format(
            content=content,
            language=chunk.language or "unknown",
            rel_path=chunk.rel_path,
            symbol=chunk.symbol or "(anonymous)",
        )

        # Use unified inference backend
        from ...inference import generate

        text = generate(
            prompt,
            max_tokens=self._config.max_tokens,
            temperature=self._config.temperature,
        )

        return self._clean_description(text)
