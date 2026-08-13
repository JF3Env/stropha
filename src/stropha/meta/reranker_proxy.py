"""Trusted parent proxy for executing generated rerankers in a sandbox."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from ..models import SearchHit
from ..pipeline.base import StageHealth
from .sandbox import MAX_INPUT_BYTES, SandboxError, SandboxResult, SubprocessSandbox

MAX_RERANKER_QUERY_BYTES = 16 * 1024
MAX_RERANKER_HITS = 500
MAX_RERANKER_TOP_K = 200


class SandboxedRerankerConfig(BaseModel):
    """Empty protocol config; candidate config remains in its artifact directory."""


class SandboxedRerankerProxy:
    """Run one generated reranker in a fresh subprocess for every call."""

    Config = SandboxedRerankerConfig

    def __init__(
        self,
        candidate_dir: Path,
        *,
        adapter_name: str,
        sandbox_factory: Callable[[], SubprocessSandbox] = SubprocessSandbox,
    ) -> None:
        if not adapter_name:
            raise ValueError("adapter_name must not be empty")
        self._candidate_dir = candidate_dir
        self._adapter_name = adapter_name
        self._sandbox_factory = sandbox_factory

    @property
    def stage_name(self) -> str:
        return "reranker"

    @property
    def adapter_name(self) -> str:
        return self._adapter_name

    @property
    def adapter_id(self) -> str:
        return f"reranker:sandboxed:{self._adapter_name}:{self._candidate_dir.name}"

    @property
    def config_schema(self) -> type[BaseModel]:
        return SandboxedRerankerConfig

    def health(self) -> StageHealth:
        return StageHealth(
            status="ready",
            message="generated reranker executes in a fresh subprocess sandbox",
        )

    def rerank(
        self,
        query: str,
        hits: list[SearchHit],
        *,
        top_k: int = 10,
    ) -> list[SearchHit]:
        input_ids = [hit.chunk_id for hit in hits]
        if len(input_ids) != len(set(input_ids)):
            raise SandboxError("reranker input contains duplicate chunk IDs")
        payload = _serialize_request(query, hits, top_k=top_k)
        if not hits:
            return []
        result = self._sandbox_factory().run(
            self._candidate_dir,
            job="reranker_apply",
            input_data=payload,
        )
        returned_ids = _validate_result(result, top_k=top_k)

        hits_by_id = dict(zip(input_ids, hits, strict=True))
        unknown = [chunk_id for chunk_id in returned_ids if chunk_id not in hits_by_id]
        if unknown:
            raise SandboxError(f"reranker output contains unknown chunk ID: {unknown[0]!r}")
        return [hits_by_id[chunk_id] for chunk_id in returned_ids]


def _serialize_request(query: str, hits: list[SearchHit], *, top_k: int) -> str:
    if not isinstance(query, str):
        raise SandboxError("reranker query must be a string")
    if len(query.encode("utf-8")) > MAX_RERANKER_QUERY_BYTES:
        raise SandboxError(f"reranker query exceeds {MAX_RERANKER_QUERY_BYTES} bytes")
    if (
        isinstance(top_k, bool)
        or not isinstance(top_k, int)
        or not 1 <= top_k <= MAX_RERANKER_TOP_K
    ):
        raise SandboxError(f"reranker top_k must be between 1 and {MAX_RERANKER_TOP_K}")
    if len(hits) > MAX_RERANKER_HITS:
        raise SandboxError(f"reranker input exceeds {MAX_RERANKER_HITS} hits")

    wire_hits = [
        {
            "rank": hit.rank,
            "score": hit.score,
            "rel_path": hit.rel_path,
            "language": hit.language,
            "kind": hit.kind,
            "symbol": hit.symbol,
            "start_line": hit.start_line,
            "end_line": hit.end_line,
            "snippet": hit.snippet,
            "chunk_id": hit.chunk_id,
            "streams": hit.streams,
        }
        for hit in hits
    ]
    encoded = json.dumps(
        {"query": query, "top_k": top_k, "hits": wire_hits},
        ensure_ascii=True,
        separators=(",", ":"),
    )
    if len(encoded.encode("utf-8")) > MAX_INPUT_BYTES:
        raise SandboxError(f"reranker input exceeds {MAX_INPUT_BYTES} serialized bytes")
    return encoded


def _validate_result(result: SandboxResult, *, top_k: int) -> list[str]:
    if not result.succeeded:
        detail = result.error or result.stderr_tail or result.stdout or "sandbox job failed"
        raise SandboxError(f"reranker sandbox failed: {detail}")

    payload: dict[str, Any] = result.payload
    if set(payload) != {"ok", "job", "returned_chunk_ids"}:
        raise SandboxError("reranker output is malformed")
    if payload.get("ok") is not True or payload.get("job") != "reranker_apply":
        raise SandboxError("reranker output is malformed")
    returned_ids = payload.get("returned_chunk_ids")
    if not isinstance(returned_ids, list) or any(
        not isinstance(chunk_id, str) or not chunk_id for chunk_id in returned_ids
    ):
        raise SandboxError("reranker output must contain a list of chunk IDs")
    if len(returned_ids) != len(set(returned_ids)):
        raise SandboxError("reranker output contains duplicate chunk IDs")
    if len(returned_ids) > top_k:
        raise SandboxError("reranker output exceeds top_k")
    return returned_ids


__all__ = [
    "MAX_RERANKER_HITS",
    "MAX_RERANKER_QUERY_BYTES",
    "MAX_RERANKER_TOP_K",
    "SandboxedRerankerConfig",
    "SandboxedRerankerProxy",
]
