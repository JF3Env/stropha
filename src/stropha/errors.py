"""Structured exceptions. Per CLAUDE.md, never raise bare Exception."""

from __future__ import annotations


class StrophaError(Exception):
    """Base class for all errors raised by stropha."""


class ConfigError(StrophaError):
    """Invalid or missing configuration."""


class WalkerError(StrophaError):
    """Failure while discovering source files."""


class ChunkerError(StrophaError):
    """Failure while splitting a file into chunks."""


class EmbeddingError(StrophaError):
    """Failure while computing embeddings (network, auth, model)."""


class InferenceError(StrophaError):
    """Failure while generating text with an inference backend."""


class InferenceTimeoutError(InferenceError, TimeoutError):
    """Inference generation exceeded its configured timeout."""


class StorageError(StrophaError):
    """Failure in the SQLite/vec layer (schema, IO, corruption)."""


class EmbeddingDimensionMismatchError(StorageError):
    """The persisted index and configured embedder use different dimensions."""


class RetrievalError(StrophaError):
    """Failure during query execution."""


class PipelineError(StrophaError):
    """Failure assembling or running the pipeline."""


class AdapterError(StrophaError):
    """Failure registering or invoking an adapter.

    Raised by ``pipeline.registry`` when an unknown adapter is requested or
    when a registered adapter does not satisfy its stage's contract.
    """
