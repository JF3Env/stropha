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


class ObservationError(StrophaError):
    """Invalid or over-quota ephemeral agent observation."""


class InstructionResolutionError(StrophaError):
    """Invalid or unsafe repository instruction context request."""


class AgentContextError(StrophaError):
    """Invalid bounded agent-context assembly request."""


class PipelineError(StrophaError):
    """Failure assembling or running the pipeline."""


class FullReconcileRequiredError(PipelineError):
    """A path-scoped refresh cannot safely preserve the complete index."""


class RepoBlacklistedError(PipelineError):
    """Indexing was requested for a repository blocked by configuration.

    Raised however the run was triggered — CLI ``--repo``, the watcher, or the
    post-commit hook — so `indexing.blacklist_repos` cannot be bypassed by
    asking a different entry point.
    """

    def __init__(self, repo: str) -> None:
        super().__init__(
            f"{repo} is listed in indexing.blacklist_repos and will not be indexed. "
            "Remove it from ~/.stropha/config.yaml (or the dashboard's Indexed data "
            "tab) to index it again."
        )
        self.repo = repo


class AdapterError(StrophaError):
    """Failure registering or invoking an adapter.

    Raised by ``pipeline.registry`` when an unknown adapter is requested or
    when a registered adapter does not satisfy its stage's contract.
    """
