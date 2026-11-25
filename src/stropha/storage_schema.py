"""Schema facts shared by every storage adapter.

A leaf module on purpose: ``stropha.storage.sqlite`` and
``stropha.adapters.storage.postgres`` both need these names, and importing
``stropha.stages.storage`` from an adapter closes an import cycle through the
adapter registry.
"""

from __future__ import annotations

# Every table keyed by ``repo_id``, ordered children-first so an adapter can
# delete straight down the list without tripping a foreign key. ``chunks`` and
# ``files`` come first because ``clear_repo`` already owns their vector and
# full-text side tables.
REPO_SCOPED_TABLES: tuple[str, ...] = (
    "chunks",
    "files",
    "graph_edges",
    "graph_nodes",
    "project_memory",
    "episodes",
    "cross_session_cache",
)

# ``meta`` rows carry the repo id in the key rather than a column, so deleting
# a repository has to name them. ``last_indexed_sha_`` is the odd one out: it
# joins the id without a colon. Mirrored by the pipeline's private constants —
# ``test_repo_scoped_meta_prefixes_match_the_pipeline`` keeps them in step.
REPO_SCOPED_META_PREFIXES: tuple[str, ...] = (
    "last_indexed_sha_",
    "source_snapshot_version:",
    "graph_dirty:",
    "active_chunker_id:",
    "active_walker_id:",
)


def repo_scoped_meta_keys(repo_id: int) -> tuple[str, ...]:
    """Every ``meta`` key that belongs to one repository."""
    return tuple(f"{prefix}{repo_id}" for prefix in REPO_SCOPED_META_PREFIXES)
