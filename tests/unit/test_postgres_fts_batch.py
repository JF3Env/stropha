"""PostgreSQL graph-derived FTS augmentation batching tests."""

from __future__ import annotations

from contextlib import nullcontext
from typing import Any

from stropha.adapters.storage.postgres import PostgresStorage


class _Result:
    def __init__(
        self,
        *,
        one: dict[str, Any] | None = None,
        rows: list[dict[str, Any]] | None = None,
    ) -> None:
        self._one = one
        self._rows = rows or []

    def fetchone(self) -> dict[str, Any] | None:
        return self._one

    def fetchall(self) -> list[dict[str, Any]]:
        return self._rows


class _Raw:
    def __init__(self, rows: list[dict[str, Any]]) -> None:
        self.rows = rows
        self.calls: list[tuple[str, Any]] = []
        self.commits = 0

    def execute(self, query: str, params: Any = None) -> _Result:
        self.calls.append((query, params))
        if "SELECT 1 AS x FROM graph_nodes" in query:
            return _Result(one={"x": 1})
        if "SELECT DISTINCT ON (c.id)" in query:
            return _Result(rows=self.rows)
        return _Result()

    def commit(self) -> None:
        self.commits += 1

    def transaction(self):
        return nullcontext()


def test_postgres_fts_augmentation_deduplicates_and_batches() -> None:
    rows = [
        {
            "id": index,
            "rel_path": "deployment.yaml",
            "symbol": "Deployment/api",
            "content": f"replicas: {index}\n",
            "label": "Deployment",
            "community_label": "GitOps",
            "source_location": f"L{index}",
        }
        for index in range(1_001)
    ]
    raw = _Raw(rows)
    storage = object.__new__(PostgresStorage)
    storage._raw = raw

    result = storage.augment_fts_with_graph()

    select_query = next(query for query, _ in raw.calls if "FROM chunks c" in query)
    update_calls = [
        (query, params) for query, params in raw.calls if "UPDATE chunks AS c" in query
    ]
    assert "DISTINCT ON (c.id)" in select_query
    assert "n.repo_id IS NOT DISTINCT FROM c.repo_id" in select_query
    assert len(update_calls) == 2
    assert len(update_calls[0][1]) == 2_000
    assert len(update_calls[1][1]) == 2
    assert raw.commits == 2
    assert result == {"augmented": 1_001, "unchanged": 0, "missing_graph": 0}
