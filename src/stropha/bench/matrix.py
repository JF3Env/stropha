"""Matrix spec: parse a YAML of axes → enumerate configuration cells.

A matrix declares **axes** (each a named knob with a list of values) plus a
golden set, top_k, repo, and an optional baseline. Expanding the matrix is the
Cartesian product of the axes, minus any cells excluded by ``constraints``.

Axes are classified *indexing* vs *query-time* by name (``INDEXING_AXES``).
Cells that share the same indexing-axis values share one index — the runner
exploits this to build each index once.

Example::

    repo: .
    golden: tests/eval/golden.jsonl
    top_k: 10
    axes:
      embedder: [{adapter: voyage, dim: 512}, {adapter: voyage, dim: 1024}]
      enricher: [noop, hierarchical, contextual]
      reranker: [null, code-aware, cross-encoder]
      hyde:     [false, true]
    baseline:
      embedder: {adapter: voyage, dim: 512}
      enricher: noop
      reranker: null
      hyde: false
    constraints:
      - skip_when: {embedder: {adapter: voyage, dim: 2048}, enricher: contextual}
"""

from __future__ import annotations

import hashlib
import itertools
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml  # type: ignore[import-untyped]

from ..errors import StrophaError

# Axes whose value changes the *stored* index (vectors / FTS / chunks) and
# therefore force a rebuild. Everything else is query-time and reuses the
# index built for its indexing-axis combination.
INDEXING_AXES: frozenset[str] = frozenset(
    {"embedder", "enricher", "chunker", "walker", "graph_fts_augment"}
)

# Axes routed to the global ``retrieval_features`` (read by hybrid_rrf at query
# time) rather than the pipeline ``retrieval.config`` block.
FEATURE_AXES: frozenset[str] = frozenset(
    {
        "hyde",
        "hyde_code_tuned",
        "recursive",
        "recursive_adjacency",
        "query_rewrite",
        "query_decomposition",
        "multi_query",
        "multi_query_count",
        "adaptive_weighting",
        "cache_enabled",
    }
)


class MatrixError(StrophaError):
    """Raised when a matrix spec is malformed."""


def _canonical(value: Any) -> str:
    """Stable string for an axis value (dicts sorted, JSON-encoded)."""
    return json.dumps(value, sort_keys=True, ensure_ascii=False, default=str)


def _short_hash(payload: str) -> str:
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]


@dataclass(frozen=True)
class Cell:
    """One fully-specified configuration in the matrix."""

    values: dict[str, Any]  # axis_name -> value

    @property
    def cell_id(self) -> str:
        """Stable id over ALL axes (the unit of resume / dedup)."""
        payload = ";".join(f"{k}={_canonical(self.values[k])}" for k in sorted(self.values))
        return _short_hash(payload)

    def indexing_values(self) -> dict[str, Any]:
        return {k: v for k, v in self.values.items() if k in INDEXING_AXES}

    @property
    def indexing_sig(self) -> str:
        """Stable id over only the indexing axes — cells sharing it share an
        index. ``"base"`` when the matrix varies no indexing axis."""
        idx = self.indexing_values()
        if not idx:
            return "base"
        payload = ";".join(f"{k}={_canonical(idx[k])}" for k in sorted(idx))
        return _short_hash(payload)

    def label(self) -> str:
        """Human-readable one-liner, e.g. ``embedder=voyage/512 enricher=noop``."""
        parts = []
        for k in sorted(self.values):
            v = self.values[k]
            if isinstance(v, dict):
                v = "/".join(str(x) for x in v.values())
            parts.append(f"{k}={v}")
        return " ".join(parts)

    def to_dict(self) -> dict[str, Any]:
        return {
            "cell_id": self.cell_id,
            "indexing_sig": self.indexing_sig,
            "values": self.values,
        }


@dataclass
class Matrix:
    """A parsed matrix spec."""

    axes: dict[str, list[Any]]
    golden: str = "tests/eval/golden.jsonl"
    top_k: int = 10
    repo: str = "."
    baseline: dict[str, Any] | None = None
    constraints: list[dict[str, Any]] = field(default_factory=list)

    def expand(self) -> list[Cell]:
        """Cartesian product of axes, minus constraint-excluded cells."""
        if not self.axes:
            raise MatrixError("matrix has no axes")
        names = sorted(self.axes)
        value_lists = [self.axes[n] for n in names]
        cells: list[Cell] = []
        seen: set[str] = set()
        for combo in itertools.product(*value_lists):
            values = dict(zip(names, combo, strict=True))
            if self._excluded(values):
                continue
            cell = Cell(values=values)
            if cell.cell_id in seen:
                continue
            seen.add(cell.cell_id)
            cells.append(cell)
        if not cells:
            raise MatrixError("matrix expanded to zero cells (all excluded?)")
        return cells

    def groups(self) -> dict[str, list[Cell]]:
        """Cells bucketed by ``indexing_sig`` (build order for the runner)."""
        out: dict[str, list[Cell]] = {}
        for cell in self.expand():
            out.setdefault(cell.indexing_sig, []).append(cell)
        return out

    def baseline_cell(self) -> Cell | None:
        """Resolve the declared baseline to the matching expanded cell.

        The baseline only needs to pin the axes it cares about; it's matched
        against expanded cells by those axes. Returns None when undeclared or
        unmatched.
        """
        if not self.baseline:
            return None
        want = {k: _canonical(v) for k, v in self.baseline.items()}
        for cell in self.expand():
            if all(
                k in cell.values and _canonical(cell.values[k]) == want[k]
                for k in want
            ):
                return cell
        return None

    def _excluded(self, values: dict[str, Any]) -> bool:
        for con in self.constraints:
            when = con.get("skip_when") or con.get("when") or {}
            if when and all(
                k in values and _canonical(values[k]) == _canonical(v)
                for k, v in when.items()
            ):
                return True
        return False


def load_matrix(path: Path | str) -> Matrix:
    """Parse a matrix YAML into a :class:`Matrix`."""
    p = Path(path)
    if not p.is_file():
        raise MatrixError(f"matrix spec not found: {p}")
    try:
        raw = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError as exc:
        raise MatrixError(f"matrix spec malformed: {exc}") from exc
    if not isinstance(raw, dict):
        raise MatrixError("matrix spec root must be a mapping")

    axes = raw.get("axes")
    if not isinstance(axes, dict) or not axes:
        raise MatrixError("matrix spec must declare a non-empty `axes` mapping")
    norm_axes: dict[str, list[Any]] = {}
    for name, vals in axes.items():
        if not isinstance(vals, list) or not vals:
            raise MatrixError(f"axis {name!r} must be a non-empty list of values")
        norm_axes[str(name)] = list(vals)

    constraints = raw.get("constraints") or []
    if not isinstance(constraints, list):
        raise MatrixError("`constraints` must be a list")

    return Matrix(
        axes=norm_axes,
        golden=str(raw.get("golden", "tests/eval/golden.jsonl")),
        top_k=int(raw.get("top_k", 10)),
        repo=str(raw.get("repo", ".")),
        baseline=raw.get("baseline"),
        constraints=[c for c in constraints if isinstance(c, dict)],
    )


__all__ = [
    "FEATURE_AXES",
    "INDEXING_AXES",
    "Cell",
    "Matrix",
    "MatrixError",
    "load_matrix",
]
