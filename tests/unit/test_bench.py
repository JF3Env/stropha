"""Unit tests for the config-matrix benchmark (pure logic — no indexing)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from stropha.bench.matrix import Cell, MatrixError, load_matrix
from stropha.bench.report import BenchReport, load_results, to_csv

# --------------------------------------------------------------------------- matrix

_SPEC = """
repo: .
golden: g.jsonl
top_k: 5
axes:
  embedder:
    - {adapter: voyage, dim: 512}
    - {adapter: mlx}
  enricher: [noop, hierarchical]
  reranker: [null, code-aware]
  hyde: [false, true]
baseline:
  embedder: {adapter: voyage, dim: 512}
  enricher: noop
  reranker: null
  hyde: false
constraints:
  - skip_when: {embedder: {adapter: mlx}, enricher: hierarchical}
"""


@pytest.fixture
def matrix(tmp_path: Path):
    p = tmp_path / "m.yaml"
    p.write_text(_SPEC, encoding="utf-8")
    return load_matrix(p)


def test_load_parses_fields(matrix) -> None:
    assert matrix.top_k == 5
    assert matrix.golden == "g.jsonl"
    assert set(matrix.axes) == {"embedder", "enricher", "reranker", "hyde"}


def test_expand_product_minus_constraints(matrix) -> None:
    cells = matrix.expand()
    # 2*2*2*2 = 16, minus mlx+hierarchical (reranker2 * hyde2 = 4) = 12.
    assert len(cells) == 12
    # No cell violates the skip constraint.
    for c in cells:
        assert not (
            c.values["embedder"] == {"adapter": "mlx"}
            and c.values["enricher"] == "hierarchical"
        )


def test_cell_ids_unique_and_stable(matrix) -> None:
    cells = matrix.expand()
    ids = [c.cell_id for c in cells]
    assert len(ids) == len(set(ids))
    # Stable: same values → same id regardless of dict order.
    a = Cell(values={"hyde": False, "reranker": None})
    b = Cell(values={"reranker": None, "hyde": False})
    assert a.cell_id == b.cell_id


def test_groups_by_indexing_signature(matrix) -> None:
    groups = matrix.groups()
    # indexing axes = embedder x enricher, minus mlx+hierarchical = 3 indexes.
    assert len(groups) == 3
    # Each index is swept by reranker(2) x hyde(2) = 4 query variants.
    for sig, cells in groups.items():
        assert len(cells) == 4
        assert len({c.indexing_sig for c in cells}) == 1
        assert all(c.indexing_sig == sig for c in cells)
        # Within a group, only query axes vary.
        assert len({(str(c.values["embedder"]), c.values["enricher"]) for c in cells}) == 1


def test_query_only_change_shares_index(matrix) -> None:
    a = Cell(values={"embedder": {"adapter": "mlx"}, "enricher": "noop",
                     "reranker": None, "hyde": False})
    b = Cell(values={"embedder": {"adapter": "mlx"}, "enricher": "noop",
                     "reranker": "code-aware", "hyde": True})
    assert a.indexing_sig == b.indexing_sig  # only query axes differ
    assert a.cell_id != b.cell_id


def test_baseline_cell_resolves(matrix) -> None:
    base = matrix.baseline_cell()
    assert base is not None
    assert base.values["enricher"] == "noop"
    assert base.values["reranker"] is None


def test_empty_axes_raises(tmp_path: Path) -> None:
    p = tmp_path / "bad.yaml"
    p.write_text("axes: {}\n", encoding="utf-8")
    with pytest.raises(MatrixError):
        load_matrix(p)


def test_all_excluded_raises(tmp_path: Path) -> None:
    p = tmp_path / "x.yaml"
    p.write_text(
        "axes:\n  enricher: [noop]\nconstraints:\n  - skip_when: {enricher: noop}\n",
        encoding="utf-8",
    )
    with pytest.raises(MatrixError):
        load_matrix(p).expand()


# --------------------------------------------------------------------------- report

def _results() -> list[dict]:
    return [
        {
            "ok": True, "cell_id": "a", "indexing_sig": "i1",
            "label": "reranker=None", "values": {"reranker": None, "hyde": False},
            "metrics": {"recall@10": 0.8, "mrr": 0.5},
            "per_query": {"q1": 1, "q2": None, "q3": 2},
        },
        {
            "ok": True, "cell_id": "b", "indexing_sig": "i1",
            "label": "reranker=code-aware", "values": {"reranker": "code-aware", "hyde": False},
            "metrics": {"recall@10": 0.7, "mrr": 0.55},
            "per_query": {"q1": 1, "q2": 3, "q3": None},
        },
    ]


def test_load_results_filters_failures(tmp_path: Path) -> None:
    p = tmp_path / "r.jsonl"
    lines = [json.dumps(r) for r in _results()]
    lines.append(json.dumps({"ok": False, "cell_id": "c", "error": "boom"}))
    lines.append("")  # blank line tolerated
    p.write_text("\n".join(lines), encoding="utf-8")
    out = load_results(p)
    assert [r["cell_id"] for r in out] == ["a", "b"]


def test_rank_table_sorts_desc() -> None:
    rep = BenchReport(_results())
    ranked = rep.rank_table("recall@10")
    assert [r["cell_id"] for r in ranked] == ["a", "b"]
    assert rep.rank_table("mrr")[0]["cell_id"] == "b"  # b has higher mrr


def test_marginals_mean_per_axis_value() -> None:
    rep = BenchReport(_results())
    marg = rep.marginals("recall@10")
    assert marg["reranker"]["None"]["mean"] == pytest.approx(0.8)
    assert marg["reranker"]["code-aware"]["mean"] == pytest.approx(0.7)
    # Both cells share hyde=False → one bucket averaging the two.
    assert marg["hyde"]["False"]["mean"] == pytest.approx(0.75)
    assert marg["hyde"]["False"]["n"] == 2.0


def test_baseline_deltas_shape_and_sign() -> None:
    rep = BenchReport(_results())
    deltas = rep.baseline_deltas("a")
    assert len(deltas) == 1
    d = deltas[0]
    assert d["cell_id"] == "b"
    # each delta = (point, lo, hi, p_gt_0)
    assert len(d["d_recall"]) == 4 and len(d["d_mrr"]) == 4
    # recall identical (2/3 each) → point 0; mrr lower for b → negative point.
    assert d["d_recall"][0] == pytest.approx(0.0)
    assert d["d_mrr"][0] < 0


def test_to_csv_has_header_and_rows() -> None:
    csv = to_csv(_results())
    lines = csv.strip().splitlines()
    assert lines[0].startswith("cell_id,indexing_sig,")
    assert "recall@10" in lines[0] and "reranker" in lines[0]
    assert len(lines) == 3  # header + 2 rows
