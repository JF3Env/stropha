"""Benchmark: code-aware vs cross-encoder vs noop reranker.

WIN-297 — hard A/B benchmark on the golden dataset.

Runs the eval harness 3x with different reranker configs, holding everything
else constant (same index, same embedder, same streams). Reports Recall@K,
MRR, NDCG@10, MAP@10, Precision@10, and latency percentiles for each.

Usage:
    uv run python tests/benchmarks/bench_reranker.py [--top-k 10] [--json]

Requires an existing index at the default path (run `stropha index` first).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

# Ensure project root is importable
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from stropha.eval.harness import EvalReport, load_golden, run_eval
from stropha.pipeline.builder import build_stages
from stropha.pipeline.config import load_pipeline_config


# Three reranker configurations to benchmark
RERANKER_CONFIGS: dict[str, dict[str, Any] | None] = {
    "noop": None,  # No reranker — pure RRF output
    "cross-encoder": {
        "adapter": "cross-encoder",
        "config": {"model": "BAAI/bge-reranker-base", "batch_size": 32},
    },
    "code-aware": {
        "adapter": "code-aware",
        "config": {
            "definition_boost": 1.5,
            "stem_match_boost": 1.2,
            "noise_penalty": 0.7,
            "file_coherence": True,
        },
    },
}


def _build_retrieval_with_reranker(
    reranker_config: dict[str, Any] | None,
) -> Any:
    """Build a full BuiltStages with a specific reranker config."""
    # Build overrides that will win over the user config
    overrides: dict[str, Any] = {
        "retrieval": {
            "config": {
                "reranker": reranker_config,
                "candidate_k": 50 if reranker_config else 10,
            }
        }
    }
    resolved = load_pipeline_config(overrides=overrides)

    built = build_stages(resolved)
    return built


def _report_to_dict(name: str, report: EvalReport, top_k: int = 10) -> dict[str, Any]:
    """Extract key metrics from an EvalReport."""
    result = {
        "reranker": name,
        "n_cases": report.n,
        "recall_at_k": round(report.recall_at_k, 4),
        "mrr": round(report.mrr, 4),
        "ndcg_at_k": round(report.ndcg_at_k, 4),
        "precision_at_k": round(report.precision_at_k, 4),
        "map_at_k": round(report.map_at_k, 4),
        "latency_p50_ms": round(report.latency_p50_ms, 1),
        "latency_p95_ms": round(report.latency_p95_ms, 1),
    }

    # Per-tag breakdown
    by_tag = report.by_tag()
    if by_tag:
        result["by_tag"] = {}
        for tag, tag_report in sorted(by_tag.items()):
            result["by_tag"][tag] = {
                "n": tag_report.n,
                "recall": round(tag_report.recall_at_k, 4),
                "mrr": round(tag_report.mrr, 4),
                "ndcg": round(tag_report.ndcg_at_k, 4),
            }

    return result


def run_benchmark(
    golden_path: Path,
    top_k: int = 10,
    json_out: bool = False,
) -> list[dict[str, Any]]:
    """Run the full reranker benchmark."""
    cases = load_golden(golden_path)
    if not cases:
        print("ERROR: No cases loaded from golden file", file=sys.stderr)
        sys.exit(1)

    print(f"Loaded {len(cases)} eval cases from {golden_path}")
    print(f"Top-K: {top_k}")
    print("=" * 72)

    results: list[dict[str, Any]] = []

    for name, reranker_cfg in RERANKER_CONFIGS.items():
        print(f"\n--- Benchmarking reranker: {name} ---")
        start = time.perf_counter()

        built = _build_retrieval_with_reranker(reranker_cfg)

        with built.storage:
            report = run_eval(cases, built.retrieval, top_k=top_k)

        elapsed = time.perf_counter() - start
        metrics = _report_to_dict(name, report, top_k=top_k)
        metrics["total_time_s"] = round(elapsed, 2)
        results.append(metrics)

        if not json_out:
            print(f"  Recall@{top_k}:    {metrics['recall_at_k']:.4f}")
            print(f"  MRR:           {metrics['mrr']:.4f}")
            print(f"  NDCG@{top_k}:     {metrics['ndcg_at_k']:.4f}")
            print(f"  Precision@{top_k}: {metrics['precision_at_k']:.4f}")
            print(f"  MAP@{top_k}:      {metrics['map_at_k']:.4f}")
            print(f"  Latency p50:   {metrics['latency_p50_ms']:.1f} ms")
            print(f"  Latency p95:   {metrics['latency_p95_ms']:.1f} ms")
            print(f"  Total time:    {metrics['total_time_s']:.2f} s")

    if json_out:
        print(json.dumps(results, indent=2))
    else:
        # Print comparison table
        print("\n" + "=" * 72)
        print("COMPARISON TABLE")
        print("=" * 72)
        header = f"{'Reranker':<15} {'Recall@K':<10} {'MRR':<8} {'NDCG@K':<9} {'P@K':<8} {'MAP@K':<8} {'p50ms':<8} {'p95ms':<8}"
        print(header)
        print("-" * len(header))
        for r in results:
            print(
                f"{r['reranker']:<15} "
                f"{r['recall_at_k']:<10.4f} "
                f"{r['mrr']:<8.4f} "
                f"{r['ndcg_at_k']:<9.4f} "
                f"{r['precision_at_k']:<8.4f} "
                f"{r['map_at_k']:<8.4f} "
                f"{r['latency_p50_ms']:<8.1f} "
                f"{r['latency_p95_ms']:<8.1f}"
            )

        # Find winner
        print("\n" + "-" * 72)
        best_recall = max(results, key=lambda x: x["recall_at_k"])
        best_mrr = max(results, key=lambda x: x["mrr"])
        best_ndcg = max(results, key=lambda x: x["ndcg_at_k"])
        fastest = min(results, key=lambda x: x["latency_p95_ms"])

        print(f"Best Recall@K:  {best_recall['reranker']} ({best_recall['recall_at_k']:.4f})")
        print(f"Best MRR:       {best_mrr['reranker']} ({best_mrr['mrr']:.4f})")
        print(f"Best NDCG@K:    {best_ndcg['reranker']} ({best_ndcg['ndcg_at_k']:.4f})")
        print(f"Fastest (p95):  {fastest['reranker']} ({fastest['latency_p95_ms']:.1f} ms)")

        # Per-tag winner analysis
        all_tags = set()
        for r in results:
            all_tags.update(r.get("by_tag", {}).keys())

        if all_tags:
            print(f"\nPER-TAG ANALYSIS")
            print("-" * 72)
            for tag in sorted(all_tags):
                tag_results = []
                for r in results:
                    tag_data = r.get("by_tag", {}).get(tag)
                    if tag_data:
                        tag_results.append((r["reranker"], tag_data))
                if tag_results:
                    best = max(tag_results, key=lambda x: x[1]["recall"])
                    print(
                        f"  {tag:<20} winner: {best[0]} "
                        f"(recall={best[1]['recall']:.4f}, "
                        f"ndcg={best[1]['ndcg']:.4f})"
                    )

    return results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark reranker configurations against golden dataset."
    )
    parser.add_argument(
        "--golden", "-g",
        type=Path,
        default=Path("tests/eval/golden.jsonl"),
        help="Path to golden JSONL file.",
    )
    parser.add_argument("--top-k", "-k", type=int, default=10)
    parser.add_argument("--json", action="store_true", help="JSON output.")
    args = parser.parse_args()

    run_benchmark(args.golden, top_k=args.top_k, json_out=args.json)


if __name__ == "__main__":
    main()
