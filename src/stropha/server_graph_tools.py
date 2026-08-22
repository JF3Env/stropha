"""Graph and analysis helpers extracted from ``stropha.server``."""

from __future__ import annotations

import contextlib
from collections.abc import Callable
from pathlib import Path

from .server_models import (
    BlastRadiusResponse,
    CitationAuditResponse,
    CitationEntry,
    ClaimAuditEntry,
    ContractTestEntry,
    ContractTestsResponse,
    FreshnessResponse,
    HallucinationGuardResponse,
    ImpactedTestEntry,
    PerSymbolCounts,
    SpecConflictEntry,
    SpecConsistencyResponse,
    SpecInvariantEntry,
    SymbolCheckEntry,
    TestImpactResponse,
    VerdictResponse,
)
from .server_runtime import AppContext, chunk_belongs_to_repo

GraphLoaded = Callable[..., bool]
HasRationaleEdges = Callable[..., bool]
Unavailable = Callable[[], dict]


def find_callers_impl(
    symbol: str,
    app: AppContext,
    *,
    depth: int,
    limit: int,
    repo_id: int | None,
    graph_loaded_fn: GraphLoaded,
    unavailable_fn: Unavailable,
    graph_find_callers_fn: Callable[..., dict],
) -> dict:
    if not graph_loaded_fn(app.storage, repo_id=repo_id):
        return unavailable_fn() | {"symbol": symbol}
    return graph_find_callers_fn(
        app.storage, symbol, depth=depth, limit=limit, repo_id=repo_id
    )


def find_related_impl(
    symbol: str,
    app: AppContext,
    *,
    depth: int,
    limit: int,
    relations: list[str] | None,
    repo_id: int | None,
    graph_loaded_fn: GraphLoaded,
    unavailable_fn: Unavailable,
    graph_find_related_fn: Callable[..., dict],
) -> dict:
    if not graph_loaded_fn(app.storage, repo_id=repo_id):
        return unavailable_fn() | {"symbol": symbol}
    return graph_find_related_fn(
        app.storage,
        symbol,
        depth=depth,
        limit=limit,
        relations=tuple(relations) if relations else None,
        repo_id=repo_id,
    )


def get_community_impl(
    symbol_or_community_id: str,
    app: AppContext,
    *,
    limit: int,
    repo_id: int | None,
    graph_loaded_fn: GraphLoaded,
    unavailable_fn: Unavailable,
    graph_get_community_fn: Callable[..., dict],
) -> dict:
    if not graph_loaded_fn(app.storage, repo_id=repo_id):
        return unavailable_fn() | {"query": symbol_or_community_id}
    query: str | int = symbol_or_community_id
    with contextlib.suppress(ValueError, TypeError):
        query = int(symbol_or_community_id)
    return graph_get_community_fn(app.storage, query, limit=limit, repo_id=repo_id)


def find_rationale_impl(
    symbol: str,
    app: AppContext,
    *,
    limit: int,
    repo_id: int | None,
    graph_loaded_fn: GraphLoaded,
    has_rationale_edges_fn: HasRationaleEdges,
    unavailable_fn: Unavailable,
    graph_find_rationale_fn: Callable[..., dict],
) -> dict:
    if not graph_loaded_fn(app.storage, repo_id=repo_id):
        return unavailable_fn() | {"symbol": symbol}
    if not has_rationale_edges_fn(app.storage, repo_id=repo_id):
        return {
            "symbol": symbol,
            "rationale": [],
            "message": "Graph loaded but no `rationale_for` edges present yet.",
        }
    return graph_find_rationale_fn(app.storage, symbol, limit=limit, repo_id=repo_id)


def find_tests_for_impl(
    symbol: str,
    app: AppContext,
    *,
    limit: int,
    test_path_patterns: list[str] | None,
    repo_id: int | None,
    graph_loaded_fn: GraphLoaded,
    unavailable_fn: Unavailable,
    graph_find_tests_for_fn: Callable[..., dict],
) -> dict:
    if not graph_loaded_fn(app.storage, repo_id=repo_id):
        return unavailable_fn() | {"symbol": symbol}
    patterns = (
        tuple(test_path_patterns)
        if test_path_patterns
        else ("test_", "_test", "/tests/", "/test/", ".spec.", ".test.")
    )
    return graph_find_tests_for_fn(
        app.storage,
        symbol,
        limit=limit,
        test_path_patterns=patterns,
        repo_id=repo_id,
    )


def trace_feature_impl(
    feature: str,
    app: AppContext,
    *,
    max_paths: int,
    max_depth: int,
    repo_id: int | None,
    graph_loaded_fn: GraphLoaded,
    unavailable_fn: Unavailable,
    graph_trace_feature_fn: Callable[..., dict],
) -> dict:
    if not graph_loaded_fn(app.storage, repo_id=repo_id):
        return unavailable_fn() | {"feature": feature}
    return graph_trace_feature_fn(
        app.storage,
        feature,
        max_paths=max_paths,
        max_depth=max_depth,
        repo_id=repo_id,
    )


def verify_claim_impl(
    claim: str,
    app: AppContext,
    *,
    graph_loaded_fn: GraphLoaded,
    repo_id: int | None,
) -> VerdictResponse:
    from .retrieval.graph import (
        _fetch_node_rows,
        _hydrate_node,
        find_callers,
        find_related,
        find_tests_for,
        resolve_symbol_to_node,
    )
    from .retrieval.verify import parse_claim, verify_claim

    parsed = parse_claim(claim)
    if parsed is None:
        return VerdictResponse(
            raw_text=claim,
            parsed=False,
            verified=None,
            reason=(
                "Claim did not match any supported predicate. "
                "Try 'X calls Y', 'X is a class', 'X is in src/foo.py', 'X has tests'."
            ),
        )
    if not graph_loaded_fn(app.storage, repo_id=repo_id):
        return VerdictResponse(
            raw_text=parsed.raw_text,
            parsed=True,
            subject=parsed.subject,
            predicate=parsed.predicate,
            object=parsed.object or "",
            verified=None,
            reason=(
                "graphify mirror is not loaded — cannot verify structural claims. "
                "Run `graphify .` and re-index."
            ),
        )

    def callers(symbol: str) -> dict:
        return find_callers(app.storage, symbol, repo_id=repo_id)

    def related(symbol: str, relations: list[str] | None) -> dict:
        return find_related(
            app.storage,
            symbol,
            relations=tuple(relations) if relations else None,
            repo_id=repo_id,
        )

    def tests(symbol: str) -> dict:
        return find_tests_for(app.storage, symbol, repo_id=repo_id)

    def resolve(symbol: str) -> dict | None:
        node_id = resolve_symbol_to_node(app.storage, symbol, repo_id=repo_id)
        if node_id is None:
            return None
        rows = _fetch_node_rows(app.storage, [node_id], repo_id=repo_id)
        if node_id not in rows:
            return None
        return _hydrate_node(app.storage, rows[node_id]).as_dict()

    verdict = verify_claim(parsed, callers_fn=callers, related_fn=related, tests_fn=tests, resolve_fn=resolve)
    return VerdictResponse(
        raw_text=verdict.claim.raw_text,
        parsed=True,
        subject=verdict.claim.subject,
        predicate=verdict.claim.predicate,
        object=verdict.claim.object or "",
        verified=verdict.verified,
        reason=verdict.reason,
        evidence=list(verdict.evidence),
        alternatives=list(verdict.alternatives),
    )


def check_symbols_impl(
    symbols: list[str],
    app: AppContext,
    *,
    max_suggestions: int,
    repo_id: int | None,
) -> HallucinationGuardResponse:
    from .retrieval.hallucination_guard import check_symbols

    max_suggestions = max(1, min(int(max_suggestions), 20))

    def exists(name: str) -> bool:
        if repo_id is None:
            return bool(app.storage.lookup_by_symbol(name, limit=1))
        return bool(app.storage.lookup_by_symbol(name, limit=1, repo_id=repo_id))

    def similar(name: str) -> list[str]:
        try:
            if repo_id is None:
                hits = app.storage.lookup_by_symbol(name, limit=20)
            else:
                hits = app.storage.lookup_by_symbol(name, limit=20, repo_id=repo_id)
        except Exception:
            return []
        out: list[str] = []
        for hit in hits:
            sym = getattr(hit, "symbol", None)
            if sym and sym not in out:
                out.append(str(sym))
        return out

    report = check_symbols(symbols, exists_fn=exists, similar_fn=similar, max_suggestions=max_suggestions)
    return HallucinationGuardResponse(
        checks=[
            SymbolCheckEntry(
                symbol=check.symbol,
                exists=check.exists,
                suggestions=list(check.suggestions),
                reason=check.reason,
            )
            for check in report.checks
        ],
        present_count=report.present_count,
        missing_count=report.missing_count,
        notes=list(report.notes),
    )


def blast_radius_impl(
    symbols: list[str],
    app: AppContext,
    *,
    graph_loaded_fn: GraphLoaded,
    has_rationale_edges_fn: HasRationaleEdges,
    repo_id: int | None,
) -> BlastRadiusResponse:
    from .retrieval.blast_radius import compute_blast_radius
    from .retrieval.graph import find_callers, find_rationale, find_related, find_tests_for

    graph_ok = graph_loaded_fn(app.storage, repo_id=repo_id)
    if not graph_ok:
        return BlastRadiusResponse(
            changed_symbols=[s.strip() for s in symbols if s and s.strip()],
            graph_loaded=False,
            summary="graphify mirror not loaded — cannot compute blast radius. Run `graphify .` and re-index.",
            notes=["graph_loaded=False"],
        )

    rationale_fn = None
    if has_rationale_edges_fn(app.storage, repo_id=repo_id):
        def rationale(symbol: str) -> dict:
            return find_rationale(app.storage, symbol, repo_id=repo_id)
        rationale_fn = rationale

    report = compute_blast_radius(
        symbols,
        callers_fn=lambda symbol: find_callers(app.storage, symbol, repo_id=repo_id),
        tests_fn=lambda symbol: find_tests_for(app.storage, symbol, repo_id=repo_id),
        related_fn=lambda symbol: find_related(app.storage, symbol, repo_id=repo_id),
        rationale_fn=rationale_fn,
    )
    return BlastRadiusResponse(
        changed_symbols=report.changed_symbols,
        callers=report.callers,
        tests=report.tests,
        related=report.related,
        rationale=report.rationale,
        per_symbol={symbol: PerSymbolCounts(**counts) for symbol, counts in report.per_symbol.items()},
        notes=report.notes,
        summary=report.summary,
        graph_loaded=True,
    )


def assert_snippet_freshness_impl(
    chunk_id: str,
    expected_snippet: str,
    app: AppContext,
    *,
    strict: bool,
    repo_id: int | None,
) -> FreshnessResponse:
    from .retrieval.freshness import assert_snippet_freshness

    try:
        chunk = app.storage.get_chunk_by_id(chunk_id)
        allowed = chunk is not None and (
            repo_id is None or chunk.get("repo_id") == repo_id
        )
        hit = app.storage.lookup_by_chunk_id(chunk_id) if allowed else None
        source = getattr(hit, "source", None)
        indexed_file_sha256 = getattr(source, "indexed_file_sha256", None)
        if not isinstance(indexed_file_sha256, str):
            indexed_file_sha256 = None
    except Exception as exc:
        return FreshnessResponse(
            chunk_id=chunk_id,
            fresh=None,
            reason=f"lookup raised: {exc!r}",
            current_snippet=None,
            expected_snippet=expected_snippet,
            drift_details=[],
        )
    verdict = assert_snippet_freshness(
        chunk_id,
        expected_snippet,
        lookup_fn=lambda _cid: hit,
        strict=bool(strict),
        indexed_file_sha256=indexed_file_sha256,
    )
    return FreshnessResponse(
        chunk_id=verdict.chunk_id,
        fresh=verdict.fresh,
        reason=verdict.reason,
        current_snippet=verdict.current_snippet,
        expected_snippet=verdict.expected_snippet,
        drift_details=list(verdict.drift_details),
        source=verdict.source,
    )


def test_impact_impl(
    symbols: list[str],
    app: AppContext,
    *,
    transitive_depth: int,
    runner: str,
    graph_loaded_fn: GraphLoaded,
    repo_id: int | None,
) -> TestImpactResponse:
    from .retrieval.graph import find_callers, find_tests_for
    from .retrieval.test_impact import analyse_test_impact

    graph_ok = graph_loaded_fn(app.storage, repo_id=repo_id)
    if not graph_ok:
        return TestImpactResponse(
            changed_symbols=[s.strip() for s in symbols if s and s.strip()],
            graph_loaded=False,
            run_command="",
            notes=["graph_loaded=False"],
        )
    report = analyse_test_impact(
        symbols,
        tests_fn=lambda symbol: find_tests_for(app.storage, symbol, repo_id=repo_id),
        callers_fn=lambda symbol: find_callers(app.storage, symbol, repo_id=repo_id),
        transitive_depth=max(0, int(transitive_depth)),
        runner=str(runner) if runner else "pytest",
    )
    return TestImpactResponse(
        changed_symbols=report.changed_symbols,
        tests=[ImpactedTestEntry(path=test.path, symbol=test.symbol, via=list(test.via)) for test in report.tests],
        per_symbol=dict(report.per_symbol),
        notes=list(report.notes),
        run_command=report.run_command,
        graph_loaded=True,
    )


def audit_citations_impl(
    text: str,
    app: AppContext,
    *,
    repo_id: int | None,
) -> CitationAuditResponse:
    from .retrieval.citation import Citation, audit_citations

    def lookup(citation: Citation) -> bool:
        if citation.kind == "chunk_id":
            try:
                return (
                    app.storage.lookup_by_chunk_id(citation.chunk_id) is not None
                    and chunk_belongs_to_repo(app.storage, citation.chunk_id, repo_id)
                )
            except Exception:
                return False
        if citation.kind == "file_line":
            try:
                if repo_id is None:
                    hits = app.storage.lookup_by_symbol(citation.path, limit=20)
                else:
                    hits = app.storage.lookup_by_symbol(
                        citation.path, limit=20, repo_id=repo_id
                    )
            except Exception:
                return False
            for hit in hits or []:
                if not chunk_belongs_to_repo(app.storage, hit.chunk_id, repo_id):
                    continue
                if getattr(hit, "rel_path", "") == citation.path:
                    start = int(getattr(hit, "start_line", 0) or 0)
                    end = int(getattr(hit, "end_line", 0) or 0)
                    if start <= citation.end_line and end >= citation.start_line:
                        return True
            return False
        return False

    report = audit_citations(text, lookup_fn=lookup)
    return CitationAuditResponse(
        claims=[
            ClaimAuditEntry(
                sentence=claim.sentence,
                citations=[
                    CitationEntry(
                        fragment=citation.fragment,
                        kind=citation.kind,
                        chunk_id=citation.chunk_id,
                        path=citation.path,
                        start_line=citation.start_line,
                        end_line=citation.end_line,
                        resolved=citation.resolved,
                        reason=citation.reason,
                    )
                    for citation in claim.citations
                ],
                valid_count=claim.valid_count,
            )
            for claim in report.claims
        ],
        total_claims=report.total_claims,
        cited_claims=report.cited_claims,
        uncited_claims=report.uncited_claims,
        broken_citations=list(report.broken_citations),
        notes=list(report.notes),
    )


def default_spec_loader(repo_root: str) -> dict[str, str]:
    out: dict[str, str] = {}
    root = Path(repo_root)
    candidates = [root / "CLAUDE.md", root / "AGENTS.md"]
    arch_dir = root / "docs" / "architecture"
    if arch_dir.is_dir():
        for path in sorted(arch_dir.glob("*.md")):
            candidates.append(path)
    for path in candidates:
        if not path.is_file():
            continue
        try:
            out[str(path.relative_to(root))] = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
    return out


def spec_consistency_impl(
    changed_symbols: list[str],
    changed_files: list[str] | None,
    *,
    repo_root: str,
    default_spec_loader_fn: Callable[[str], dict[str, str]],
) -> SpecConsistencyResponse:
    from .retrieval.spec_consistency import check_spec_consistency

    def loader() -> dict[str, str]:
        if not repo_root:
            return {}
        return default_spec_loader_fn(repo_root)

    report = check_spec_consistency(changed_symbols, changed_files or [], spec_loader=loader)
    return SpecConsistencyResponse(
        invariants_found=report.invariants_found,
        conflicts=[
            SpecConflictEntry(
                invariant=SpecInvariantEntry(
                    text=conflict.invariant.text,
                    spec_file=conflict.invariant.spec_file,
                    line=conflict.invariant.line,
                    keyword=conflict.invariant.keyword,
                    severity=conflict.invariant.severity,
                ),
                matched=conflict.matched,
                severity=conflict.severity,
                reason=conflict.reason,
            )
            for conflict in report.conflicts
        ],
        notes=list(report.notes),
        spec_files_scanned=list(report.spec_files_scanned),
    )


def contract_tests_impl(
    symbol: str,
    app: AppContext,
    *,
    graph_loaded_fn: GraphLoaded,
    repo_id: int | None,
) -> ContractTestsResponse:
    from .retrieval.contract_tests import surface_contract_tests
    from .retrieval.graph import find_tests_for

    graph_ok = graph_loaded_fn(app.storage, repo_id=repo_id)
    if not graph_ok:
        return ContractTestsResponse(
            target_symbol=symbol,
            graph_loaded=False,
            notes=["graph_loaded=False"],
        )
    report = surface_contract_tests(
        symbol,
        tests_fn=lambda value: find_tests_for(app.storage, value, repo_id=repo_id),
    )
    return ContractTestsResponse(
        target_symbol=report.target_symbol,
        contract_tests=[
            ContractTestEntry(path=test.path, symbol=test.symbol, signals=list(test.signals))
            for test in report.contract_tests
        ],
        other_tests=[
            ContractTestEntry(path=test.path, symbol=test.symbol, signals=list(test.signals))
            for test in report.other_tests
        ],
        signal_counts=dict(report.signal_counts),
        notes=list(report.notes),
        graph_loaded=True,
    )
