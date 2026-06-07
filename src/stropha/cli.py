"""Typer CLI — composition root.

Subcommands:
- ``index``    : walk → chunk → enrich → embed → store (uses pipeline-adapters).
- ``search``   : embed query → hybrid top-k.
- ``stats``    : print index metadata.
- ``pipeline`` : introspect the resolved pipeline (``show``, ``validate``).
- ``adapters`` : list available adapters per stage.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from . import __version__
from .config import Config, get_config
from .errors import StrophaError
from .logging import configure_logging, get_logger
from .pipeline import (
    Pipeline,
    all_adapters,
    build_stages,
    load_pipeline_config,
)

# Note: We don't call load_dotenv() here anymore. The get_config() singleton
# handles .env loading with proper precedence (env vars > .env file).
app = typer.Typer(
    name="stropha",
    help="stropha — index a codebase and serve it to LLM clients via MCP.",
    no_args_is_help=True,
    add_completion=False,
)
console = Console()
log = get_logger(__name__)


def _load_config() -> Config:
    """Get the singleton config instance."""
    try:
        return get_config()
    except Exception as exc:
        raise typer.BadParameter(f"Config error: {exc}") from exc


def _format_size(num_bytes: int) -> str:
    """Render a byte count in the largest fitting unit."""
    units = ("B", "KB", "MB", "GB")
    size = float(num_bytes)
    idx = 0
    while size >= 1024 and idx < len(units) - 1:
        size /= 1024
        idx += 1
    return f"{size:.1f} {units[idx]}"


def _add_batch_row(table, batch_size: int, enricher_name: str) -> None:
    """Add the ``batch size`` row to the inference panel.

    Only the ``contextual`` enricher implements batching today. For other
    enrichers we surface the configured value with a note that the knob is
    inactive — so the user is not confused about why throughput hasn't
    changed when they bumped the setting against e.g. the ``ollama`` enricher.
    """
    if enricher_name == "contextual" and batch_size > 1:
        # Effective throughput multiplier under the same rate budget.
        table.add_row(
            "  batch size",
            f"[green]{batch_size}[/green] chunks/request  "
            f"[dim](×{batch_size} throughput vs per-chunk)[/dim]",
        )
    elif batch_size > 1:
        table.add_row(
            "  batch size",
            f"[dim]{batch_size} (configured)[/dim]  "
            f"[yellow]ignored — {enricher_name} enricher has no batch support[/yellow]",
        )
    else:
        table.add_row(
            "  batch size",
            "1  [dim](per-chunk request)[/dim]",
        )


def _print_index_summary(
    cfg: Config,
    built,
    targets: list[Path],
    mode: str,
    *,
    rebuild: bool,
    vacuum: bool,
    enricher_concurrency: int = 1,
    enricher_batch_size: int = 1,
    graphify: bool = True,
) -> None:
    """Print a consolidated configuration table at the start of `stropha index`.

    Shows the user every knob that will be in effect for this run — index path
    + size, target repos, mode (full / incremental / auto + rebuild flag),
    every pipeline stage with its adapter_id, the LLM inference backend with
    its model + timeout + retry policy, query-time feature toggles, and graph
    integration settings. Replaces the 4 scattered ``console.print`` lines of
    the previous "Target / Index / Embedder / Enricher" output with a single
    auditable view.

    The structlog ``pipeline.adapter.built`` events still fire — this panel
    is the human-facing summary; the JSON log stream is the machine-facing
    one. Both stay in sync via the same ``built.<stage>.adapter_id`` source.
    """
    index_path = cfg.resolve_index_path()
    # DB stats are cheap (single SQL count) — fetch them so the panel shows
    # current state of the index file before we mutate it.
    try:
        stats = built.storage.stats()
        db_chunks = stats.get("chunks", "?")
        db_files = stats.get("files", "?")
        db_size = _format_size(int(stats.get("size_bytes", 0)))
    except Exception:
        db_chunks = db_files = "?"
        db_size = _format_size(index_path.stat().st_size) if index_path.exists() else "0 B"

    # Left col: tight label (no wrap). Right col: flexes + wraps long values
    # (overflow=fold) so the panel never truncates URLs / adapter_ids with `…`.
    table = Table.grid(padding=(0, 2))
    table.add_column(style="bold cyan", no_wrap=True, justify="left")
    table.add_column(overflow="fold", ratio=1)

    def section(label: str, hint: str = "") -> None:
        """Add a vertically-spaced section header.

        ``label`` lands in the (tight) left column. ``hint`` is an optional
        clarifier rendered dim in the value column on the same row, so the
        section divider doubles as inline documentation without widening
        the label column.
        """
        table.add_row("", "")  # blank spacer line
        table.add_row(
            f"[bold yellow]{label}[/bold yellow]",
            f"[dim]{hint}[/dim]" if hint else "",
        )

    # ---- Resources ------------------------------------------------------
    targets_str = "\n".join(str(t) for t in targets) if len(targets) > 1 else str(targets[0])
    table.add_row(
        "Target",
        targets_str + (f"  [dim]({len(targets)} repos)[/dim]" if len(targets) > 1 else ""),
    )
    table.add_row("Index DB", str(index_path))
    table.add_row(
        "  state",
        f"[dim]{db_chunks} chunks · {db_files} files · {db_size}[/dim]",
    )

    # ---- Run mode -------------------------------------------------------
    mode_flags = [f"[green]{mode}[/green]"]
    if rebuild:
        mode_flags.append("[red]rebuild[/red] (wipes chunks)")
    if vacuum:
        mode_flags.append("vacuum (reclaim disk)")
    table.add_row("Mode", " · ".join(mode_flags))

    # ---- Pipeline stages -----------------------------------------------
    section("Pipeline")
    table.add_row("Walker", f"{built.walker.adapter_name}  [dim]({built.walker.adapter_id})[/dim]")
    table.add_row("Chunker", f"{built.chunker.adapter_name}  [dim]({built.chunker.adapter_id})[/dim]")
    table.add_row("Enricher", f"{built.enricher.adapter_name}  [dim]({built.enricher.adapter_id})[/dim]")
    table.add_row(
        "Embedder",
        f"{built.embedder.adapter_name}:{built.embedder.model_name}  "
        f"[dim](dim={built.embedder.dim})[/dim]",
    )
    table.add_row("Storage", f"{built.storage.adapter_name}  [dim]({built.storage.adapter_id})[/dim]")
    table.add_row(
        "Retrieval",
        f"{built.retrieval.adapter_name}  [dim]({built.retrieval.adapter_id})[/dim]",
    )

    # ---- Inference (only meaningful when enricher actually calls LLM) --
    enricher_uses_llm = built.enricher.adapter_name in {"contextual", "ollama", "mlx"}
    if enricher_uses_llm:
        section("Inference", "LLM for enricher / HyDE / query rewrite")
        backend = cfg.inference.backend or "auto"
        if backend == "deepseek":
            table.add_row(
                "Backend",
                f"[magenta]deepseek[/magenta] [dim](cloud)[/dim]  →  "
                f"{cfg.inference.deepseek_model}",
            )
            table.add_row("  endpoint", cfg.inference.deepseek_base_url)
            api_key_set = bool(cfg.secrets.has_deepseek)
            table.add_row(
                "  api key",
                "[green]set[/green]" if api_key_set else "[red]MISSING[/red] — set DEEPSEEK_API_KEY",
            )
            table.add_row(
                "  resilience",
                f"timeout={cfg.inference.deepseek_timeout:.0f}s · "
                f"retries={cfg.inference.deepseek_max_retries} (exp backoff)",
            )
            concurrency_color = (
                "green" if enricher_concurrency > 1 else "yellow"
            )
            concurrency_hint = (
                "parallel" if enricher_concurrency > 1 else "serial — slow for cloud"
            )
            table.add_row(
                "  concurrency",
                f"[{concurrency_color}]{enricher_concurrency}[/{concurrency_color}] "
                f"in-flight  [dim]({concurrency_hint})[/dim]",
            )
            _add_batch_row(table, enricher_batch_size, built.enricher.adapter_name)
        elif backend == "anthropic":
            table.add_row(
                "Backend",
                f"[magenta]anthropic[/magenta] [dim](cloud)[/dim]  →  "
                f"{cfg.inference.anthropic_model}",
            )
            table.add_row("  endpoint", cfg.inference.anthropic_base_url)
            api_key_set = bool(cfg.secrets.has_anthropic)
            table.add_row(
                "  api key",
                "[green]set[/green]" if api_key_set else "[red]MISSING[/red] — set ANTHROPIC_API_KEY",
            )
            table.add_row(
                "  resilience",
                f"timeout={cfg.inference.anthropic_timeout:.0f}s · "
                f"retries={cfg.inference.anthropic_max_retries} (exp backoff)",
            )
            concurrency_color = (
                "green" if enricher_concurrency > 1 else "yellow"
            )
            concurrency_hint = (
                "parallel" if enricher_concurrency > 1 else "serial — slow for cloud"
            )
            table.add_row(
                "  concurrency",
                f"[{concurrency_color}]{enricher_concurrency}[/{concurrency_color}] "
                f"in-flight  [dim]({concurrency_hint})[/dim]",
            )
            _add_batch_row(table, enricher_batch_size, built.enricher.adapter_name)
        elif backend == "mlx":
            table.add_row(
                "Backend",
                f"[green]mlx[/green] [dim](local, Apple Silicon)[/dim]  →  "
                f"{cfg.inference.mlx_model}",
            )
            if enricher_concurrency > 1:
                table.add_row(
                    "  concurrency",
                    f"[yellow]{enricher_concurrency}[/yellow] in-flight  "
                    "[dim](local GPU saturates — parallelism rarely helps)[/dim]",
                )
            else:
                table.add_row("  concurrency", "1  [dim](serial — fits local GPU)[/dim]")
        elif backend == "ollama":
            table.add_row(
                "Backend",
                f"[green]ollama[/green] [dim](local)[/dim]  →  "
                f"{cfg.inference.ollama_model}",
            )
            table.add_row("  endpoint", cfg.inference.ollama_host)
            if enricher_concurrency > 1:
                table.add_row(
                    "  concurrency",
                    f"[yellow]{enricher_concurrency}[/yellow] in-flight  "
                    "[dim](Ollama queues internally — minimal gain)[/dim]",
                )
        else:
            table.add_row("Backend", f"{backend} [dim](auto-detect: mlx → ollama)[/dim]")

    # ---- Query-time features -------------------------------------------
    rf = cfg.retrieval_features
    enabled_features = []
    if rf.hyde_enabled:
        enabled_features.append("hyde" + ("(code-tuned)" if rf.hyde_code_tuned else ""))
    if rf.query_decomposition_enabled:
        enabled_features.append("decomp" + ("(llm)" if rf.query_decomposition_llm else ""))
    if rf.query_router_llm:
        enabled_features.append("router(llm)")
    if rf.query_rewrite_enabled:
        enabled_features.append("rewrite")
    if rf.multi_query_enabled:
        enabled_features.append(f"multi-query×{rf.multi_query_count}")
    if rf.recursive_enabled:
        enabled_features.append(f"recursive(adj={rf.recursive_adjacency})")
    if rf.cache_enabled:
        enabled_features.append(f"cache(n={rf.cache_size},ttl={int(rf.cache_ttl)}s)")
    if rf.adaptive_weighting:
        enabled_features.append("adaptive-weights")
    if enabled_features:
        section("Features", "query-time toggles (no re-index needed)")
        table.add_row("Enabled", " · ".join(enabled_features))

    # ---- Graph integration ---------------------------------------------
    section("Graph", "graphify symbol mirror")
    if graphify:
        from stropha.pipeline.graphify_runner import find_graphify
        gbin = find_graphify()
        if gbin:
            table.add_row(
                "Graphify",
                f"[green]enabled[/green]  [dim]({gbin})[/dim]",
            )
        else:
            table.add_row(
                "Graphify",
                "[yellow]enabled but binary not found[/yellow]  "
                "[dim](install graphify for graph tools)[/dim]",
            )
    else:
        table.add_row("Graphify", "[dim]disabled (--no-graphify)[/dim]")
    table.add_row(
        "Source",
        str(cfg.graph.graphify_out) if cfg.graph.graphify_out else "<repo>/graphify-out",
    )
    table.add_row(
        "Filter",
        f"confidence={cfg.graph.confidence} · "
        f"fts_augment={'on' if cfg.graph.fts_augment else 'off'}",
    )

    console.print()
    console.print(Panel(
        table,
        title="[bold]stropha index — configuration[/bold]",
        title_align="left",
        border_style="cyan",
        padding=(0, 1),
    ))
    console.print()


@app.callback()
def _root(
    log_level: str = typer.Option(
        None, "--log-level", help="Override STROPHA_LOG_LEVEL (DEBUG/INFO/WARNING)."
    ),
) -> None:
    """Initialize logging before any subcommand runs."""
    cfg = _load_config()
    configure_logging(log_level or cfg.log_level)


@app.command()
def index(
    repo: list[Path] = typer.Option(
        None,
        "--repo",
        "-r",
        help=(
            "Repository to index. Repeat for multi-repo indexing "
            "(e.g. `--repo /a --repo /b`). Falls back to STROPHA_TARGET_REPO "
            "when omitted."
        ),
    ),
    manifest: Path | None = typer.Option(
        None,
        "--manifest", "-m",
        help=(
            "Path to a YAML manifest with `repos: [{path: ...}]`. "
            "Mutually exclusive with --repo."
        ),
    ),
    rebuild: bool = typer.Option(
        False, "--rebuild", help="Clear the index before reindexing."
    ),
    vacuum: bool = typer.Option(
        False, "--vacuum",
        help=(
            "Reclaim unused disk space after indexing. SQLite ``DELETE FROM`` "
            "(used by --rebuild) marks pages as free but does not shrink the "
            "file. VACUUM rewrites the file to actually reduce disk usage. "
            "Slower on large databases."
        ),
    ),
    full: bool = typer.Option(
        False, "--full",
        help=(
            "Force a full walk (every file checked). "
            "Useful when --incremental defaults would otherwise apply."
        ),
    ),
    incremental: bool = typer.Option(
        False, "--incremental",
        help=(
            "Force git-diff incremental mode. Errors if no last_indexed_sha "
            "checkpoint exists (run a full index first, or pass --since)."
        ),
    ),
    since: str | None = typer.Option(
        None, "--since",
        help=(
            "Override last_indexed_sha for incremental mode. "
            "Useful to re-run against a specific checkpoint."
        ),
    ),
    enricher: str | None = typer.Option(
        None,
        "--enricher",
        help="Override pipeline.enricher.adapter (e.g. `noop`, `hierarchical`).",
    ),
    embedder: str | None = typer.Option(
        None,
        "--embedder",
        help="Override pipeline.embedder.adapter (e.g. `local`, `voyage`).",
    ),
    concurrency: int | None = typer.Option(
        None,
        "--concurrency", "-c",
        help=(
            "Number of concurrent enricher (LLM) calls. Overrides "
            "inference.enricher_concurrency from stropha.yaml. "
            "1 = serial (default, safe for local backends). 8 is a "
            "good starting point for cloud backends (deepseek / openai); "
            "raise if you have higher rate limits."
        ),
    ),
    batch_size: int | None = typer.Option(
        None,
        "--batch-size", "-b",
        help=(
            "Chunks packed into a single LLM call. Overrides "
            "inference.enricher_batch_size from stropha.yaml. 1 = "
            "per-chunk (legacy). N>1 multiplies effective throughput by "
            "N under the same rate-limit budget. Only the `contextual` "
            "enricher implements batching today; other enrichers ignore "
            "this flag. Suggested: 10 for Anthropic / DeepSeek / OpenAI."
        ),
    ),
    graphify: bool = typer.Option(
        True,
        "--graphify/--no-graphify",
        help=(
            "Run `graphify update` for each repo before loading the "
            "graph mirror. Creates graphify-out/graph.json on first run "
            "and refreshes it incrementally on subsequent runs. Enables "
            "the graph MCP tools (find_callers, find_related, etc.) "
            "without a separate graphify invocation. Gracefully skipped "
            "when the graphify binary is not installed. Default: enabled."
        ),
    ),
) -> None:
    """Walk one or more repos, chunk every file, enrich, embed, and store."""
    cfg = _load_config()
    if manifest and repo:
        console.print("[red]--manifest and --repo are mutually exclusive[/red]")
        raise typer.Exit(code=1)

    targets: list[Path] = []
    if manifest:
        from .ingest.manifest import ManifestError, load_manifest
        try:
            entries = load_manifest(manifest)
        except ManifestError as exc:
            console.print(f"[red]Manifest error:[/red] {exc}")
            raise typer.Exit(code=1) from exc
        targets = [e.path for e in entries]
        if not targets:
            console.print("[yellow]Manifest has no enabled repos[/yellow]")
            raise typer.Exit(code=1)
    else:
        targets = [p.resolve() for p in (repo or [])]
        if not targets:
            targets = [cfg.target_repo.resolve()]
    for t in targets:
        if not t.is_dir():
            console.print(f"[red]Target not a directory:[/red] {t}")
            raise typer.Exit(code=1)

    overrides: dict = {}
    if enricher:
        overrides.setdefault("enricher", {})["adapter"] = enricher
    if embedder:
        overrides.setdefault("embedder", {})["adapter"] = embedder
    resolved = load_pipeline_config(overrides=overrides or None)

    try:
        # Pass --rebuild through so the storage adapter can drop and
        # recreate dim-dependent tables (vec_chunks) when the embedder dim
        # changed between runs. Without this, the dim consistency check
        # raises BEFORE Pipeline.run(rebuild=True) gets a chance to clear
        # — leaving the user stuck in a loop where the error tells them
        # to run --rebuild but --rebuild is what they were already running.
        built = build_stages(resolved, force_rebuild=rebuild)
    except StrophaError as exc:
        console.print(f"[red]Pipeline build error:[/red] {exc}")
        raise typer.Exit(code=1) from exc

    if full and incremental:
        console.print("[red]--full and --incremental are mutually exclusive[/red]")
        raise typer.Exit(code=1)
    if full:
        mode = "full"
    elif incremental:
        mode = "incremental"
    else:
        mode = "auto"

    # Print the consolidated config summary right before the index loop starts.
    # Resolve enricher concurrency + batch size: CLI flag wins over YAML.
    # For batch_size, the YAML path is:
    #   per-backend override (e.g. ``deepseek_batch_size``) > global
    #   ``enricher_batch_size``. The CLI flag overrides both.
    effective_concurrency = (
        concurrency if concurrency is not None
        else cfg.inference.enricher_concurrency
    )
    effective_batch_size = (
        batch_size if batch_size is not None
        else cfg.inference.effective_batch_size()
    )

    # Everything resolved above (target repos, adapter ids from build_stages,
    # mode flags, inference backend, feature toggles) feeds into one panel so
    # the user sees the full state in a single glance.
    _print_index_summary(
        cfg, built, targets, mode,
        rebuild=rebuild, vacuum=vacuum,
        enricher_concurrency=effective_concurrency,
        enricher_batch_size=effective_batch_size,
        graphify=graphify,
    )

    try:
        with built.storage as storage:  # type: ignore[union-attr]
            pipeline = Pipeline(
                storage=storage,
                embedder=built.embedder,
                enricher=built.enricher,
                walker=built.walker,
                chunker=built.chunker,
                repos=targets,
                enricher_concurrency=effective_concurrency,
                enricher_batch_size=effective_batch_size,
                graphify=graphify,
            )
            stats = pipeline.run(rebuild=rebuild, mode=mode, since_sha=since)
            if vacuum:
                console.print("[bold]Vacuuming database...[/bold]")
                size_before, size_after = storage.vacuum()  # type: ignore[union-attr]
                reclaimed = size_before - size_after
                console.print(
                    f"[green]VACUUM done.[/green] "
                    f"{size_before / 1024 / 1024:.1f} MB -> "
                    f"{size_after / 1024 / 1024:.1f} MB "
                    f"(reclaimed {reclaimed / 1024 / 1024:.1f} MB)"
                )
        # Surface file-level cache hits prominently — they are the
        # biggest perf win of Phase A incremental.
        skipped_part = (
            f"{stats.files_skipped_fresh} files fresh · "
            if stats.files_skipped_fresh
            else ""
        )
        evicted_part = (
            f"{stats.files_evicted} evicted · "
            if stats.files_evicted
            else ""
        )
        console.print(
            f"[green]Done.[/green] {stats.files_visited} files · "
            f"{skipped_part}{evicted_part}"
            f"{stats.chunks_seen} chunks · "
            f"{stats.chunks_embedded} embedded · "
            f"{stats.chunks_skipped_fresh} reused · "
            f"{stats.chunks_enriched_from_cache} enrich-cache hits · "
            f"{stats.chunks_enriched_fresh} enrich-fresh "
            f"across {len(stats.repos)} repo"
            + ("s" if len(stats.repos) != 1 else "")
        )
        for r in stats.repos:
            extras = []
            if r.files_skipped_fresh:
                extras.append(f"{r.files_skipped_fresh} fresh")
            if r.files_evicted:
                extras.append(f"{r.files_evicted} evicted")
            extra = f" ({', '.join(extras)})" if extras else ""
            console.print(
                f"  [dim]·[/dim] {r.normalized_key}"
                + (f"  ([blue]{r.url}[/blue])" if r.url else "")
                + f"  — {r.files_visited} files, {r.chunks_embedded} embedded{extra}"
            )
    except StrophaError as exc:
        console.print(f"[red]Indexing failed:[/red] {exc}")
        raise typer.Exit(code=1) from exc
    except ValueError as exc:
        console.print(f"[red]Invalid input:[/red] {exc}")
        raise typer.Exit(code=1) from exc


def _apply_cli_filters(
    hits: list,
    *,
    language: list[str] | None = None,
    path_prefix: str | None = None,
    exclude_tests: bool = False,
) -> list:
    """Apply post-retrieval filters to search hits for CLI."""
    result = hits
    
    if language:
        lang_set = {lang.lower() for lang in language}
        result = [h for h in result if h.language.lower() in lang_set]
    
    if path_prefix:
        result = [h for h in result if h.rel_path.startswith(path_prefix)]
    
    if exclude_tests:
        test_patterns = ("test_", "_test.", ".test.", "/tests/", "/test/", "Test.")
        result = [
            h for h in result
            if not any(p in h.rel_path for p in test_patterns)
        ]
    
    return result


@app.command()
def search(
    query: str = typer.Argument(..., help="Natural-language or symbol query."),
    repo_url: str | None = typer.Argument(
        None,
        help="Optional git URL to search (e.g., https://github.com/org/repo).",
    ),
    top_k: int = typer.Option(10, "--top-k", "-k", min=1, max=50),
    recursive: bool = typer.Option(
        False,
        "--recursive", "-r",
        help="Enable auto-merging of sibling and adjacent chunks.",
    ),
    refresh: bool = typer.Option(
        False,
        "--refresh",
        help="Re-clone and re-index remote repository (for git URLs).",
    ),
    language: list[str] | None = typer.Option(
        None,
        "--language", "-l",
        help="Filter by language(s), e.g., -l python -l java.",
    ),
    path_prefix: str | None = typer.Option(
        None,
        "--path", "-p",
        help="Filter by path prefix, e.g., --path src/services/",
    ),
    exclude_tests: bool = typer.Option(
        False,
        "--no-tests",
        help="Exclude test files from results.",
    ),
    context_budget: int | None = typer.Option(
        None, "--context-budget",
        help=(
            "Token budget for result packing (WIN-208). When set, "
            "results are greedy-packed in score order up to this "
            "many estimated tokens. May return fewer than --top-k "
            "hits; surfaces the dropped tail count for visibility."
        ),
    ),
    compress: bool = typer.Option(
        False, "--compress",
        help=(
            "Extractive compression (WIN-210): keep each hit's "
            "signature + matched lines + small context window, drop "
            "boilerplate. Doubles useful-signal-per-token. Combine "
            "with --context-budget for max throughput."
        ),
    ),
) -> None:
    """Hybrid retrieval (dense + BM25 + symbol-token fused via RRF).

    Search local repos:
        stropha search "how does auth work"
        stropha search "auth" --language python --no-tests
        stropha search "auth" --context-budget 2000   # pack to 2K tokens

    Search remote repos via git URL:
        stropha search "auth" https://github.com/org/repo
        stropha search "auth" https://github.com/org/repo --refresh
    """
    import os

    from .tools.git_url import is_git_url, search_git_url

    # Set env var to enable recursive retrieval for this search
    if recursive:
        os.environ["STROPHA_RECURSIVE_RETRIEVAL"] = "1"

    # Fetch more candidates when filtering
    has_filters = any([language, path_prefix, exclude_tests])
    fetch_k = top_k * 3 if has_filters else top_k

    spilled_count = 0
    truncated_oversize = False
    try:
        # Handle git URL search
        if repo_url and is_git_url(repo_url):
            console.print(f"[dim]Searching remote repo: {repo_url}[/dim]")
            if refresh:
                console.print("[dim]Refreshing cache...[/dim]")
            hits = search_git_url(query, repo_url, top_k=fetch_k, refresh=refresh)
        else:
            # Local search
            resolved = load_pipeline_config()
            built = build_stages(resolved)
            with built.storage:  # type: ignore[union-attr]
                hits = built.retrieval.search(query, top_k=fetch_k)

        # Apply filters
        if has_filters:
            hits = _apply_cli_filters(
                hits,
                language=language,
                path_prefix=path_prefix,
                exclude_tests=exclude_tests,
            )
            hits = hits[:top_k]

        # WIN-210: Extractive compression — applied BEFORE budget
        # packing so the budget reflects compressed cost. The compressor
        # rewrites snippet in place; the displayed table picks up the
        # changes automatically.
        if compress and hits:
            from .retrieval.compression import compress_hits

            hits = compress_hits(hits, query)

        # WIN-208: token-budget packing — applied AFTER filtering + top_k
        # truncation so the budget controls the final response, not the
        # intermediate candidate set. Engine-level packing in
        # SearchEngine.search is the same logic; we invoke pack_to_budget
        # directly here to keep the CLI path linear and surface the
        # spilled count for the operator's table.
        if context_budget is not None and context_budget > 0:
            from .retrieval.packing import pack_to_budget

            pack = pack_to_budget(hits, context_budget)
            hits = pack.packed
            spilled_count = len(pack.spilled)
            truncated_oversize = pack.truncated
    except StrophaError as exc:
        console.print(f"[red]Search failed:[/red] {exc}")
        raise typer.Exit(code=1) from exc
    finally:
        # Clean up env var
        if recursive:
            os.environ.pop("STROPHA_RECURSIVE_RETRIEVAL", None)

    if not hits:
        console.print("[yellow]No results.[/yellow]")
        return

    table_title = f"Top {len(hits)} for {query!r}"
    if context_budget is not None:
        table_title += f" (budget {context_budget} tokens)"
    table = Table(title=table_title, show_lines=False)
    table.add_column("#", justify="right", style="cyan", no_wrap=True)
    table.add_column("Score", justify="right")
    table.add_column("Repo", overflow="fold", style="dim")
    table.add_column("Path:Lines", overflow="fold")
    table.add_column("Kind")
    table.add_column("Symbol", overflow="fold")
    table.add_column("Snippet", overflow="fold")
    for h in hits:
        repo_label = h.repo.normalized_key if h.repo else "—"
        # Truncate to last two path segments for compactness.
        if "/" in repo_label:
            parts = repo_label.split("/")
            if len(parts) > 2:
                repo_label = ".../" + "/".join(parts[-2:])
        table.add_row(
            str(h.rank),
            f"{h.score:.3f}",
            repo_label,
            f"{h.rel_path}:{h.start_line}-{h.end_line}",
            h.kind,
            h.symbol or "—",
            h.snippet.replace("\n", " ⏎ ")[:140],
        )
    console.print(table)
    # WIN-208 packing visibility — surfaces the dropped tail and the
    # single-first-hit-oversize escape so the operator knows when the
    # budget is tight.
    if spilled_count > 0:
        console.print(
            f"[dim]Packed {len(hits)} hits; {spilled_count} more dropped to "
            f"fit the {context_budget}-token budget.[/dim]"
        )
    if truncated_oversize:
        console.print(
            f"[yellow]⚠ Top hit alone exceeds the {context_budget}-token "
            f"budget — returned anyway. Raise --context-budget for more "
            f"results.[/yellow]"
        )


@app.command()
def init(
    target: Path = typer.Argument(
        Path("."),
        help="Target directory to initialize. Defaults to current directory.",
    ),
    mcp_client: str | None = typer.Option(
        None,
        "--mcp-client", "-m",
        help=(
            "MCP client to configure: 'claude-code', 'cursor', 'opencode', 'continue'. "
            "Auto-detects if not specified."
        ),
    ),
    mcp_url: str = typer.Option(
        "http://127.0.0.1:9741/mcp",
        "--mcp-url",
        help=(
            "URL of the shared stropha-mcp HTTP server. Defaults to the "
            "localhost singleton started by `bin/stropha-mcp-ensure`."
        ),
    ),
    run_index: bool = typer.Option(
        True,
        "--index/--no-index",
        help="Run initial index after setup.",
    ),
    force: bool = typer.Option(
        False,
        "--force", "-f",
        help="Overwrite existing MCP configuration.",
    ),
) -> None:
    """Initialize a project for use with stropha.
    
    This command:
    1. Creates .stropha/ directory with default .strophaignore
    2. Detects MCP clients and configures them (optional)
    3. Runs first index (optional)
    4. Prints next steps
    """
    from .tools.init_project import (
        MCP_CLIENTS,
        configure_mcp_client,
        create_stropha_dir,
        detect_mcp_clients,
        print_next_steps,
    )
    
    target = target.resolve()
    if not target.is_dir():
        console.print(f"[red]Error:[/red] {target} is not a directory")
        raise typer.Exit(code=1)
    
    console.print(f"[bold]Initializing stropha in {target}[/bold]\n")
    
    # Step 1: Create .stropha/ directory
    console.print("[dim]Step 1:[/dim] Creating .stropha/ directory...")
    _created, messages = create_stropha_dir(target)
    for msg in messages:
        console.print(f"  {msg}")
    
    # Step 2: Configure MCP client
    console.print("\n[dim]Step 2:[/dim] Configuring MCP client...")
    
    if mcp_client:
        # Find specified client
        client_map = {c.name.lower().replace(" ", "-"): c for c in MCP_CLIENTS}
        client = client_map.get(mcp_client.lower())
        if not client:
            console.print(f"[yellow]Unknown MCP client: {mcp_client}[/yellow]")
            console.print(f"  Available: {', '.join(client_map.keys())}")
        else:
            success, msg = configure_mcp_client(
                client,
                server_name="stropha_rag",
                url=mcp_url,
            )
            if success:
                console.print(f"  [green]{msg}[/green]")
            else:
                console.print(f"  [yellow]{msg}[/yellow]")
    else:
        # Auto-detect clients
        detected = detect_mcp_clients()
        if not detected:
            console.print("  [dim]No MCP clients detected[/dim]")
            console.print("  [dim]Configure manually after install[/dim]")
        else:
            for client in detected:
                console.print(f"  Found {client.name}...")
                if not force:
                    # Check if already configured
                    config_path = client.find_config()
                    if config_path and config_path.exists():
                        try:
                            import json
                            config = json.loads(config_path.read_text())
                            servers = config.get(client.key, {})
                            if "stropha_rag" in servers:
                                console.print("    [dim]Already configured (use --force to overwrite)[/dim]")
                                continue
                        except Exception:
                            pass

                success, msg = configure_mcp_client(
                    client,
                    server_name="stropha_rag",
                    url=mcp_url,
                )
                if success:
                    console.print(f"    [green]{msg}[/green]")
                else:
                    console.print(f"    [yellow]{msg}[/yellow]")
    
    # Step 3: Run initial index
    if run_index:
        console.print("\n[dim]Step 3:[/dim] Running initial index...")
        console.print("  [dim]This may take a moment for large repos...[/dim]")
        
        import os
        os.environ["STROPHA_TARGET_REPO"] = str(target)
        # STROPHA_INDEX_PATH removed: dead env var (read by no code path).
        # The unified YAML `paths.index` controls the DB location.

        resolved = load_pipeline_config()
        try:
            built = build_stages(resolved)
            pipeline = Pipeline(
                storage=built.storage,
                embedder=built.embedder,
                enricher=built.enricher,
                walker=built.walker,
                chunker=built.chunker,
                repos=[target],
            )
            with built.storage:
                stats_result = pipeline.run(rebuild=False)
            console.print(
                f"  [green]Indexed {stats_result.files_visited} files, "
                f"{stats_result.chunks_embedded} chunks[/green]"
            )
        except StrophaError as exc:
            console.print(f"  [red]Indexing failed:[/red] {exc}")
            console.print("  [dim]You can retry later with: stropha index[/dim]")
    else:
        console.print("\n[dim]Step 3:[/dim] Skipping initial index (--no-index)")
    
    # Step 4: Print next steps
    console.print("\n[bold]Next steps:[/bold]")
    steps = print_next_steps(target, indexed=run_index)
    for i, step in enumerate(steps, 1):
        console.print(f"  {i}. {step}")
    
    console.print("\n[green]Done![/green]")


@app.command()
def stats() -> None:
    """Print index metadata."""
    resolved = load_pipeline_config()
    try:
        built = build_stages(resolved)
        with built.storage as storage:  # type: ignore[union-attr]
            info = storage.stats()
    except StrophaError as exc:
        console.print(f"[red]Stats failed:[/red] {exc}")
        raise typer.Exit(code=1) from exc

    console.print(f"[bold]stropha[/bold] v{__version__}")
    console.print(f"DB         : {info['db_path']}")
    console.print(f"DB size    : {info['size_bytes'] / 1024:.1f} KB")
    console.print(f"Chunks     : {info['chunks']}")
    console.print(f"Files      : {info['files']}")
    console.print(f"Index dim  : {info['index_dim']}")
    if info["models"]:
        table = Table(title="Embedding models in index")
        table.add_column("Model")
        table.add_column("Dim", justify="right")
        table.add_column("Chunks", justify="right")
        for m in info["models"]:
            table.add_row(m["embedding_model"], str(m["embedding_dim"]), str(m["n"]))
        console.print(table)
    if len(info["models"]) > 1:
        console.print(
            "[yellow]Warning:[/yellow] index contains multiple embedding models. "
            "Run `stropha index --rebuild` to normalize."
        )
    if info.get("enrichers"):
        enrichers_table = Table(title="Enrichers in index (drift detection)")
        enrichers_table.add_column("Enricher adapter_id")
        enrichers_table.add_column("Chunks", justify="right")
        for e in info["enrichers"]:
            enrichers_table.add_row(e["enricher_id"], str(e["n"]))
        console.print(enrichers_table)
        cache = info.get("enrichment_cache_size", 0)
        console.print(f"[dim]Enrichment cache rows: {cache}[/dim]")
    if info["repos"]:
        repos_table = Table(title="Repositories in index")
        repos_table.add_column("Repo", overflow="fold")
        repos_table.add_column("Clone URL", overflow="fold")
        repos_table.add_column("Branch")
        repos_table.add_column("Files", justify="right")
        repos_table.add_column("Chunks", justify="right")
        repos_table.add_column("HEAD")
        for r in info["repos"]:
            head = (r["head_commit"] or "")[:8] or "—"
            repos_table.add_row(
                r["normalized_key"],
                r["url"] or "—",
                r["default_branch"] or "—",
                str(r["files"]),
                str(r["chunks"]),
                head,
            )
        console.print(repos_table)


@app.command()
def vacuum() -> None:
    """Reclaim unused disk space in the index file.

    SQLite's ``DELETE FROM`` (used by ``index --rebuild``) marks pages as
    free for reuse but does not shrink the file on disk. Run this command
    to rewrite the database file and reclaim disk space.

    Can be slow on large databases — VACUUM rewrites the entire file.
    """
    resolved = load_pipeline_config()
    try:
        built = build_stages(resolved)
        with built.storage as storage:  # type: ignore[union-attr]
            console.print("[bold]Vacuuming database...[/bold]")
            size_before, size_after = storage.vacuum()  # type: ignore[union-attr]
            reclaimed = size_before - size_after
            console.print(
                f"[green]VACUUM done.[/green] "
                f"{size_before / 1024 / 1024:.1f} MB -> "
                f"{size_after / 1024 / 1024:.1f} MB "
                f"(reclaimed {reclaimed / 1024 / 1024:.1f} MB)"
            )
    except StrophaError as exc:
        console.print(f"[red]Vacuum failed:[/red] {exc}")
        raise typer.Exit(code=1) from exc


# ---------------------------------------------------------------------------
# Pipeline introspection (Phase 1 — embedder + enricher are adapter-aware;
# other stages still appear as "legacy" until later phases migrate them).
# ---------------------------------------------------------------------------

pipeline_app = typer.Typer(
    name="pipeline",
    help="Inspect the resolved pipeline composition.",
    no_args_is_help=True,
)
app.add_typer(pipeline_app, name="pipeline")


@pipeline_app.command("show")
def pipeline_show(
    enricher: str | None = typer.Option(None, "--enricher"),
    embedder: str | None = typer.Option(None, "--embedder"),
    no_open: bool = typer.Option(
        False,
        "--no-open",
        help="Do not open the storage backend (skip storage + retrieval probes).",
    ),
) -> None:
    """Print the fully-resolved pipeline composition (YAML + env + CLI)."""
    overrides: dict = {}
    if enricher:
        overrides.setdefault("enricher", {})["adapter"] = enricher
    if embedder:
        overrides.setdefault("embedder", {})["adapter"] = embedder
    resolved = load_pipeline_config(overrides=overrides or None)
    try:
        built = build_stages(resolved, open_storage=not no_open)
    except StrophaError as exc:
        console.print(f"[red]Build error:[/red] {exc}")
        raise typer.Exit(code=1) from exc

    table = Table(title="Pipeline composition", show_lines=False)
    table.add_column("Stage", style="cyan")
    table.add_column("Adapter", style="bold")
    table.add_column("adapter_id", overflow="fold")
    table.add_column("Status")
    table.add_column("Notes", overflow="fold")

    # ``Any`` here because each row's runtime type is a different Stage
    # subclass; they all share the ``adapter_name`` / ``adapter_id`` /
    # ``health`` surface (enforced by the registry at build time) but
    # there is no common nominal supertype to express that statically
    # without dragging every Stage Protocol into a Union.
    rows: list[tuple[str, Any]] = [
        ("walker", built.walker),
        ("chunker", built.chunker),
        ("enricher", built.enricher),
        ("embedder", built.embedder),
        ("storage", built.storage),
        ("retrieval", built.retrieval),
    ]
    for stage_name, instance in rows:
        if instance is None:
            table.add_row(stage_name, "(skipped)", "—", "—", "use without --no-open to probe")
            continue
        h = instance.health()
        marker = {"ready": "✓", "warning": "⚠", "error": "✗"}.get(h.status, "?")
        table.add_row(
            stage_name,
            instance.adapter_name,
            instance.adapter_id,
            f"{marker} {h.status}",
            h.message,
        )
    console.print(table)
    if built.storage is not None:
        built.storage.close()


@pipeline_app.command("validate")
def pipeline_validate() -> None:
    """Run a lightweight health probe on every adapter. Exit non-zero on error."""
    resolved = load_pipeline_config()
    try:
        built = build_stages(resolved)
    except StrophaError as exc:
        console.print(f"[red]Build error:[/red] {exc}")
        raise typer.Exit(code=1) from exc

    worst_status = "ready"
    for instance in (
        built.walker,
        built.chunker,
        built.enricher,
        built.embedder,
        built.storage,
        built.retrieval,
    ):
        if instance is None:
            continue
        h = instance.health()
        marker = {"ready": "✓", "warning": "⚠", "error": "✗"}.get(h.status, "?")
        console.print(f"  {marker} {instance.stage_name}/{instance.adapter_name}: {h.message}")
        if h.status == "error":
            worst_status = "error"
        elif h.status == "warning" and worst_status == "ready":
            worst_status = "warning"
    if built.storage is not None:
        built.storage.close()
    raise typer.Exit(code=0 if worst_status != "error" else 1)


adapters_app = typer.Typer(
    name="adapters",
    help="Enumerate adapters registered for each pipeline stage.",
    no_args_is_help=True,
)
app.add_typer(adapters_app, name="adapters")


@adapters_app.command("list")
def adapters_list(
    stage: str | None = typer.Option(None, "--stage", help="Filter by stage."),
) -> None:
    """List every adapter the registry knows about."""
    registry = all_adapters()
    if stage:
        if stage not in registry:
            console.print(f"[yellow]No adapters registered for stage {stage!r}.[/yellow]")
            raise typer.Exit(code=1)
        registry = {stage: registry[stage]}

    table = Table(title="Available adapters", show_lines=False)
    table.add_column("Stage", style="cyan")
    table.add_column("Adapter", style="bold")
    for s, names in sorted(registry.items()):
        for n in names:
            table.add_row(s, n)
    console.print(table)


# ---------------------------------------------------------------------------
# Cost dashboard (Phase 3 §11.4 — aggregate structlog by adapter / repo)
# ---------------------------------------------------------------------------


@app.command()
def cost(
    log_path: Path = typer.Option(
        None, "--log-path",
        help="Override the hook log path. Defaults to ~/.cache/stropha-hook*.log",
    ),
    json_out: bool = typer.Option(
        False, "--json",
        help="Emit JSON instead of the rich table.",
    ),
) -> None:
    """Aggregate the structured log trail into a per-repo / per-adapter view."""
    from .cost import aggregate, default_log_paths, render_json, render_table

    paths = [log_path] if log_path else default_log_paths()
    if not paths:
        console.print("[yellow]No logs found. Has the hook ever run?[/yellow]")
        raise typer.Exit(code=1)

    report = aggregate(paths)
    if json_out:
        import sys
        sys.stdout.write(render_json(report))
        sys.stdout.write("\n")
        raise typer.Exit(code=0)
    rendered = render_table(report)
    if rendered is None:
        console.print(f"[yellow]Log present but contained no recognisable events:[/yellow] "
                      f"{', '.join(str(p) for p in paths)}")
        raise typer.Exit(code=1)
    console.print(f"[dim]Scanned: {', '.join(str(p) for p in paths)}[/dim]")
    console.print(rendered)


# ---------------------------------------------------------------------------
# Soft index (Phase 3 §8 — file watcher with debounce)
# ---------------------------------------------------------------------------


@app.command()
def watch(
    repo: Path = typer.Option(
        None, "--repo", "-r",
        help="Repo to watch. Defaults to STROPHA_TARGET_REPO or cwd.",
    ),
    interval: float = typer.Option(
        1.0, "--interval", min=0.1, max=60.0,
        help="Polling interval in seconds.",
    ),
    debounce: float = typer.Option(
        2.0, "--debounce", min=0.0, max=60.0,
        help="Wait this many seconds of no further changes before re-indexing.",
    ),
    full_refresh: bool = typer.Option(
        False, "--full-refresh",
        help="Run a full Pipeline pass on every cycle (slower but covers graphify reload).",
    ),
) -> None:
    """Watch a working tree and re-index on file changes.

    Press Ctrl-C to stop.
    """
    from .watch import watch_repo

    target = repo or _load_config().target_repo
    console.print(
        f"[bold]Watching:[/bold] {target.resolve()}\n"
        f"[dim]interval={interval}s · debounce={debounce}s · full_refresh={full_refresh}[/dim]\n"
        "[dim]Ctrl-C to stop.[/dim]"
    )
    try:
        watch_repo(
            target, interval_s=interval, debounce_s=debounce,
            full_refresh=full_refresh,
        )
    except StrophaError as exc:
        console.print(f"[red]Watch failed:[/red] {exc}")
        raise typer.Exit(code=1) from exc


# ---------------------------------------------------------------------------
# Evaluation harness (Phase 2 §16 — golden set + Recall@K + MRR)
# ---------------------------------------------------------------------------


@app.command()
def eval(
    golden: Path = typer.Option(
        Path("tests/eval/golden.jsonl"),
        "--golden", "-g",
        help="Path to JSONL golden file (one case per line).",
    ),
    top_k: int = typer.Option(10, "--top-k", "-k", min=1, max=100),
    tag: str | None = typer.Option(
        None, "--tag",
        help="Run only cases with this tag (e.g. `--tag retrieval`).",
    ),
    json_out: bool = typer.Option(
        False, "--json",
        help="Emit machine-readable JSON instead of the rich table.",
    ),
    track: bool = typer.Option(
        False, "--track",
        help="Append scores to JSONL log and check for regression (WIN-382).",
    ),
    track_log: Path | None = typer.Option(
        None, "--track-log",
        help="Custom path for the eval score log (default: ~/.stropha/eval-scores.jsonl).",
    ),
) -> None:
    """Run the offline eval harness against the golden dataset."""
    from .eval import load_golden, run_eval

    if not golden.is_file():
        console.print(f"[red]Golden file not found:[/red] {golden}")
        raise typer.Exit(code=1)

    try:
        cases = load_golden(golden)
    except ValueError as exc:
        console.print(f"[red]Golden file invalid:[/red] {exc}")
        raise typer.Exit(code=1) from exc

    if tag:
        cases = [c for c in cases if tag in c.tags]
        if not cases:
            console.print(f"[yellow]No cases with tag {tag!r}.[/yellow]")
            raise typer.Exit(code=1)

    resolved = load_pipeline_config()
    try:
        built = build_stages(resolved)
    except StrophaError as exc:
        console.print(f"[red]Pipeline build error:[/red] {exc}")
        raise typer.Exit(code=1) from exc

    try:
        with built.storage:  # type: ignore[union-attr]
            report = run_eval(cases, built.retrieval, top_k=top_k)
    except StrophaError as exc:
        console.print(f"[red]Eval failed:[/red] {exc}")
        raise typer.Exit(code=1) from exc

    if json_out:
        import json as _json

        payload = {
            "summary": report.summary(),
            "results": [
                {
                    "id": r.case.id, "query": r.case.query,
                    "tags": list(r.case.tags),
                    "hit": r.hit, "rank": r.rank_of_first_hit,
                    "matched_path": r.matched_path,
                    "matched_symbol": r.matched_symbol,
                }
                for r in report.results
            ],
            "by_tag": {
                t: sub.summary() for t, sub in report.by_tag().items()
            },
        }
        console.print_json(_json.dumps(payload))
        raise typer.Exit(code=0 if report.recall_at_k >= 0.85 else 2)

    table = Table(
        title=f"Evaluation — {report.n} cases (top_k={report.top_k})",
        show_lines=False,
    )
    table.add_column("Case", style="cyan")
    table.add_column("Query", overflow="fold")
    table.add_column("Hit", justify="center")
    table.add_column("Rank", justify="right")
    table.add_column("Tags", overflow="fold")
    for r in report.results:
        marker = "✓" if r.hit else "✗"
        rank_s = str(r.rank_of_first_hit) if r.rank_of_first_hit else "—"
        table.add_row(r.case.id, r.case.query[:80], marker, rank_s, ",".join(r.case.tags))
    console.print(table)

    summary = report.summary()
    console.print(
        f"[bold]Summary:[/bold] {summary['hits']}/{summary['cases']} hits · "
        f"recall@{report.top_k}={summary[f'recall@{report.top_k}']} · "
        f"mrr={summary['mrr']} · "
        f"ndcg@{report.top_k}={summary[f'ndcg@{report.top_k}']} · "
        f"precision@{report.top_k}={summary[f'precision@{report.top_k}']} · "
        f"map@{report.top_k}={summary[f'map@{report.top_k}']} · "
        f"latency_p95={summary['latency_p95_ms']}ms"
    )

    # WIN-172: every quality metric in the by-tag table so a regression
    # in one tag (e.g. retrieval ranking gets worse on graph queries
    # while recall stays high) is visible at a glance.
    by_tag = report.by_tag()
    if len(by_tag) > 1:
        tag_table = Table(title="By tag", show_lines=False)
        tag_table.add_column("Tag", style="cyan")
        tag_table.add_column("Cases", justify="right")
        tag_table.add_column(f"Recall@{report.top_k}", justify="right")
        tag_table.add_column("MRR", justify="right")
        tag_table.add_column(f"NDCG@{report.top_k}", justify="right")
        tag_table.add_column(f"Prec@{report.top_k}", justify="right")
        tag_table.add_column(f"MAP@{report.top_k}", justify="right")
        for t, sub in sorted(by_tag.items()):
            tag_table.add_row(
                t,
                str(sub.n),
                f"{sub.recall_at_k:.2f}",
                f"{sub.mrr:.2f}",
                f"{sub.ndcg_at_k:.2f}",
                f"{sub.precision_at_k:.2f}",
                f"{sub.map_at_k:.2f}",
            )
        console.print(tag_table)

    # WIN-168: surface split-aware metrics too (search vs holdout) so
    # the operator can spot overfitting at a glance.
    by_split = report.by_split()
    if len(by_split) > 1:
        split_table = Table(title="By split", show_lines=False)
        split_table.add_column("Split", style="cyan")
        split_table.add_column("Cases", justify="right")
        split_table.add_column(f"Recall@{report.top_k}", justify="right")
        split_table.add_column(f"NDCG@{report.top_k}", justify="right")
        split_table.add_column(f"MAP@{report.top_k}", justify="right")
        for s in sorted(by_split):
            sub = by_split[s]
            split_table.add_row(
                s,
                str(sub.n),
                f"{sub.recall_at_k:.2f}",
                f"{sub.ndcg_at_k:.2f}",
                f"{sub.map_at_k:.2f}",
            )
        console.print(split_table)

    # WIN-382: append scores to log and check for regression.
    if track:
        from .eval.tracker import append_score, check_regression

        append_score(report, log_path=track_log)
        is_regression, msg = check_regression(
            report.recall_at_k, log_path=track_log
        )
        if is_regression:
            console.print(f"[bold red]⚠ {msg}[/bold red]")
        elif not json_out:
            console.print(f"[green]{msg}[/green]")

    # Non-zero exit when below the spec threshold so CI catches regressions.
    raise typer.Exit(code=0 if report.recall_at_k >= 0.85 else 2)


# ---------------------------------------------------------------------------
# Hook management (RFC §7.1 — Trilha A 1.5c)
# ---------------------------------------------------------------------------

hook_app = typer.Typer(
    name="hook",
    help="Manage the post-commit hook (refreshes graphify graph + stropha index).",
    no_args_is_help=True,
)
app.add_typer(hook_app, name="hook")


def _resolve_hook_target(target: Path | None) -> Path:
    """Default to the configured target_repo, then cwd. Validate ``.git/`` exists."""
    if target is None:
        cfg = _load_config()
        target = cfg.target_repo or Path.cwd()
    target = target.expanduser().resolve()
    if not (target / ".git").exists():
        console.print(f"[red]Not a git repo:[/red] {target}")
        raise typer.Exit(code=1)
    return target


@hook_app.command("install")
def hook_install_cmd(
    target: Path = typer.Option(
        None, "--target", "-t",
        help="Repo to install into. Defaults to STROPHA_TARGET_REPO or cwd.",
    ),
    force: bool = typer.Option(
        False, "--force",
        help="Silence the warning when a graphify-hook coexists.",
    ),
    project_dir: Path = typer.Option(
        None, "--project-dir",
        help=(
            "Where stropha is installed (the directory with the `uv` venv). "
            "Bake into the hook so cross-repo installs work without manual "
            "env vars. Defaults to --target (dogfooding case)."
        ),
    ),
    # --index-path removed in v=9 (hook v=9 / config v=9). The hook now
    # honours the unified ~/.stropha/config.yaml `paths.index` — there
    # is no separate per-hook DB override.
    log_path: Path = typer.Option(
        None, "--log-path",
        help=(
            "Bake the hook log file path. Useful for separating per-repo "
            "refresh trails. Defaults to ~/.cache/stropha-hook.log."
        ),
    ),
    meta_iterations: int = typer.Option(
        0, "--meta-iterations",
        help=(
            "Sleep-time bandit iterations per commit (WIN-203). When > 0, "
            "the hook runs `stropha meta run --iterations N` in the same "
            "detached background as the index refresh. The foreground "
            "stays <100ms either way. 0 (default) disables the meta step."
        ),
    ),
) -> None:
    """Install or refresh the post-commit hook for the target repo."""
    from .tools.hook_install import install

    repo = _resolve_hook_target(target)

    # Light validation: warn (don't block) when --project-dir lacks the
    # usual project marker. Keeps the hook flexible while flagging obvious
    # typos.
    if project_dir is not None:
        resolved_project = project_dir.expanduser().resolve()
        if not (resolved_project / "pyproject.toml").is_file():
            console.print(
                f"[yellow]⚠ --project-dir {resolved_project} has no "
                f"pyproject.toml — `uv run` may fail at commit time.[/yellow]"
            )

    if meta_iterations < 0:
        console.print(
            "[red]--meta-iterations must be >= 0[/red] (0 disables the "
            "meta step). Got: " + str(meta_iterations)
        )
        raise typer.Exit(code=1)

    try:
        result = install(
            repo, force=force,
            project_dir=project_dir,
            log_path=log_path,
            meta_iterations=meta_iterations,
        )
    except (ValueError, OSError) as exc:
        console.print(f"[red]Install failed:[/red] {exc}")
        raise typer.Exit(code=1) from exc

    marker = {"created": "✓", "updated": "✓", "appended": "+"}.get(result["action"], "?")
    console.print(
        f"{marker} Hook {result['action']}: [cyan]{result['hook_path']}[/cyan]"
        f"  (v={result['version']})"
    )
    # Surface baked defaults so the user knows what got injected.
    baked_lines: list[str] = []
    if result.get("project_dir"):
        baked_lines.append(f"  • project_dir: [cyan]{result['project_dir']}[/cyan]")
    if log_path is not None:
        baked_lines.append(f"  • log_path:    [cyan]{result['log_path']}[/cyan]")
    if result.get("meta_iterations"):
        baked_lines.append(
            f"  • meta_iterations: [cyan]{result['meta_iterations']}[/cyan]"
            f" [dim](sleep-time bandit, WIN-203)[/dim]"
        )
    if baked_lines:
        console.print("Baked defaults:")
        for ln in baked_lines:
            console.print(ln)
    if result["coexist_warning"]:
        console.print(
            "[yellow]⚠ A `graphify hook` block was detected in the same file.[/yellow]\n"
            "  This stropha hook already calls `graphify update`; keeping both "
            "will duplicate work.\n"
            "  Either run `graphify hook uninstall` or pass `--force` to silence."
        )
    console.print(
        f"[dim]Logs: {result['log_path']}[/dim]\n"
        "Test with: [cyan]git commit --allow-empty -m 'hook test'[/cyan] then "
        f"[cyan]tail -f {result['log_path']}[/cyan]"
    )


@hook_app.command("uninstall")
def hook_uninstall_cmd(
    target: Path = typer.Option(
        None, "--target", "-t",
        help="Repo to uninstall from. Defaults to STROPHA_TARGET_REPO or cwd.",
    ),
) -> None:
    """Remove the stropha block from the post-commit hook."""
    from .tools.hook_install import uninstall

    repo = _resolve_hook_target(target)
    result = uninstall(repo)
    if result["action"] == "noop":
        console.print(f"[yellow]Nothing to do:[/yellow] {result['reason']}")
        raise typer.Exit(code=0)
    if result["action"] == "removed":
        console.print(f"[green]✓ Hook file removed:[/green] {result['hook_path']}")
    else:
        console.print(
            f"[green]✓ stropha block removed; rest of hook left intact:[/green] "
            f"{result['hook_path']}"
        )


@hook_app.command("status")
def hook_status_cmd(
    target: Path = typer.Option(
        None, "--target", "-t",
        help="Repo to inspect. Defaults to STROPHA_TARGET_REPO or cwd.",
    ),
) -> None:
    """Report whether the post-commit hook is installed."""
    from .tools.hook_install import HOOK_VERSION, status

    repo = _resolve_hook_target(target)
    s = status(repo)
    table = Table(title=f"Hook status — {repo}", show_lines=False)
    table.add_column("Field", style="cyan")
    table.add_column("Value", overflow="fold")
    table.add_row("Hook file", str(s.hook_path))
    if s.installed:
        ver_marker = "✓" if s.version == HOOK_VERSION else "⚠ outdated"
        table.add_row("Installed", f"yes (v={s.version}) {ver_marker}")
    else:
        table.add_row("Installed", "no")
    table.add_row("Graphify cohabit", "yes" if s.graphify_cohabit else "no")
    # v=3 baked defaults — show only the ones actually populated.
    # (v=9 dropped the index_path bake; see hook_install._render_hook_block.)
    if s.project_dir:
        table.add_row("Baked project_dir", str(s.project_dir))
    if s.log_path_default:
        table.add_row("Baked log_path", str(s.log_path_default))
    # v=6: sleep-time meta loop visibility. Show the value when present —
    # even 0 deserves a row so the operator knows the bake is the older
    # explicit "disabled" rather than missing.
    if s.meta_iterations is not None:
        meta_label = (
            f"{s.meta_iterations} iter/commit (sleep-time bandit)"
            if s.meta_iterations > 0
            else "disabled"
        )
        table.add_row("Meta loop", meta_label)
    # Effective log path: baked default wins, else legacy fallback.
    effective_log = s.log_path_default or s.log_path
    table.add_row("Log file", str(effective_log))
    if effective_log.is_file():
        try:
            size = effective_log.stat().st_size
            table.add_row("Log size", f"{size} bytes")
        except OSError:
            pass
    console.print(table)
    raise typer.Exit(code=0 if s.installed else 1)


# ============================================================================ #
#                           GLOSSARY SUB-APP                                   #
# ============================================================================ #

glossary_app = typer.Typer(
    name="glossary",
    help="Manage domain glossary (term definitions for better conceptual search).",
    no_args_is_help=True,
)
app.add_typer(glossary_app, name="glossary")


@glossary_app.command("add")
def glossary_add_cmd(
    term: str = typer.Argument(..., help="The term to add."),
    definition: str = typer.Argument(..., help="Definition of the term."),
    aliases: list[str] = typer.Option(
        [], "--alias", "-a",
        help="Alternate names for the term (can be repeated).",
    ),
) -> None:
    """Add a single term to the glossary."""
    from .ingest.glossary import GlossaryLoader
    
    resolved = load_pipeline_config()
    try:
        built = build_stages(resolved)
        with built.storage as storage:  # type: ignore[union-attr]
            loader = GlossaryLoader(storage)
            entry = loader.add_term(
                term=term,
                definition=definition,
                aliases=aliases if aliases else None,
                embedder=built.embedder,
                enricher_id=built.enricher.adapter_id if built.enricher else "noop",
            )
            storage._conn.commit()
            console.print(f"[green]Added:[/green] {entry.term}")
    except StrophaError as exc:
        console.print(f"[red]Failed:[/red] {exc}")
        raise typer.Exit(code=1) from exc


@glossary_app.command("remove")
def glossary_remove_cmd(
    term: str = typer.Argument(..., help="The term to remove."),
) -> None:
    """Remove a term from the glossary."""
    from .ingest.glossary import GlossaryLoader
    
    resolved = load_pipeline_config()
    try:
        built = build_stages(resolved)
        with built.storage as storage:  # type: ignore[union-attr]
            loader = GlossaryLoader(storage)
            if loader.remove_term(term):
                storage._conn.commit()
                console.print(f"[green]Removed:[/green] {term}")
            else:
                console.print(f"[yellow]Not found:[/yellow] {term}")
                raise typer.Exit(code=1)
    except StrophaError as exc:
        console.print(f"[red]Failed:[/red] {exc}")
        raise typer.Exit(code=1) from exc


@glossary_app.command("list")
def glossary_list_cmd() -> None:
    """List all glossary terms."""
    from .ingest.glossary import GlossaryLoader
    
    resolved = load_pipeline_config()
    try:
        built = build_stages(resolved)
        with built.storage as storage:  # type: ignore[union-attr]
            loader = GlossaryLoader(storage)
            terms = loader.list_terms()
    except StrophaError as exc:
        console.print(f"[red]Failed:[/red] {exc}")
        raise typer.Exit(code=1) from exc
    
    if not terms:
        console.print("[yellow]No glossary terms found.[/yellow]")
        return
    
    table = Table(title=f"Glossary ({len(terms)} terms)", show_lines=False)
    table.add_column("Term", style="cyan")
    table.add_column("Definition", overflow="fold")
    table.add_column("Aliases", overflow="fold", style="dim")
    
    for t in terms:
        aliases = ", ".join(t.get("aliases", [])) or "—"
        table.add_row(t["term"], t["definition"][:100], aliases)
    
    console.print(table)


@glossary_app.command("import")
def glossary_import_cmd(
    yaml_file: Path = typer.Argument(
        ...,
        help="Path to YAML file with glossary terms.",
        exists=True,
    ),
) -> None:
    """Import glossary terms from a YAML file."""
    from .ingest.glossary import GlossaryLoader
    
    resolved = load_pipeline_config()
    try:
        built = build_stages(resolved)
        with built.storage as storage:  # type: ignore[union-attr]
            loader = GlossaryLoader(storage)
            terms = loader.load_yaml(yaml_file)
            if not terms:
                console.print("[yellow]No terms found in file.[/yellow]")
                return
            
            indexed = loader.index_terms(
                terms,
                embedder=built.embedder,
                enricher_id=built.enricher.adapter_id if built.enricher else "noop",
            )
            storage._conn.commit()
            console.print(f"[green]Imported {indexed} terms from {yaml_file}[/green]")
    except StrophaError as exc:
        console.print(f"[red]Failed:[/red] {exc}")
        raise typer.Exit(code=1) from exc


@glossary_app.command("export")
def glossary_export_cmd(
    output: Path = typer.Argument(
        ...,
        help="Path to write the YAML file.",
    ),
) -> None:
    """Export glossary terms to a YAML file."""
    from .ingest.glossary import GlossaryLoader
    
    resolved = load_pipeline_config()
    try:
        built = build_stages(resolved)
        with built.storage as storage:  # type: ignore[union-attr]
            loader = GlossaryLoader(storage)
            count = loader.export_yaml(output)
            if count == 0:
                console.print("[yellow]No terms to export.[/yellow]")
            else:
                console.print(f"[green]Exported {count} terms to {output}[/green]")
    except StrophaError as exc:
        console.print(f"[red]Failed:[/red] {exc}")
        raise typer.Exit(code=1) from exc


@glossary_app.command("stats")
def glossary_stats_cmd() -> None:
    """Show glossary statistics."""
    from .ingest.glossary import GlossaryLoader
    
    resolved = load_pipeline_config()
    try:
        built = build_stages(resolved)
        with built.storage as storage:  # type: ignore[union-attr]
            loader = GlossaryLoader(storage)
            stats = loader.stats()
    except StrophaError as exc:
        console.print(f"[red]Failed:[/red] {exc}")
        raise typer.Exit(code=1) from exc
    
    table = Table(title="Glossary Statistics", show_lines=False)
    table.add_column("Metric", style="cyan")
    table.add_column("Value")
    
    table.add_row("Total terms", str(stats.total_terms))
    table.add_row("Last updated", stats.last_updated or "—")
    
    console.print(table)


# ============================================================================ #
#                       BENCH (config matrix) SUB-APP                          #
# ============================================================================ #

bench_app = typer.Typer(
    name="bench",
    help="Structured configuration-matrix benchmark (enumerate configs + compare).",
    no_args_is_help=True,
)
app.add_typer(bench_app, name="bench")


def _default_matrix_path() -> Path:
    from . import bench as _bench

    return Path(_bench.__file__).resolve().parent / "default_matrix.yaml"


@bench_app.command("run")
def bench_run_cmd(
    matrix: Path = typer.Option(
        None, "--matrix", "-m",
        help="Matrix YAML (default: bundled default_matrix.yaml).",
    ),
    out: Path = typer.Option(
        Path(".stropha/bench/results.jsonl"), "--out", "-o",
        help="Append-only results JSONL (resumable).",
    ),
    index_dir: Path = typer.Option(
        None, "--index-dir",
        help="Where per-index sqlite DBs live (default: <out dir>/indexes).",
    ),
    limit: int | None = typer.Option(
        None, "--limit", help="Stop after evaluating N new cells (smoke runs).",
    ),
    resume: bool = typer.Option(
        True, "--resume/--no-resume",
        help="Skip cells already present in the results file.",
    ),
    dry_run: bool = typer.Option(
        False, "--dry-run",
        help="Print the plan (cells + unique indexes) without running.",
    ),
) -> None:
    """Run a configuration matrix; append one result line per cell."""
    from .bench import load_matrix
    from .bench.matrix import MatrixError
    from .bench.runner import run_matrix

    mpath = matrix or _default_matrix_path()
    try:
        m = load_matrix(mpath)
    except MatrixError as exc:
        console.print(f"[red]Matrix error:[/red] {exc}")
        raise typer.Exit(code=1) from exc

    if dry_run:
        plan = run_matrix(m, out, index_dir=index_dir, dry_run=True)
        console.print(
            f"[bold]Plan:[/bold] {plan['total_cells']} cells · "
            f"{plan['unique_indexes']} unique indexes · "
            f"{plan['already_done']} already done"
        )
        for sig, labels in plan["groups"].items():
            console.print(f"  [cyan]{sig}[/cyan] — {len(labels)} query variants")
        return

    def _on_cell(seen: int, total: int, rec: dict) -> None:
        if rec.get("ok"):
            met = rec.get("metrics", {})
            shown = "  ".join(
                f"{k}={v:.3f}"
                for k, v in met.items()
                if isinstance(v, (int, float)) and "latency" not in k
            )
            console.print(f"[green]✓[/green] [{seen}/{total}] {rec['label']}  {shown}")
        else:
            console.print(
                f"[red]✗[/red] [{seen}/{total}] {rec.get('label', '?')} "
                f"— {str(rec.get('error', ''))[:80]}"
            )

    try:
        summary = run_matrix(
            m, out, index_dir=index_dir, resume=resume, limit=limit, on_cell=_on_cell,
        )
    except StrophaError as exc:
        console.print(f"[red]Bench failed:[/red] {exc}")
        raise typer.Exit(code=1) from exc
    console.print(
        f"[bold green]Done.[/bold green] evaluated={summary['evaluated']} "
        f"failed={summary['failed']} skipped={summary['skipped']} → {summary['out']}"
    )


@bench_app.command("report")
def bench_report_cmd(
    results: Path = typer.Argument(..., help="Results JSONL produced by `bench run`."),
    metric: str = typer.Option("recall@10", "--metric", help="Metric to rank / marginalise by."),
    baseline: str | None = typer.Option(
        None, "--baseline", help="cell_id to diff against (paired bootstrap CI).",
    ),
    top: int = typer.Option(20, "--top", help="How many ranked rows to show."),
    csv: Path | None = typer.Option(None, "--csv", help="Also write a flat CSV here."),
    marginals: bool = typer.Option(True, "--marginals/--no-marginals"),
) -> None:
    """Rank cells, show per-axis marginals, and optionally diff vs a baseline."""
    from .bench import BenchReport, load_results
    from .bench.report import ReportError, to_csv

    try:
        recs = load_results(results)
    except ReportError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=1) from exc
    if not recs:
        console.print("[yellow]No successful cells in results.[/yellow]")
        raise typer.Exit(code=1)

    rep = BenchReport(recs)
    metrics = rep.metric_names()
    if metric not in metrics:
        console.print(f"[yellow]metric {metric!r} not found; have: {metrics}[/yellow]")
        raise typer.Exit(code=1)

    table = Table(title=f"Top {top} of {len(recs)} cells by {metric}")
    table.add_column("id", style="dim")
    table.add_column("config", overflow="fold")
    for mn in metrics:
        table.add_column(mn, justify="right")
    for r in rep.rank_table(metric)[:top]:
        met = r.get("metrics", {})
        cells = [
            f"{met[mn]:.3f}" if isinstance(met.get(mn), (int, float)) else "-"
            for mn in metrics
        ]
        table.add_row(str(r.get("cell_id", ""))[:8], r.get("label", ""), *cells)
    console.print(table)

    if marginals:
        mt = Table(title=f"Per-axis marginal {metric} (mean over the other axes)")
        mt.add_column("axis", style="cyan")
        mt.add_column("value")
        mt.add_column(metric, justify="right")
        mt.add_column("n", justify="right")
        for axis, vals in rep.marginals(metric).items():
            for label, d in sorted(vals.items(), key=lambda kv: kv[1]["mean"], reverse=True):
                mt.add_row(axis, label, f"{d['mean']:.3f}", str(int(d["n"])))
        console.print(mt)

    if baseline:
        try:
            deltas = rep.baseline_deltas(baseline)
        except ReportError as exc:
            console.print(f"[yellow]{exc}[/yellow]")
            deltas = []
        if deltas:
            dt = Table(title=f"Δ vs baseline {baseline[:8]} (paired bootstrap, 95% CI)")
            dt.add_column("config", overflow="fold")
            dt.add_column("Δrecall", justify="right")
            dt.add_column("95% CI")
            dt.add_column("P>0", justify="right")
            dt.add_column("Δmrr", justify="right")
            dt.add_column("95% CI")
            dt.add_column("P>0", justify="right")
            for d in sorted(deltas, key=lambda x: x["d_recall"][0], reverse=True):
                pr, lor, hir, fpr = d["d_recall"]
                pm, lom, him, fpm = d["d_mrr"]
                dt.add_row(
                    d["label"],
                    f"{pr:+.3f}", f"[{lor:+.3f},{hir:+.3f}]", f"{fpr:.2f}",
                    f"{pm:+.3f}", f"[{lom:+.3f},{him:+.3f}]", f"{fpm:.2f}",
                )
            console.print(dt)

    if csv:
        csv.parent.mkdir(parents=True, exist_ok=True)
        csv.write_text(to_csv(recs), encoding="utf-8")
        console.print(f"[green]CSV →[/green] {csv}")


# ============================================================================ #
#                          META / OUTCOME SUB-APP                              #
# ============================================================================ #

meta_app = typer.Typer(
    name="meta",
    help="Meta-Harness apparatus (trace store, outcome labeling, backfill).",
    no_args_is_help=True,
)
outcome_app = typer.Typer(
    name="outcome",
    help="Outcome labeling against the trace stream.",
    no_args_is_help=True,
)
app.add_typer(meta_app, name="meta")
meta_app.add_typer(outcome_app, name="outcome")


@outcome_app.command("backfill")
def outcome_backfill_cmd(
    since: str = typer.Option(
        ...,
        "--since",
        help="Previous commit SHA; the diff is computed `since..HEAD`.",
    ),
    meta_dir: Path = typer.Option(
        None,
        "--meta-dir",
        help=(
            "Override the meta directory. Defaults to "
            "<index_path_parent>/meta/ next to the active index."
        ),
    ),
    repo_dir: Path = typer.Option(
        None,
        "--repo",
        help="Working tree to run git in. Defaults to STROPHA_TARGET_REPO or cwd.",
    ),
    json_out: bool = typer.Option(
        False,
        "--json",
        help="Emit the BackfillSummary as JSON to stdout.",
    ),
) -> None:
    """Label every trace since ``--since`` and persist the outcome summary.

    Reads:
    - The trace stream at ``<meta_dir>/traces/*.jsonl`` (produced by WIN-159).
    - The git diff ``since..HEAD`` (paths + heuristic symbol extraction).
    - The bad-result feedback at ``<meta_dir>/bad_results.jsonl`` (WIN-163).

    Writes:
    - ``<meta_dir>/outcomes/<head_sha>.json`` — the outcome summary the
      Phase 1 evaluator consumes (WIN-167).

    The post-commit hook v=5 runs this in its detached background process
    so ``git commit`` is never blocked.
    """
    from .meta.backfill import BackfillError, run_backfill

    cfg = _load_config()
    if repo_dir is None:
        repo_dir = cfg.target_repo or Path.cwd()
    repo_dir = repo_dir.expanduser().resolve()

    if meta_dir is None:
        index_path = Path(cfg.resolve_index_path()).expanduser().resolve()
        meta_dir = index_path.parent / "meta"
    meta_dir = meta_dir.expanduser().resolve()

    try:
        summary = run_backfill(
            since_sha=since,
            meta_dir=meta_dir,
            repo_dir=repo_dir,
        )
    except BackfillError as exc:
        console.print(f"[red]Backfill failed:[/red] {exc}")
        raise typer.Exit(code=1) from exc

    if json_out:
        import json as _json
        console.print_json(_json.dumps(summary.to_dict()))
        return

    table = Table(title="Outcome backfill", show_lines=False)
    table.add_column("Metric", style="cyan")
    table.add_column("Value")
    table.add_row("since_sha", summary.since_sha[:12])
    table.add_row("head_sha", summary.head_sha[:12])
    table.add_row("diff_paths", str(len(summary.diff_paths)))
    table.add_row("diff_symbols", str(len(summary.diff_symbols)))
    table.add_row("traces_scanned", str(summary.traces_scanned))
    table.add_row("hits_labeled", str(summary.hits_labeled))
    table.add_row("overall_score", f"{summary.overall_score:+.4f}")
    table.add_row("output", str(summary.output_path or "—"))
    console.print(table)


@meta_app.command("prune")
def meta_prune_cmd(
    meta_dir: Path = typer.Option(
        None,
        "--meta-dir",
        help=(
            "Override the meta directory. Defaults to "
            "<index_path_parent>/meta/ next to the active index."
        ),
    ),
    max_age_days: int = typer.Option(
        90,
        "--max-age-days",
        help="Remove trace JSONL files older than N days. Default 90.",
    ),
    keep_top: int = typer.Option(
        10,
        "--keep-top",
        help="Keep the top-N candidates by fitness. Default 10.",
    ),
    keep_bottom: int = typer.Option(
        10,
        "--keep-bottom",
        help="Keep the bottom-N candidates (anti-examples). Default 10.",
    ),
    keep_recent: int = typer.Option(
        50,
        "--keep-recent",
        help="Keep the most-recent-N candidates by created_at. Default 50.",
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help="Preview what would be removed; do not touch disk.",
    ),
    json_out: bool = typer.Option(
        False,
        "--json",
        help="Emit the PrunePlan as JSON to stdout.",
    ),
) -> None:
    """Prune `.stropha/meta/` per the WIN-166 disk-growth policy.

    Three policies, applied independently and unioned:

    1. Trace JSONL files older than --max-age-days are removed.
    2. Candidates outside (top-N ∪ bottom-N ∪ recent-N) are pruned
       (their directories removed; a `pruned: true` tombstone remains
       in `index.jsonl`).
    3. Disk usage is reported back so `stropha cost` can publish the
       per-run meta footprint.

    Pass --dry-run first to see what WOULD be removed.
    """
    from .meta.prune import prune as run_prune

    cfg = _load_config()
    if meta_dir is None:
        index_path = Path(cfg.resolve_index_path()).expanduser().resolve()
        meta_dir = index_path.parent / "meta"
    meta_dir = meta_dir.expanduser().resolve()

    if not meta_dir.is_dir():
        console.print(f"[yellow]Meta dir does not exist:[/yellow] {meta_dir}")
        raise typer.Exit(code=0)  # nothing to prune is not an error.

    plan = run_prune(
        meta_dir,
        max_age_days=max_age_days,
        keep_top=keep_top,
        keep_bottom=keep_bottom,
        keep_recent=keep_recent,
        dry_run=dry_run,
    )

    if json_out:
        import json as _json
        console.print_json(_json.dumps(plan.to_dict()))
        return

    mode = "DRY RUN" if dry_run else "PRUNED"
    table = Table(title=f"Meta prune — {mode}", show_lines=False)
    table.add_column("Metric", style="cyan")
    table.add_column("Value")
    table.add_row("meta_dir", str(plan.meta_dir))
    table.add_row("total_meta_bytes", _format_size(plan.total_meta_bytes))
    table.add_row("  traces", _format_size(plan.traces_dir_bytes))
    table.add_row("  outcomes", _format_size(plan.outcomes_dir_bytes))
    table.add_row("  candidates", _format_size(plan.candidates_dir_bytes))
    table.add_row("max_age_days", str(plan.max_age_days))
    table.add_row("traces_to_remove", str(len(plan.traces_to_remove)))
    table.add_row("candidates_to_prune", str(len(plan.candidates_to_prune)))
    table.add_row("bytes_freed", _format_size(plan.bytes_freed))
    console.print(table)


# ============================================================================ #
#                       META — Phase 4 LOOP COMMANDS (WIN-193)                 #
# ============================================================================ #

# ``meta_app`` is already declared earlier in this file for the outcome
# backfill command. We extend it here with the Phase 4 surface:
# ``stropha meta {run, status, report, promote, trace, snapshot}``.


def _meta_dir_default() -> Path:
    cfg = _load_config()
    index_path = Path(cfg.resolve_index_path()).expanduser().resolve()
    return index_path.parent / "meta"


@meta_app.command("run")
def meta_run_cmd(
    iterations: int = typer.Option(
        20, "--iterations", "-n",
        help="Number of Algorithm 1 iterations (default 20).",
    ),
    wall_clock_s: float | None = typer.Option(
        None, "--wall-clock",
        help="Wall-clock budget in seconds. Stops on whichever budget fires first.",
    ),
    proposer: str = typer.Option(
        "bandit", "--proposer",
        help="Proposer strategy: bandit | local-agentic | cloud-agentic.",
    ),
    seed: int | None = typer.Option(
        None, "--seed",
        help="RNG seed for the bandit + fallback paths (reproducibility).",
    ),
    meta_dir: Path = typer.Option(
        None, "--meta-dir",
        help="Override meta dir. Defaults to <index_parent>/meta/.",
    ),
) -> None:
    """Run the Meta-Harness outer loop (Algorithm 1, WIN-187).

    Loads the search space from the bundled manifest, instantiates the
    chosen proposer, and iterates ``--iterations`` times (or until
    ``--wall-clock`` expires). Persists every candidate via the
    ArtifactStore (WIN-161). Reports the final frontier.

    The evaluator is a stub identity function for now — the WIN-186 epic
    ships the loop infrastructure; the production evaluator wiring
    against a live SearchEngine lands in Phase 5 (WIN-195+).
    """
    from .meta import (
        ArtifactStore,
        ScoreCard,
        load_search_space,
        run_loop,
    )
    from .meta.proposer import (
        make_bandit_proposer,
        make_cloud_agentic_proposer,
        make_local_agentic_proposer,
    )

    if meta_dir is None:
        meta_dir = _meta_dir_default()
    meta_dir = meta_dir.expanduser().resolve()

    space = load_search_space()
    # ``prop`` holds whichever concrete proposer the user picked; all three
    # implement the ``ProposerFn`` shape (``proposer(archive) -> dict``) but
    # they don't share a nominal supertype, so widen via ``Any`` instead of
    # union-typing every concrete class at this seam.
    prop: Any
    if proposer == "bandit":
        prop = make_bandit_proposer(space, seed=seed)
    elif proposer == "local-agentic":
        prop = make_local_agentic_proposer(space, fallback_seed=seed)
    elif proposer == "cloud-agentic":
        prop = make_cloud_agentic_proposer(space, fallback_seed=seed)
    else:
        console.print(f"[red]Unknown proposer:[/red] {proposer}")
        raise typer.Exit(code=1)

    # Stub evaluator — returns a deterministic ScoreCard for now so the
    # loop's plumbing (proposer + persist + Pareto) is exercisable
    # without a live SearchEngine. The full evaluator wires in Phase 5.
    import random as _random
    rng = _random.Random(seed)

    def stub_eval(cfg: dict[str, Any]) -> ScoreCard:
        return ScoreCard(
            quality=rng.uniform(0.4, 0.95),
            context_efficiency=rng.uniform(0.3, 0.9),
            outcome_score=rng.uniform(0.0, 0.7),
            latency_ms=rng.uniform(10, 200),
            n_queries=20,
            n_hits=int(rng.uniform(10, 20)),
        )

    store = ArtifactStore(meta_dir)
    result = run_loop(
        proposer=prop,
        evaluator=stub_eval,
        store=store,
        max_iterations=iterations,
        wall_clock_s=wall_clock_s,
    )
    table = Table(title=f"stropha meta run — {result.stopped_reason}")
    table.add_column("Metric", style="cyan")
    table.add_column("Value")
    table.add_row("iterations", str(result.iterations))
    table.add_row("elapsed_s", f"{result.elapsed_s:.2f}")
    table.add_row("archive_size", str(len(result.archive)))
    table.add_row("frontier_size", str(len(result.frontier)))
    table.add_row("errors", str(len(result.errors)))
    console.print(table)


@meta_app.command("status")
def meta_status_cmd(
    meta_dir: Path = typer.Option(None, "--meta-dir"),
    json_out: bool = typer.Option(False, "--json"),
) -> None:
    """Summary of the current archive + Pareto frontier."""
    from .meta import ArtifactStore, ScoreCard, scorecard_pareto_frontier

    if meta_dir is None:
        meta_dir = _meta_dir_default()
    meta_dir = meta_dir.expanduser().resolve()
    if not (meta_dir / "candidates").is_dir():
        console.print(f"[yellow]No candidates at:[/yellow] {meta_dir}")
        raise typer.Exit(code=0)
    store = ArtifactStore(meta_dir)
    candidates = store.list_candidates()
    cards = [
        ScoreCard(**{k: v for k, v in c.scores.items() if k in {
            "quality", "context_efficiency", "outcome_score", "latency_ms",
        }})
        for c in candidates if c.scores
    ]
    frontier = scorecard_pareto_frontier(cards)
    if json_out:
        import json as _json
        console.print_json(_json.dumps({
            "archive_size": len(candidates),
            "frontier_size": len(frontier),
            "meta_dir": str(meta_dir),
        }))
        return
    console.print(
        f"[bold]archive:[/bold] {len(candidates)} candidates · "
        f"[bold]frontier:[/bold] {len(frontier)} non-dominated"
    )


@meta_app.command("report")
def meta_report_cmd(
    meta_dir: Path = typer.Option(None, "--meta-dir"),
    limit: int = typer.Option(10, "--limit", "-n",
                              help="Top-N candidates by fitness."),
    sort_by: str = typer.Option("fitness", "--sort-by",
                                 help="created_at | id | fitness | <score_key>"),
    json_out: bool = typer.Option(False, "--json"),
    regression_threshold: float = typer.Option(
        0.0, "--regression-threshold",
        help=(
            "Minimum positive fitness delta vs the live baseline that "
            "qualifies as a 'beats baseline' alert (WIN-204). Default 0 "
            "catches every strict improvement; raise to suppress noise."
        ),
    ),
) -> None:
    """List the top candidates by fitness with their score breakdown.

    When a baseline has been recorded (via ``meta promote --apply``),
    the report also flags every candidate that beats the live
    ``stropha.yaml`` on fitness — the regression-alert section that
    answers "is the bandit finding something better than what I just
    promoted?".
    """
    from .meta import (
        ArtifactStore,
        BaselineError,
        compare_candidates,
        read_baseline,
        regressions,
    )

    if meta_dir is None:
        meta_dir = _meta_dir_default()
    meta_dir = meta_dir.expanduser().resolve()
    if not (meta_dir / "candidates").is_dir():
        console.print(f"[yellow]No candidates at:[/yellow] {meta_dir}")
        raise typer.Exit(code=0)
    store = ArtifactStore(meta_dir)
    candidates = store.list_candidates(limit=limit, sort_by=sort_by)

    # Load the live baseline (if any) and pre-compute comparisons so
    # both JSON and table outputs see the same data.
    try:
        baseline = read_baseline(meta_dir)
    except BaselineError as exc:
        console.print(f"[red]Baseline read failed:[/red] {exc}")
        baseline = None
    comparisons = (
        compare_candidates(candidates, baseline) if baseline else []
    )
    delta_by_id = {c.candidate.id: c for c in comparisons}
    alerts = regressions(comparisons, threshold=regression_threshold)

    if json_out:
        import json as _json
        payload: dict[str, Any] = {
            "candidates": [
                c.to_dict() if hasattr(c, "to_dict") else {
                    "id": c.id, "label": c.label, "scores": c.scores,
                    "fitness": c.fitness(),
                }
                for c in candidates
            ],
            "baseline": baseline.to_dict() if baseline else None,
            "alerts": [
                {
                    "id": a.candidate.id,
                    "label": a.candidate.label,
                    "fitness": a.candidate_fitness,
                    "delta": a.delta,
                }
                for a in alerts
            ],
        }
        console.print_json(_json.dumps(payload))
        return

    table_title = f"Top {limit} candidates ({sort_by})"
    if baseline is not None:
        table_title += f" — baseline: {baseline.candidate_id}"
    table = Table(title=table_title)
    table.add_column("ID", style="cyan")
    table.add_column("Label")
    table.add_column("Fitness", justify="right")
    table.add_column("Quality", justify="right")
    table.add_column("Latency_ms", justify="right")
    if baseline is not None:
        table.add_column("Δ vs baseline", justify="right")
    for c in candidates:
        row = [
            c.id,
            c.label or "—",
            f"{c.fitness():+.4f}",
            f"{c.scores.get('quality', 0):.3f}" if c.scores else "—",
            f"{c.scores.get('latency_ms', 0):.1f}" if c.scores else "—",
        ]
        if baseline is not None:
            cmp = delta_by_id.get(c.id)
            if cmp is None:
                row.append("—")
            elif cmp.is_baseline:
                row.append("[dim]live[/dim]")
            elif cmp.delta > regression_threshold:
                row.append(f"[green]{cmp.delta:+.4f}[/green]")
            elif cmp.delta < -regression_threshold:
                row.append(f"[red]{cmp.delta:+.4f}[/red]")
            else:
                row.append(f"{cmp.delta:+.4f}")
        table.add_row(*row)
    console.print(table)

    if baseline is None:
        console.print(
            "[dim]No baseline recorded — run "
            "`stropha meta promote <id> --apply` to start tracking "
            "regressions.[/dim]"
        )
        return

    if alerts:
        alert_table = Table(
            title=(
                f"[bold yellow]⚠ Regression alerts[/bold yellow] — "
                f"{len(alerts)} candidate(s) beat baseline by "
                f"> {regression_threshold:.4f}"
            ),
        )
        alert_table.add_column("ID", style="cyan")
        alert_table.add_column("Label")
        alert_table.add_column("Δ vs baseline", justify="right",
                               style="green")
        alert_table.add_column("Candidate fitness", justify="right")
        alert_table.add_column("Baseline fitness", justify="right")
        for a in alerts:
            alert_table.add_row(
                a.candidate.id,
                a.candidate.label or "—",
                f"{a.delta:+.4f}",
                f"{a.candidate_fitness:+.4f}",
                f"{a.baseline_fitness:+.4f}",
            )
        console.print(alert_table)
    else:
        console.print(
            f"[green]✓ No regressions[/green] — baseline "
            f"{baseline.candidate_id} still leads "
            f"(top fitness vs baseline within {regression_threshold:.4f})."
        )


@meta_app.command("promote")
def meta_promote_cmd(
    candidate_id: str = typer.Argument(..., help="Candidate id to promote."),
    meta_dir: Path = typer.Option(None, "--meta-dir"),
    apply: bool = typer.Option(
        False,
        "--apply",
        help=(
            "Write the candidate config to stropha.yaml. Without this "
            "flag, promote prints the unified diff and exits — human "
            "gating is preserved (WIN-202)."
        ),
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help=(
            "Deprecated alias for the default diff-only behaviour. "
            "Kept for back-compat; emits a banner when used."
        ),
    ),
    target: Path = typer.Option(
        None,
        "--target",
        help=(
            "Path to the YAML file to compare / write against. Defaults "
            "to ./stropha.yaml in the current working directory."
        ),
    ),
) -> None:
    """Plan promotion of a Meta-Harness candidate to ``stropha.yaml``.

    Default behaviour (no ``--apply``) is **diff-only**: the unified diff
    between the current ``stropha.yaml`` (if any) and the candidate's
    config is printed to stdout. The operator can pipe the diff into
    ``git apply`` or copy-paste it for review — promotion stays
    human-gated.

    Pass ``--apply`` to overwrite ``stropha.yaml`` with the candidate
    config. The write is atomic (temp file + replace).
    """
    from .meta import ArtifactStore, PromotionError, apply_plan, build_plan

    if meta_dir is None:
        meta_dir = _meta_dir_default()
    meta_dir = meta_dir.expanduser().resolve()
    store = ArtifactStore(meta_dir)
    try:
        rec = store.read_candidate(candidate_id)
    except Exception as exc:
        console.print(f"[red]Cannot read candidate {candidate_id}:[/red] {exc}")
        raise typer.Exit(code=1) from exc

    if target is None:
        target = Path.cwd() / "stropha.yaml"
    target = target.expanduser().resolve()

    try:
        plan = build_plan(rec, target)
    except PromotionError as exc:
        console.print(f"[red]Cannot plan promotion:[/red] {exc}")
        raise typer.Exit(code=1) from exc

    if dry_run:
        console.print(
            "[yellow]--dry-run is deprecated[/yellow] — diff-only is now "
            "the default. Pass --apply to write."
        )

    if plan.is_noop:
        console.print(
            f"[yellow]No changes[/yellow] — candidate {candidate_id} "
            f"matches current {target}."
        )
        return

    if not apply:
        # Diff-only mode (default) — print the unified diff and exit.
        if plan.current_config is None:
            console.print(
                f"[cyan]New file[/cyan] — {target} does not exist; "
                f"diff shows the full candidate config as additions."
            )
        else:
            console.print(
                f"[cyan]Diff[/cyan] — candidate {candidate_id} vs {target}"
            )
        # Print the raw diff (no rich markup so it can be piped to git
        # apply / copy-pasted into a review comment).
        typer.echo(plan.diff_text, nl=False)
        console.print(
            f"\n[dim]Run with --apply to write to {target}.[/dim]"
        )
        return

    # Apply path — opt-in only.
    apply_plan(plan)
    # Record the promotion as the new baseline so subsequent `meta
    # report` runs surface candidates that beat the live config
    # (WIN-204). The baseline lives next to the meta dir as a
    # side-car JSON; existing prior records are atomically replaced.
    from .meta import write_baseline

    baseline = write_baseline(meta_dir, candidate=rec)
    console.print(
        f"[green]Promoted[/green] {candidate_id} → {target}"
    )
    if baseline.fitness is not None:
        console.print(
            f"[dim]Baseline recorded (fitness {baseline.fitness:+.4f}, "
            f"promoted_at {baseline.promoted_at}).[/dim]"
        )
    else:
        console.print(
            "[dim]Baseline recorded (no fitness — candidate had no scores).[/dim]"
        )


@meta_app.command("trace")
def meta_trace_cmd(
    candidate_id: str = typer.Argument(..., help="Candidate id to inspect."),
    meta_dir: Path = typer.Option(None, "--meta-dir"),
) -> None:
    """Show the trace files for one candidate."""
    from .meta import ArtifactStore

    if meta_dir is None:
        meta_dir = _meta_dir_default()
    meta_dir = meta_dir.expanduser().resolve()
    store = ArtifactStore(meta_dir)
    try:
        rec = store.read_candidate(candidate_id)
    except Exception as exc:
        console.print(f"[red]Cannot read candidate {candidate_id}:[/red] {exc}")
        raise typer.Exit(code=1) from exc

    if rec.base_dir is None:
        console.print(f"[red]No base_dir on candidate {candidate_id}[/red]")
        raise typer.Exit(code=1)
    traces_dir = rec.base_dir / "traces"
    if not traces_dir.is_dir():
        console.print(f"[yellow]No traces/ for[/yellow] {candidate_id}")
        return
    trace_files = sorted(traces_dir.glob("*.jsonl"))
    table = Table(title=f"Traces — {candidate_id}")
    table.add_column("File")
    table.add_column("Bytes", justify="right")
    for tf in trace_files:
        table.add_row(tf.name, str(tf.stat().st_size))
    console.print(table)


@meta_app.command("snapshot")
def meta_snapshot_cmd(
    ref: str = typer.Option("HEAD", "--ref",
                             help="Git ref to pin (HEAD, branch, SHA)."),
    repo: Path = typer.Option(None, "--repo"),
    list_only: bool = typer.Option(False, "--list",
                                    help="List existing snapshots and exit."),
) -> None:
    """Create or list git-worktree snapshots (WIN-175)."""
    from .meta import create_snapshot, list_snapshots

    cfg = _load_config()
    if repo is None:
        repo = cfg.target_repo or Path.cwd()
    repo = repo.expanduser().resolve()

    if list_only:
        root = repo / ".stropha" / "meta" / "snapshot" / "worktrees"
        snaps = list_snapshots(root)
        for sha in snaps:
            console.print(sha)
        return

    snap = create_snapshot(source_repo=repo, ref=ref)
    console.print(f"[green]Snapshot:[/green] {snap.sha} → {snap.path}")


# ---- memory consolidation (WIN-226, EPIC 1.9 #18) --------------------------


memory_app = typer.Typer(
    name="memory",
    help="Inspect and consolidate stropha's session / project memory.",
    no_args_is_help=True,
)
app.add_typer(memory_app, name="memory")


@memory_app.command("consolidate")
def memory_consolidate_cmd(
    repo: Path = typer.Option(
        None, "--repo",
        help="Target repo (defaults to STROPHA_TARGET_REPO).",
    ),
    min_episodes: int = typer.Option(
        3, "--min-episodes",
        help="Skip the pass when fewer than N episodes exist.",
    ),
    min_count: int = typer.Option(
        2, "--min-count",
        help="Minimum co-touch count for a neighbour to qualify.",
    ),
    top_n: int = typer.Option(
        5, "--top-n",
        help="Cap on neighbours per candidate (per category).",
    ),
    prune_after_days: int = typer.Option(
        90, "--prune-after-days",
        help="Delete episodes older than N days. 0 to disable pruning.",
    ),
    dry_run: bool = typer.Option(
        False, "--dry-run",
        help="Show what would be written + pruned, don't mutate.",
    ),
) -> None:
    """Compact episodic memory → project_memory (WIN-226).

    Mines recent episodes for high-frequency co-occurrence patterns,
    writes them as ``kind="co-touched"`` project_memory entries, and
    (optionally) prunes old episodes. Idempotent — running it
    repeatedly upserts the same entries with the latest counts.

    Recommended cron: weekly. The pass is fast (O(episodes_scanned ×
    distinct_keys)) on the corpora we target.
    """
    from .meta.consolidation import apply_consolidation, plan_consolidation
    from .pipeline import build_stages, load_pipeline_config

    cfg = _load_config()
    if repo is None:
        repo = cfg.target_repo or Path.cwd()
    repo = repo.expanduser().resolve()

    # Resolve the repo's row id via the existing storage path.
    from .ingest.git_meta import detect

    identity = detect(repo)
    resolved = load_pipeline_config()
    built = build_stages(resolved)
    with built.storage:  # type: ignore[union-attr]
        repo_id = built.storage.lookup_repo_id(identity.normalized_key)  # type: ignore[union-attr]
        if repo_id is None:
            console.print(
                f"[red]No indexed repo for[/red] {repo}.\n"
                "Run `stropha index` first to register the repo."
            )
            raise typer.Exit(code=1)

        prune_arg: int | None = (
            prune_after_days if prune_after_days > 0 else None
        )
        plan = plan_consolidation(
            built.storage,  # type: ignore[arg-type]
            repo_id=repo_id,
            min_episodes=min_episodes,
            min_count=min_count,
            top_n=top_n,
            prune_after_days=prune_arg,
        )

        if not plan.candidates and plan.episodes_to_prune == 0:
            console.print(
                "[yellow]No consolidation candidates and nothing to "
                "prune.[/yellow]"
            )
            return

        # Render the plan.
        console.print(
            f"[bold]Consolidation plan for {identity.normalized_key}:[/bold]"
        )
        console.print(
            f"  • {len(plan.candidates)} co-touched fact(s) to write."
        )
        if plan.prune_cutoff_iso is not None:
            console.print(
                f"  • {plan.episodes_to_prune} episode(s) older than "
                f"{prune_after_days} day(s) to prune."
            )
        if plan.candidates:
            sample = plan.candidates[:5]
            console.print("[dim]Sample candidates:[/dim]")
            for c in sample:
                preview = c.render_value()
                if len(preview) > 120:
                    preview = preview[:117] + "..."
                console.print(f"  - [cyan]{c.key}[/cyan]: {preview}")

        if dry_run:
            console.print("[yellow]Dry run — no changes written.[/yellow]")
            return

        result = apply_consolidation(built.storage, plan)  # type: ignore[arg-type]
        console.print(
            f"[green]Wrote[/green] {result['written']} fact(s); "
            f"[green]pruned[/green] {result['pruned']} episode(s)."
        )


if __name__ == "__main__":
    app()
