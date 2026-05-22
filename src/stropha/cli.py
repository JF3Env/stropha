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
    effective_concurrency = (
        concurrency if concurrency is not None
        else cfg.inference.enricher_concurrency
    )
    effective_batch_size = (
        batch_size if batch_size is not None
        else cfg.inference.enricher_batch_size
    )

    # Everything resolved above (target repos, adapter ids from build_stages,
    # mode flags, inference backend, feature toggles) feeds into one panel so
    # the user sees the full state in a single glance.
    _print_index_summary(
        cfg, built, targets, mode,
        rebuild=rebuild, vacuum=vacuum,
        enricher_concurrency=effective_concurrency,
        enricher_batch_size=effective_batch_size,
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
) -> None:
    """Hybrid retrieval (dense + BM25 + symbol-token fused via RRF).
    
    Search local repos:
        stropha search "how does auth work"
        stropha search "auth" --language python --no-tests
    
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

    table = Table(title=f"Top {len(hits)} for {query!r}", show_lines=False)
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
    stropha_dir: Path | None = typer.Option(
        None,
        "--stropha-dir",
        help=(
            "Path to stropha project directory (for uv run). "
            "Defaults to the installed location."
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
    created, messages = create_stropha_dir(target)
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
                target,
                stropha_dir,
                server_name="stropha_rag",
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
                                console.print(f"    [dim]Already configured (use --force to overwrite)[/dim]")
                                continue
                        except Exception:
                            pass
                
                success, msg = configure_mcp_client(
                    client,
                    target,
                    stropha_dir,
                    server_name="stropha_rag",
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
        os.environ["STROPHA_INDEX_PATH"] = str(target / ".stropha" / "index.db")
        
        resolved = load_pipeline_config()
        try:
            built = build_stages(resolved)
            pipeline = Pipeline(
                built.walker, built.chunker, built.enricher, built.embedder, built.storage
            )
            with built.storage:  # type: ignore[union-attr]
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

    rows: list[tuple[str, object | None]] = [
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
        h = instance.health()  # type: ignore[union-attr]
        marker = {"ready": "✓", "warning": "⚠", "error": "✗"}.get(h.status, "?")
        table.add_row(
            stage_name,
            instance.adapter_name,  # type: ignore[union-attr]
            instance.adapter_id,  # type: ignore[union-attr]
            f"{marker} {h.status}",
            h.message,
        )
    console.print(table)
    if built.storage is not None:
        built.storage.close()  # type: ignore[union-attr]


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
    index_path: Path = typer.Option(
        None, "--index-path",
        help=(
            "Bake STROPHA_INDEX_PATH into the hook so the index step doesn't "
            "inherit from --project-dir/.env. Recommended when --project-dir "
            "differs from --target."
        ),
    ),
    log_path: Path = typer.Option(
        None, "--log-path",
        help=(
            "Bake the hook log file path. Useful for separating per-repo "
            "refresh trails. Defaults to ~/.cache/stropha-hook.log."
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

    try:
        result = install(
            repo, force=force,
            project_dir=project_dir,
            index_path=index_path,
            log_path=log_path,
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
    if result.get("index_path"):
        baked_lines.append(f"  • index_path:  [cyan]{result['index_path']}[/cyan]")
    if log_path is not None:
        baked_lines.append(f"  • log_path:    [cyan]{result['log_path']}[/cyan]")
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
    if s.project_dir:
        table.add_row("Baked project_dir", str(s.project_dir))
    if s.index_path:
        table.add_row("Baked index_path", str(s.index_path))
    if s.log_path_default:
        table.add_row("Baked log_path", str(s.log_path_default))
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
) -> None:
    """List the top candidates by fitness with their score breakdown."""
    from .meta import ArtifactStore

    if meta_dir is None:
        meta_dir = _meta_dir_default()
    meta_dir = meta_dir.expanduser().resolve()
    if not (meta_dir / "candidates").is_dir():
        console.print(f"[yellow]No candidates at:[/yellow] {meta_dir}")
        raise typer.Exit(code=0)
    store = ArtifactStore(meta_dir)
    candidates = store.list_candidates(limit=limit, sort_by=sort_by)

    if json_out:
        import json as _json
        console.print_json(_json.dumps([c.to_dict() if hasattr(c, "to_dict") else {
            "id": c.id, "label": c.label, "scores": c.scores,
            "fitness": c.fitness(),
        } for c in candidates]))
        return

    table = Table(title=f"Top {limit} candidates ({sort_by})")
    table.add_column("ID", style="cyan")
    table.add_column("Label")
    table.add_column("Fitness", justify="right")
    table.add_column("Quality", justify="right")
    table.add_column("Latency_ms", justify="right")
    for c in candidates:
        table.add_row(
            c.id,
            c.label or "—",
            f"{c.fitness():+.4f}",
            f"{c.scores.get('quality', 0):.3f}" if c.scores else "—",
            f"{c.scores.get('latency_ms', 0):.1f}" if c.scores else "—",
        )
    console.print(table)


@meta_app.command("promote")
def meta_promote_cmd(
    candidate_id: str = typer.Argument(..., help="Candidate id to promote."),
    meta_dir: Path = typer.Option(None, "--meta-dir"),
    dry_run: bool = typer.Option(False, "--dry-run",
                                  help="Show the candidate's config but do NOT write stropha.yaml."),
) -> None:
    """Promote a candidate's config to the production ``stropha.yaml``.

    Writes the chosen candidate's ``config.yaml`` to the cwd's
    ``stropha.yaml`` (overwriting it). The operator should review the
    diff before committing.
    """
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

    target = Path.cwd() / "stropha.yaml"
    if dry_run:
        console.print(f"[yellow]DRY RUN[/yellow] — would write to {target}")
        import yaml as _yaml  # type: ignore[import-untyped]
        console.print(_yaml.safe_dump(rec.config, sort_keys=True))
        return
    import yaml as _yaml  # type: ignore[import-untyped]
    target.write_text(_yaml.safe_dump(rec.config, sort_keys=True))
    console.print(f"[green]Promoted[/green] {candidate_id} → {target}")


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


if __name__ == "__main__":
    app()
