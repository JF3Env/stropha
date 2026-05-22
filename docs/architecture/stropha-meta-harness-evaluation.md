# Meta-Harness RFC — deep evaluation

> **Status:** Evaluation · 2026-05-22
> **Subject:** `docs/architecture/stropha-meta-harness.md` (RFC, Status `Proposed`, dated 2026-05-21)
> **Companion docs reviewed:** `docs/comparisons/stropha-vs-meta-harness.md`; `CLAUDE.md` (snapshot §2 + invariants §3); source tree at `HEAD` after the meta-harness merge.
> **Method:** the RFC's expected features were checked against the actual code, one phase at a time. Every claim is anchored to `file:line`. This document does not propose a new design — it stress-tests the proposed one.

---

## 0. TL;DR

The RFC is **structurally sound and shippable as-is** for Phases 0–2. The decomposition matches the paper (Lee et al., 2026, arXiv:2603.28052), respects every `CLAUDE.md §3` invariant, and lands on stropha's actual strengths (the adapter framework as a declarative search space).

It also **understates how much of Phase 1 is already done** and **overstates how mechanical Phases 2, 3, and 5 are**. Specifically:

| Finding | Effect on the plan |
|---|---|
| **`NDCG@K` and `Precision@K` already exist** in `eval/harness.py:47-92,182,189,220,227` with graded-relevance support. The RFC lists them as "added in Phase 1". | Phase 1 effort drops from M (3-4d) to **S (~1d)**: only the held-out split, the fitness composer, and the per-tag rendering gap (`cli.py:947-965` ignores NDCG/Precision in the by-tag table) remain. |
| **An unwired observability pathway already exists** end-to-end: `retrieval/savings.py:116-130` (producer) → `cost.py:211-217` (consumer). Three lines of wiring activate it. | Phase 0 can land a meaningful "trace stats" milestone in **hours**, not days, before the full RFC trace schema is built on top. |
| **A unified local-LLM helper already exists** (`stropha.inference.generate`, with MLX + Ollama backends — `inference/backend.py:34-201`). The RFC's "Claude Code via CLI or Anthropic API call" framing for Phase 4's agentic proposer ignores this. | The bandit-proposer MVP needs zero new dependencies (`numpy>=1.26` is already top-level); the agentic proposer can prototype against the local helper first and treat Claude-Code-via-subprocess as a strictly opt-in escalation. |
| **No combined "indexing-relevant pipeline hash" exists.** Five individual `adapter_id` values do (walker/chunker/enricher/embedder/storage); no aggregator. | Phase 2's index-cache key is **~10 lines** added next to `BuiltStages` in `pipeline/builder.py:32-46`. The RFC's S-M estimate stands but the "where it lives" answer is non-obvious. |
| **"Indexing-relevant vs query-time" knob distinction is implicit, not declarative.** No metadata flag on any Pydantic Config field encodes it. | Phase 2 and Phase 3 both depend on this. The RFC's "the manifest is generated from the adapter registry" claim (line 184) is **partially true**: registry knows (stage, name, Config class, field types, defaults, numeric ranges); registry does NOT know the indexing/query split, the enumerable choice-set of string fields like `embedder.model`, or cross-adapter constraints. Manual annotation is unavoidable. |
| **`load_pipeline_config` unconditionally reads `~/.stropha/config.yaml` and `./stropha.yaml`** (`pipeline/config.py:79-89`). No "isolated" mode. | Phase 2 per-candidate isolation must either (a) feed `build_stages(dict)` directly, bypassing `load_pipeline_config`, or (b) add an `ignore_user_yaml=True` flag. The cleaner path is (a), which already works. |
| **Copying an open WAL SQLite DB is unsafe.** Phase 2's "copy-on-write clone of the cached index" needs `sqlite3.Connection.backup()` exposed on `Storage`. | Tiny addition to `storage/sqlite.py`, but the RFC line at `meta-harness.md:170` reads as if `cp` will work. It will not on a live WAL file. |
| **`watch.py` loop fires only on mtime change** (`watch.py:157,282`). Phase 6 sleep-time wants periodic/idle-time iterations. | `WatchController` (`watch.py:212`) is not a drop-in for meta-sleep — a new loop body or a refactor that abstracts "what fires on a tick" is required. |
| **Zero precedent for LLM CLI subprocess or LLM SDK call** in the entire `src/` tree. The only declared `anthropic_api_key` (`config/models.py:131,293`) is never read. | Phase 4's agentic proposer is **100% greenfield surface area**: subprocess wrapper (or SDK), prompt assembly, response parsing, retry, credential fallback. The RFC's L estimate (8-12d) is plausible only if it stays mocked in tests (which the RFC explicitly states — `meta-harness.md:203` — so this risk is acknowledged). |
| **No `stropha.yaml` writer exists** anywhere in `src/`. | Phase 6 promotion's "emits a diff, never auto-applies" stance is consistent with current posture — no regression risk. But the diff renderer itself is missing (no `difflib` usage in `src/`); the Rich rendering layer is mature, but the diff *computation* is new. |
| **Hook template is a monolithic f-string with no plugin slots.** | Phase 0 outcome backfill and Phase 6 sleep-time both surgery the same template. Mitigated by the existing `HOOK_VERSION` bump pattern (`tools/hook_install.py:49`), but two phases editing the same string is a soft coupling worth calling out. |

**Verdict.** Phases 0-2 should ship in roughly the order the RFC describes, with **Phase 1 demoted to S effort** (because NDCG/Precision exist) and **Phase 0 split into a "wire `savings.py`" S-effort milestone before the full TraceRecorder**. Phases 3-4 are correctly scoped but the RFC overstates registry-based manifest generation — a hand-edited `search_space.yaml` will be unavoidable. Phase 5 (code-search) carries genuine "first-time in this codebase" risk because of sandbox/subprocess greenfield. Phase 6 is well-shaped; the only sharp edge is reusing `WatchController`.

---

## 1. How to read this document

The RFC has three layers of claims: the **architecture** (§3 of the RFC — modules, CLI commands, store layout), the **phase deliverables** (§4 of the RFC, seven phases), and the **invariant compatibility statement** (§1 of the RFC). This evaluation walks all three:

- §2 — phase-by-phase evaluation. For each phase: what the RFC promises, what code reality looks like *today*, the gap, risks/contradictions, and a calibrated effort note. Each phase ends with a **verdict** (`Accept`, `Accept with changes`, or `Re-scope`).
- §3 — invariant compatibility audit against `CLAUDE.md §3`.
- §4 — critical findings the RFC understates or contradicts.
- §5 — open questions the RFC should resolve before Phase 4 lands.
- §6 — recommended adjustments to the plan.
- §7 — bottom line.

Every `file:line` reference points to the current `HEAD` after the meta-harness merge.

---

## 2. Phase-by-phase evaluation

### Phase 0 — Artifact store + execution traces

**RFC claim:** `meta/{trace,store,outcome}.py`; wire `TraceRecorder` into `server.py` and `retrieval/search.py`; outcome back-fill step in the hook. Filesystem-only, no schema migration. Effort: L (~5-7d). Exit: every `search_code` call appends a trace; trace recording adds <2 ms p95.

**Current code reality:**

| Sub-feature | Status | Evidence |
|---|---|---|
| MCP-tool boundary logging on the happy path | **MISSING** for 9 of 12 tools. Only `mcp.startup`/`mcp.shutdown` (`server.py:91,120`) and two error-only events fire. `get_symbol`, `find_callers`, `find_related`, `get_community`, `find_rationale`, `find_tests_for`, `trace_feature`, `list_repos`, `get_file_outline`, `get_config_tool` log nothing on success. | `server.py:91,120,419,424,531` (exhaustive event inventory) |
| Per-stream contributions in retrieval results | **PARTIAL** — computed and discarded. `dense_hits`, `sparse_hits`, `symbol_hits` exist as locals in `_search_single` but are consolidated into `streams = [...]` and never reach the returned `SearchHit`. The RFC's `results[].streams` field (`meta-harness.md:127`) requires either preserving them past line 119 or returning a wrapper object. | `retrieval/search.py:108-119` |
| Latency capture | **MISSING** — zero `time.perf_counter()` calls in `retrieval/search.py`. | `retrieval/search.py:29-131` (grep) |
| `request_id` / `session_id` binding | **MISSING in callers, READY in infrastructure.** `structlog.contextvars.merge_contextvars` is in the processor chain (`logging.py:25`), so binding will flow through; no caller binds anything today. Zero `request_id` / `session_id` references in `src/`. | `logging.py:25`; grep |
| Per-search trace producer | **UNWIRED.** `retrieval/savings.py:116-130` (`track_search_savings`, `log_savings`) emits the `search.served` event with 5 fields (`total_searches`, `total_returned_tokens`, `total_baseline_tokens`, `total_tokens_saved`, `total_results_returned`). Project-wide grep confirms **zero call sites** outside the module. | `retrieval/savings.py:116-130` |
| Per-search trace consumer | **EXTANT.** `cost.py:211-217` already aggregates `search.served` into `SavingsCost`. The rendering branch at `cost.py:301-312` is dead code that lights up the moment a producer fires. | `cost.py:211-217,301-312` |
| `.stropha/` discovery pattern | **EXTANT** — `Config.resolve_index_path()` (`config/models.py:116,324`) centralises per-target-repo `.stropha/` rooting; a sibling `resolve_meta_path()` mirrors trivially. `tools/init_project.py:126-150` creates the dir. | `config/models.py:116,324`; `tools/init_project.py:126-150` |
| Hook template extensibility | **NO PLUGIN SLOTS.** The hook body is one monolithic `dedent` f-string at `tools/hook_install.py:172-339`. Steps are appended inside the existing `nohup sh -c "..."` heredoc (`hook_install.py:313-336`). The graphify step at `hook_install.py:320-330` is the only precedent for a guarded conditional step. | `tools/hook_install.py:172-339` |
| Hook foreground ≤100 ms before fork | **EXTANT and invariant-protected.** Pre-fork section ends at the `nohup ... &` at `hook_install.py:313`; everything before is variable assembly and skip-check `[ -f ... ] && exit 0`. A meta-sleep step inside the `nohup` body inherits the guarantee. | `hook_install.py:228-313`; `CLAUDE.md §3` invariant #5 |
| Schema migration needed? | **NO** — RFC correctly says filesystem-only. v6 (`files` table at `storage/sqlite.py:426-446`) is the latest indexing-relevant migration; v7 added `repo_id` to graph tables. No conflict. | `storage/sqlite.py:49` (`SCHEMA_VERSION=7`) |

**Gap.** All three RFC modules (`trace.py`, `store.py`, `outcome.py`) are genuinely new. The trace **producer** point is correctly identified (RFC says "wire into `server.py` and `retrieval/search.py`" — both are right; the cheapest one-line variant is a `@traced` decorator stacked under `@mcp.tool` plus an emit at the end of `_search_single`). The store layout matches the paper's filesystem design and aligns with the existing `.stropha/` convention.

**Risks / contradictions:**

1. **The RFC misses an obvious incremental milestone.** Wiring `track_search_savings(query, hits)` in `server.py:458` (just before `return SearchResponse(...)`) lights up the existing `cost.py` savings table immediately. This is a **3-line change** that proves the producer/consumer pattern before any RFC code is written. The full `TraceRecorder` schema is then a strict superset upgrade.
2. **Trace field `results[].streams` requires a small refactor** of `retrieval/search.py:108-119` because the per-stream lists are currently shadowed by `streams = [...]`. Trivial fix; flagged because the RFC's exit criteria (line 145) assume the field is populable, and a reviewer who doesn't read `_search_single` will miss this.
3. **Disk-growth concern (`meta-harness.md:262`) is real and underspecified.** The RFC says "JSONL with rotation" and "prune below a score percentile". Neither has a concrete trigger. A 200-chunk search-code call captured fully (results array with embeddings preview, per-stream ranks, snippets) is plausibly 5-10 KB per trace. At 100 calls/day per dev session that is 0.5-1 MB/day per repo. Acceptable, but the pruning rule deserves a concrete spec.
4. **`get_config_tool` (`server.py:809-831`) returns env-var registry data** that would be useful to capture in `meta.json` per candidate so the proposer sees the full environmental context, not just the resolved `stropha.yaml`. Worth adding to the Phase 0 trace schema.
5. **Best-effort fire-and-forget** (RFC line 140) needs a concrete implementation strategy. Python's stdlib offers no truly non-blocking JSONL appender; `concurrent.futures.ThreadPoolExecutor(max_workers=1)` or `queue.Queue` + daemon thread are the realistic options. The "<2 ms p95" exit criterion is achievable only if the append happens off-thread.

**Effort calibration:** the RFC's L (~5-7d) is the right magnitude. A defensible breakdown:
- 0.5d: wire `track_search_savings` (the cheap milestone above) and verify in `stropha cost`.
- 1d: `TraceRecorder` + JSONL append off a daemon thread + `meta/store.py` directory layout + `Config.resolve_meta_path()`.
- 1.5d: `@traced` decorator wired across all 12 MCP tools + light refactor of `_search_single` to preserve stream details.
- 1d: `meta/outcome.py` post-hoc back-fill + hook template surgery (bump `HOOK_VERSION` to 5).
- 1.5d: tests (25 unit tests as the RFC scopes) + dogfood pass to verify <2 ms p95.

**Verdict — Phase 0: Accept with one change.** Add a "wire `track_search_savings`" sub-milestone before the full `TraceRecorder` work. This gives Phase 0 a 4-hour demo point and de-risks the producer/consumer wiring before the schema gets ambitious.

---

### Phase 1 — Evaluator: fitness function + held-out split

**RFC claim:** Tag `tests/eval/golden.jsonl` with `split:`; add NDCG/Precision@K; build `meta/fitness.py` composite; build `meta/evaluator.py`; validation gate. Effort: M (~3-4d).

**Current code reality:**

| Sub-feature | Status | Evidence |
|---|---|---|
| Recall@K | **EXTANT** | `eval/harness.py:210-211` |
| MRR | **EXTANT** | `eval/harness.py:214-217` |
| **NDCG@K** (per-case + aggregate, graded-relevance aware) | **EXTANT — already implemented end-to-end** | `_dcg`/`_ndcg` helpers at `eval/harness.py:47-92`; `EvalResult.ndcg` at `:182`; `EvalReport.ndcg_at_k` at `:220` |
| **Precision@K** | **EXTANT — already implemented** | `EvalResult.precision` at `eval/harness.py:189`; `EvalReport.precision_at_k` at `:227` |
| Graded relevance in golden file | **SCHEMA EXTANT, UNUSED** — `relevance: [int, ...]` field supported at `harness.py:113-123`; zero rows in `tests/eval/golden.jsonl` or `tests/eval/mimoria_golden.jsonl` actually use it. Every relevance is currently binary. | `tests/eval/golden.jsonl`, `tests/eval/mimoria_golden.jsonl` |
| CLI rendering of NDCG/Precision | **GAP** — values are computed but `cli.py:947-965` only prints `recall@K` and `mrr` in the summary and by-tag tables, ignoring the other two metrics already on `EvalReport`. | `cli.py:947-965` |
| MAP (Mean Average Precision) | **MISSING** | grep, harness.py |
| Per-query latency | **MISSING** — no `time.perf_counter()` in `run_eval` | `eval/harness.py:304-368` |
| Split (`search` / `holdout`) field on `EvalCase` | **MISSING** — no `split` field on `EvalCase`, not consumed by `from_dict`, no rows tagged. `--tag` flag is the closest workaround. | `eval/harness.py:99-132,255-274`; `cli.py:868-871` |
| Evaluator callable from Python | **EXTANT** — `run_eval(cases, engine, top_k=...)` is pure Python and accepts duck-typed engines (real `SearchEngine` or the `built.retrieval` adapter via `cli.py:905`). | `eval/harness.py:304-368`; `cli.py:905` |
| Validation gate (pipeline-validate-equivalent) | **EXTANT** — `pipeline validate` at `cli.py:708-738` already probes adapter health. The pieces are there; `meta/evaluator.py` only needs to **call** them, not build them. | `cli.py:708-738` |
| In-memory pipeline build from a candidate config | **EXTANT** — `build_stages(resolved_config: dict, open_storage=True) -> BuiltStages` at `pipeline/builder.py:49-126` accepts any dict matching the canonical shape. | `pipeline/builder.py:49-126` |
| Fitness composer | **MISSING** — `EvalReport.summary()` returns raw metric dict; no `fitness()` method anywhere. | `eval/harness.py:241-249` |
| Mimoria sibling golden | **EXTANT** — `tests/eval/mimoria_golden.jsonl` has 43 rows with richer tag taxonomy (`symbol-lookup`, `conceptual`, `multi-hop`, `natural-language`, `compound`, `needs-graph`); not loaded by default CLI. | `tests/eval/mimoria_golden.jsonl` |

**Gap.** The RFC says: "NDCG / Precision@K added to `eval/harness.py` (option #106)." This is **factually incorrect** at HEAD — both metrics exist with graded-relevance support, per-case and aggregate, with passing tests in the eval-harness test file. The RFC's source for option #106 was `docs/comparisons/stropha-vs-meta-harness.md:277` which describes it as a roadmap item, but the work has already shipped (see commit `28122ab docs: update benchmark numbers with 43-query golden set` and earlier).

**Risks / contradictions:**

1. **Phase 1 is roughly half the effort the RFC budgets.** With NDCG/Precision done, the remaining work is: (a) add `split:` tagging + filtering — S; (b) build `meta/fitness.py` — S; (c) add per-query latency capture — S; (d) build `meta/evaluator.py` glue — S; (e) backfill the CLI by-tag rendering with NDCG/Precision — S; (f) tests — S. Realistic total: **1-1.5d**, not 3-4d.
2. **The composite fitness formula needs an empirical sanity check.** The RFC proposes `w₁·NDCG + w₂·Recall + w₃·context_efficiency − w₄·latency_penalty`. Without measured weights this is a free parameter that itself needs tuning (a meta-meta-problem). Recommend starting with `w₁ = w₂ = 0.5`, `w₃` = 0 until token counts are reliably captured, `w₄` = 0; iterate after first 100 candidates land.
3. **NDCG over binary relevance == Recall-shaped curve.** Until any golden row uses graded relevance, NDCG and Recall will correlate near-perfectly. The composite fitness is therefore not as decorrelated as it looks. Concretely: every entry in `tests/eval/golden.jsonl` and `mimoria_golden.jsonl` has only `expected_paths` + `expected_symbols`, no `relevance: {path1: 3, path2: 1}` map. The RFC's anti-overfitting story relies on this graded relevance materialising.
4. **`build_stages` with `open_storage=True` requires a real SQLite file on disk.** Phase 1's evaluator therefore cannot run a fully-in-memory eval; it has to point at an index file. That is fine for Phase 1 (one shared production index is enough); it becomes a constraint for Phase 2 (per-candidate index isolation) and for the agentic-proposer prompt construction (proposers need to know which index they're scored against).
5. **Hard-coded `0.85` Recall@K threshold** at `cli.py:930,968` is appropriate as a CI tripwire but should not gate Phase 1 evaluator calls. A `meta/evaluator.py` invocation must bypass this exit code (return the `EvalReport` regardless of pass/fail).

**Effort calibration:** **demote to S (~1-1.5d)**.

**Verdict — Phase 1: Re-scope.** The RFC's deliverables list overstates the metrics work and understates the fitness-composition design choices (weights, validation strategy). Replace the "add NDCG/Precision" line with "render NDCG/Precision per-tag in `cli.py`" and a paragraph on weight selection methodology.

---

### Phase 2 — Reproducibility: pinned snapshots + index cache

**RFC claim:** `meta/snapshot.py` — `stropha meta snapshot` pins a git worktree; index cache keyed by indexing-`adapter_id`s; per-candidate `STROPHA_INDEX_PATH` isolation. Effort: M (~3-4d).

**Current code reality:**

| Sub-feature | Status | Evidence |
|---|---|---|
| `git worktree` usage anywhere | **NONE** — only doc references at `meta-harness.md:97,168` and `vs-meta-harness.md:230`. The git_meta docstring at `ingest/git_meta.py:14` mentions worktrees but the code does not use them. | grep |
| Subprocess template for new git calls | **EXTANT** — `ingest/git_meta.py:165`, `tools/git_url.py:149`, `tools/hook_install.py:114`, `adapters/walker/git_diff.py:100,150` are all canonical `subprocess.run([...], capture_output=True, text=True, timeout=...)` patterns. | git-call inventory |
| Per-stage `adapter_id` values | **EXTANT** — 22 adapter classes, each implements `adapter_id` as a stable per-instance property. See the audit table in §B below. | `adapters/**/adapter_id` implementations |
| Combined "indexing-relevant pipeline hash" | **MISSING** — zero matches for `indexing_hash`, `pipeline_hash`, `combined_hash`, `pipeline_id`, `index_hash` in the codebase. The pipeline writes `active_enricher_id` and `active_embedder_id` to `meta` (`pipeline/pipeline.py:192-193`) but not walker, chunker, or storage. | grep |
| `Storage` accepts arbitrary paths | **EXTANT** — `Storage(db_path: Path, embedding_dim: int)` at `storage/sqlite.py:244`. Path resolution is in the adapter wrapper (`SqliteVecStorage.__init__` at `storage/sqlite_vec.py:39-54`). Two instances on different paths coexist in one process. | `storage/sqlite.py:244-266`; `storage/sqlite_vec.py:39-54` |
| `sqlite3.Connection.backup()` exposed | **MISSING** — `Storage` has no `backup()` / `clone_to()` method. Direct `cp` of a WAL DB is unsafe. | `storage/sqlite.py` (no `.backup()` reference) |
| `load_pipeline_config` "isolated" mode | **MISSING** — function unconditionally reads `~/.stropha/config.yaml` (`pipeline/config.py:79-82`) and `./stropha.yaml` (`:85-89`). Workaround = bypass `load_pipeline_config`, call `build_stages(dict)` directly (this already works). | `pipeline/config.py:75-117` |
| `BuiltStages` aggregates the right stages | **EXTANT** — `walker, chunker, enricher, embedder, storage, retrieval, resolved` (`pipeline/builder.py:32-46`). A new helper `BuiltStages.indexing_hash() -> str` lives here naturally. | `pipeline/builder.py:32-46` |
| `pyproject.toml` `numpy>=1.26` | **EXTANT** — top-level dep, already used in 3 source files. | `pyproject.toml:30` |

**Gap.** Three concrete missing pieces:

1. A combined indexing hash. Five `adapter_id` strings → `sha256(":".join(...))[:16]` is the obvious implementation; the right home is `pipeline/builder.py` next to `BuiltStages`. Should also exclude `storage.adapter_id` from the *cache key* (because `storage.adapter_id` only digests `dim`, `storage/sqlite_vec.py:67-68` — and `dim` is already determined by `embedder.adapter_id`, so including it is redundant).
2. `Storage.backup_to(target_path: Path)` calling `self._conn.backup(other_conn)` (stdlib `sqlite3.Connection.backup` is the safe way to copy a live WAL DB).
3. `meta/snapshot.py` driving `subprocess.run(["git", "worktree", "add", ...])` against the snapshot dir + `add --detach` so the worktree doesn't move during evaluation.

**Risks / contradictions:**

1. **The RFC's `cp`-implied copy is unsafe.** Line `meta-harness.md:170` says "a copy-on-write clone of the cached index". Filesystem COW (`cp --reflink` on btrfs/APFS) is fine; OS-level file copy of a live SQLite WAL is not. The implementation must use `sqlite3.Connection.backup` (online, safe, exposed-but-not-yet-wrapped) or close the source DB before copying. Worth tightening the RFC language.
2. **The cache key boundary is non-obvious for retrieval vs indexing.** The RFC handwaves "walker + chunker + enricher + embedder + storage" as indexing-relevant. This is correct but the *retrieval* `hybrid-rrf:k=60:streams=<hash>:reranker=<id>` adapter_id is NOT in the cache key, which is exactly right but contradicts the surface intuition that "more knobs differ → different cache". State the rule explicitly: **the cache key is the tuple that determines what is in `chunks`/`graph_nodes`**, not what is returned at query time.
3. **The `files` dirty cache (`storage/sqlite.py:426-446`, schema v6) is already an incremental-friendly substrate.** Phase 2 should explicitly note that an incremental re-index against a cached DB is what the inner loop actually runs, not a full rebuild every time. The RFC reads as if full rebuild is the default.
4. **Two-instance concurrent isolation works in principle but needs a stress test.** WAL mode is per-connection PRAGMA; the actual concurrent-write story across two `Storage()` instances pointing at *different* DB files is robust. Concurrent writes to the *same* file are also safe but `_check_dimension_consistency()` (`storage/sqlite.py:265`) will throw if dims differ. The Phase 2 plan should pin "each candidate gets its own DB file" as the rule, not "share one DB file".
5. **Git-worktree management has rough edges on stale references** (e.g., if a snapshot worktree is deleted out from under the meta-harness state). Recommend `git worktree prune` as a hook in `meta snapshot` cleanup.

**Effort calibration:** M (~3-4d) holds — the worktree subprocess wrapper and the cache-key + backup machinery are ~2d of code + 1-2d of tests.

**Verdict — Phase 2: Accept with three small clarifications** (combined-hash placement, `sqlite3.Connection.backup` for the copy step, and explicit "each candidate = own DB file" rule).

---

### Phase 3 — Search-space manifest

**RFC claim:** `meta/search_space.yaml` declarative spec; `meta/search_space.py` loader + `sample()` + `enumerate_dimensions()`. Manifest is generated *from the adapter registry* where possible. Effort: S-M (~2d).

**Current code reality:**

| Sub-feature | Status | Evidence |
|---|---|---|
| Adapter registry knows (stage, name) pairs | **EXTANT** — `pipeline/registry.py:19,64-76`; flat `dict[tuple[str, str], type[Any]]`. | `pipeline/registry.py:19,64-76` |
| Adapter registry knows Config class per adapter | **EXTANT** — `cls.Config` is required (`registry.py:30-34`); reachable as `_REGISTRY[(stage, name)].Config` or via `lookup_adapter(stage, name).Config`. | `pipeline/registry.py:30-34` |
| Adapter Config classes are Pydantic models | **EXTANT** — every one of 28 `Config = ...` assignments is a `BaseModel` subclass (verified). `.model_fields`, `.model_json_schema()`, `.model_dump()` all work. | grep `Config = ` |
| Pydantic field metadata (default, type, range, description) | **EXTANT** — every Config field uses `pydantic.Field(default=..., description=...)`; numeric ranges via `ge=/le=` on some fields (e.g. `local.py:34` — `batch_size: int = Field(default=32, ge=1, le=512)`). | adapters Config classes |
| Auto-import of `stropha.adapters` populates the registry | **EXTANT** — `adapters/__init__.py:_autoload()` uses `pkgutil.walk_packages` (recursive) at module import. Triggered from `builder.py:18`. | `adapters/__init__.py:26`; `builder.py:18` |
| "Indexing-relevant vs query-time" knob flag on Config fields | **MISSING** — no metadata. The split is implicit in which stage owns the field (walker/chunker/enricher/embedder/storage = indexing; retrieval = query-time). | (grep) |
| Categorical "valid choice set" for string fields (e.g. `embedder.model`) | **MISSING** — the embedder model field is a plain `str` (`embedder/local.py:30-33`). Valid HuggingFace IDs are unbounded; the proposer needs a hand-curated list. | `embedder/local.py:30-33` |
| Cross-adapter compatibility constraints (e.g. `embedder=voyage` ⇒ `VOYAGE_API_KEY`) | **MISSING** — encoded only in prose. | `embedder/voyage.py` |
| Per-stream `k` knob inspection | **PARTIAL** — `vec-cosine`, `fts5-bm25`, `like-tokens`, `graph-vec` each digest `k` into their `adapter_id` (e.g. `streams/vec_cosine.py:39` — `f"vec-cosine:k={k}"`). The actual config knob lives under `pipeline.retrieval.config.streams.<name>.config.k` per `HybridRrfConfig` — note the **double-nested `config` key**, which is unusual and worth verifying against the RFC's proposed manifest examples. | `adapters/retrieval/streams/*.py`; `adapters/retrieval/hybrid_rrf.py:148` |
| `stropha.yaml` shape | **DIVERGES FROM RFC EXAMPLES.** RFC line 182 example: `retrieval.streams.vec-cosine.k ∈ [10,100]`. Current YAML uses `pipeline.retrieval.config.streams.<name>.config.k`. The manifest YAML must use real paths. | `stropha.yaml`; `pipeline/config.py:_DEFAULTS` |

**Gap.** The "generated from the adapter registry" claim is **partially true**:

- ✓ Registry knows (stage, name) and Config class.
- ✓ Pydantic gives types, defaults, numeric ranges, descriptions.
- ✗ Registry does NOT know which knobs are indexing-relevant.
- ✗ Registry does NOT know finite-choice sets for unbounded `str` fields.
- ✗ Registry does NOT know cross-adapter constraints.

**Risks / contradictions:**

1. **The RFC line "it cannot drift from reality" (line 184) is aspirational, not factual.** Anything the manifest annotates beyond what Pydantic encodes (indexing/query split, categorical choices, cross-adapter constraints) IS drift-prone. Mitigation: a `meta/search_space.py` validator that fails if a manifest mentions a `(stage, name)` no longer in the registry, OR mentions a config field not in `model_fields`. The reverse direction (registry → manifest) is partly verifiable via a `meta/search_space.py` "audit" subcommand listing registry knobs missing from the manifest.
2. **The numeric-range coverage is inconsistent.** Some fields have `ge=/le=` (e.g. `local.py:34` `batch_size`), most do not. The manifest will need to supply ranges per knob; copying the existing `ge=/le=` is a starting point but not a complete one.
3. **`embedder.model` is the largest free-text knob.** Local: any HuggingFace ID. Voyage: any Voyage model name. The manifest must enumerate a known-good set per embedder family. A reasonable seed: `local: {mxbai-embed-large-v1, bge-small-en-v1.5, all-MiniLM-L6-v2}`; `voyage: {voyage-code-3, voyage-3-lite}`; `bge-m3: {BAAI/bge-m3}`; `mlx: {…known-working subset}`. Without this curation `sample()` will draw garbage strings.
4. **Latin-hypercube sampling** (RFC line 197) needs scipy (`scipy.stats.qmc.LatinHypercube`) which is NOT in `pyproject.toml`. The bandit can be perfectly fine with pure-numpy random sampling for the first ~100 candidates; LHS can be deferred to a Phase 4.1 or use a self-rolled implementation (it's ~30 lines of numpy).
5. **Where does the manifest live?** RFC says `meta/search_space.yaml`. Should it be `~/.stropha/search_space.yaml` (user-shared across projects) or `.stropha/meta/search_space.yaml` (per-project)? The §6 of the RFC implicitly puts it per-project. Make this explicit because cross-project reuse of a tuned manifest is a real ergonomic question.

**Effort calibration:** S-M (~2d) is **right for the loader and `sample()`**, but **understates the manifest's initial population effort**. Realistic plan: 1d code + 1-2d hand-curating the initial manifest content + 1d tests = 3-4d total. Recommend bumping to **M (~3d)**.

**Verdict — Phase 3: Accept with one tightening.** Replace "the manifest is generated from the adapter registry" with "the manifest is *validated against* the adapter registry; field metadata is partially generated; categorical choice sets and the indexing/query split are hand-annotated, with the validator failing on drift."

---

### Phase 4 — The outer loop (bandit + agentic proposer)

**RFC claim:** `meta/loop.py` Algorithm 1; `meta/proposer/bandit.py` (Thompson/UCB, local-only); `meta/proposer/agentic.py` (Claude Code via CLI or Anthropic API); CLI `stropha meta run`; held-out evaluation final step. Effort: L (~8-12d).

**Current code reality:**

| Sub-feature | Status | Evidence |
|---|---|---|
| Unified local LLM helper | **EXTANT** — `stropha.inference.generate(prompt, ..., backend=None)` at `inference/backend.py:173-201` with MLX + Ollama backends. HyDE already uses it (`retrieval/hyde.py:182-189`). | `inference/backend.py:34-201`; `retrieval/hyde.py:182-189` |
| `numpy` for bandit math | **EXTANT** — top-level dep at `pyproject.toml:30`, used in 3 source files. | `pyproject.toml:30` |
| Typer sub-app pattern | **EXTANT** — 4 examples (`pipeline`, `adapters`, `hook`, `glossary`) at `cli.py:644,741,975,1157`. | `cli.py:644,741,975,1157` |
| Subprocess wrapper for `claude` / `codex` / `gemini` | **NONE** — zero matches in `src/`. The 7 `subprocess.run` sites are all `git`-related. | grep |
| Anthropic / OpenAI SDK import | **NONE** — `anthropic_api_key` declared at `config/models.py:131,293` but never read. | grep |
| Background job machinery (for sleep-time wiring; carries forward to Phase 6) | **EXTANT** — `flock` + `nohup` + `disown` in hook template at `tools/hook_install.py:313-336`; `WatchController` threaded daemon at `watch.py:212-297`. | `tools/hook_install.py:313-336`; `watch.py:212-297` |
| `meta:` config section under `Config` | **MISSING** — no dataclass in `config/models.py`. Pattern (e.g. `InferenceConfig` at `config/models.py:72-79`) is well-established. | `config/models.py:72-79` |
| Tests for "mocked agent" | **MISSING** — no mock-LLM utility in `tests/` today. `tests/unit/test_mlx_enricher.py` and `test_ollama_enricher.py` use `unittest.mock.patch` on `urllib.request`/`mlx_lm.load` — that template can extend to the agentic-proposer tests. | `tests/unit/test_ollama_enricher.py` (mock pattern) |

**Gap.** The loop file (`meta/loop.py`), the bandit proposer (`meta/proposer/bandit.py`), the agentic proposer (`meta/proposer/agentic.py`), and the `MetaConfig` dataclass are all greenfield. The bandit can rely entirely on existing dependencies; the agentic proposer needs new wiring.

**Risks / contradictions:**

1. **The "Claude Code via CLI" path has zero precedent.** Acknowledged in the RFC's open questions (line 292) as a prototype decision. Mitigation: ship the bandit MVP first, then prototype the agentic proposer behind a feature flag. The RFC already commits to this in its MVP scoping (line 238).
2. **Local-only agentic proposer is feasible NOW.** The RFC frames the agentic proposer as needing a "strong coding model" (line 260) and treats it as opt-in. But `stropha.inference.generate()` with the `mlx-community/Qwen2.5-Coder-7B-Instruct` (or `32B`, depending on hardware) could carry a first-cut local agentic proposer. It will be weaker than Claude Code Opus, but it preserves the local-only invariant *and* avoids the subprocess greenfield. Worth adding as an interim option.
3. **The agentic proposer's reading model is grep + cat over `.stropha/meta/`** (RFC line 198, paper's design). A real Claude Code subprocess invocation cannot be sandboxed to read only that directory without explicit MCP permission scoping — which means the Claude Code instance has full repo read access. That's not a security issue (it already does, for indexing), but the prompt should explicitly direct it to `.stropha/meta/`, not the full repo.
4. **Bandit reward signal is the composite fitness from Phase 1.** If that fitness has weights = `0.5 NDCG + 0.5 Recall` and NDCG≈Recall on binary-relevance golden sets (Phase 1 risk #3), then the bandit is effectively optimizing one signal. The paper's "4× fewer context tokens" win comes from token-efficiency in the fitness; without it the bandit cannot find token-efficient configs. Phase 0's trace capture for `returned_tokens` is therefore a hard prerequisite, not a sequential follow-up.
5. **Crash mid-iteration consistency** (RFC test line 203). Implementation needs atomic writes: write `<id>/scores.json.tmp` then `os.replace(...)` to `<id>/scores.json`. Standard pattern; the RFC test will catch it but worth flagging.
6. **The bandit's per-dimension reward estimates** assume dimension independence — a known weakness of UCB / Thompson over high-dim discrete spaces. The RFC line 197 says "thereafter Thompson sampling / UCB over per-dimension reward estimates"; this is fine for the first 50-100 candidates, but contextual bandits (e.g. LinUCB) become attractive past that. Recommend adding a TODO for "swap to contextual bandit once iteration count exceeds 100".
7. **Wall-clock budget enforcement** (RFC line 196) is the right primitive. Add a per-candidate timeout too: if `evaluate_candidate(config)` takes more than N minutes, kill it and score it as `-inf` rather than blocking the loop. Otherwise a runaway chunker proposal can stall everything.

**Effort calibration:** L (~8-12d) is right *if* the agentic proposer ships as a CLI-subprocess prototype. If it ships against `inference.generate()` only, drop to **M-L (~6-8d)**. If it ships fully (CLI subprocess + Anthropic SDK + credentials fallback), bump to **L+ (~12-15d)**.

**Verdict — Phase 4: Accept with one addition.** Add a third proposer mode: `inference-helper` (uses local MLX/Ollama via the existing helper). This becomes the local-only middle path between the bandit (pure numeric) and the Claude-Code-CLI agentic (cloud, opt-in). It also de-risks Phase 4 because shipping the first agentic proposer requires zero new subprocess infrastructure.

---

### Phase 5 — Code search (v1)

**RFC claim:** `meta/proposer/code.py`; subprocess isolation per candidate; `pytest` gate on every proposed code candidate; namespaced candidate registration. Effort: L (~10-15d).

**Current code reality:**

| Sub-feature | Status | Evidence |
|---|---|---|
| Subprocess sandbox pattern | **NONE** — no precedent of a subprocess running a *Python module* with isolated `sys.path` and stdin/stdout pipes in `src/`. The 7 subprocess sites shell out to `git` or `which uv`. | grep |
| `@register_adapter` namespacing | **PARTIAL** — the registry's `_REGISTRY[(stage, name)] = cls` (`pipeline/registry.py:36-38`) refuses duplicate `(stage, name)`. There is no notion of "register under a candidate namespace" today. A clean implementation prefixes candidate adapter names (e.g. `meta-candidate-abc123.hierarchical-tweaked`) or maintains a separate registry per candidate. | `pipeline/registry.py:36-38` |
| `pytest` invocation from Python | **NOT IN `src/`** but the project's CI calls it; running `subprocess.run(["uv", "run", "pytest", "-q"], capture_output=True, timeout=...)` from a candidate evaluator is straightforward, following the `uv`-discovery pattern at `tools/hook_install.py:159`. | `tools/hook_install.py:159` |
| Production `adapters/` never modified | **CULTURAL** — invariant by `CLAUDE.md §3` #2 ("Never bypass the adapter framework"). Phase 5 must enforce this by isolating candidate adapters in a per-candidate directory. | `CLAUDE.md §3.2` |

**Gap.** Two genuinely hard pieces:

1. **Subprocess isolation that runs the full stropha eval against a candidate adapter** but with the candidate's `adapters/` directory prepended to `sys.path`. The candidate adapter inherits the candidate's `config.yaml`; its registration target is the same `(stage, name)` key as production — so the candidate must override an existing entry. Strategy: spawn `uv run python -m stropha.meta.eval_subprocess --candidate <id>` which calls `_reset_for_tests()` (`registry.py:79-81`) and then imports the candidate's `adapters/` before the production `adapters/`. The `_reset_for_tests()` helper is honest about what it is (test-only), so a properly-named production equivalent (`registry.reset()`) should be added.
2. **`pytest` gate** as a side-validator: run the production test suite (which exercises every registered adapter) *with the candidate adapter active* to ensure the candidate doesn't break anything else. This is broadly correct but expensive — a full `uv run pytest -q` takes ~16s per CLAUDE.md §2.6 baseline. At 100 candidates per loop run, that is 1600s = 27 minutes of pure pytest. Mitigation: run only the relevant `tests/unit/test_*<stage>*.py` per candidate, escalating to full suite only on the running-best candidate before promotion.

**Risks / contradictions:**

1. **The "production `adapters/` is never touched" guarantee is hard to enforce mechanically.** A misbehaving candidate could write to disk anywhere. Real isolation needs a chroot-like boundary (Docker, `os.chroot`) which is heavy-handed and explicitly out of scope. Realistic posture: trust the proposer (the agentic one runs Claude Code which is generally well-behaved) + add a `git status --porcelain` check post-candidate that fails the loop run if anything in `src/stropha/` changed. The RFC should call this out as a known-limited safety guarantee, not an absolute one.
2. **A faulty proposed adapter that raises on import** (RFC test line 217) is correctly anticipated — subprocess isolation catches this. But import-time side effects that don't raise but corrupt global state (e.g., a candidate that monkey-patches `register_adapter`) can poison subsequent runs *within the subprocess* before being killed. Mitigation: the subprocess is a fresh Python interpreter per candidate, no reuse.
3. **Phase 5's "introducing storage changes is out of scope for v1"** (RFC line 216) is the right line, but the RFC could be sharper about *what changes are in-scope*. Specifically: new chunker languages, new retrieval streams, new reranker heuristics, alternative enricher prompts — yes. New adapter stages, schema changes, new storage backends — no. List this explicitly in the Phase 5 deliverables.
4. **Schema-bump-deferral** (`v8 would only be needed if candidates introduce storage changes`) is correct but the comment is buried. Worth promoting to a `Phase 5 non-goals` paragraph.
5. **Per-candidate test selection** is itself a heuristic that risks missing breakage. Concretely: a candidate that tweaks the `local` embedder's tokenization could break `test_eval_harness.py` (which uses mocked hits) — but the test name doesn't mention "embedder". The safest path is to map adapters → tests via a hand-curated dict, falling back to full suite.
6. **The L (10-15d) estimate is plausible only if candidate code is small.** A candidate that proposes a 200-line new chunker still has to be reviewed by the proposer in the next iteration; the artifact-store readout per candidate balloons. The Phase 0 trace-pruning rule needs to handle code candidates differently (keep the source file, drop the trace bodies past N iterations).

**Effort calibration:** L (~10-15d) is plausible *with the simplifications above*. Without test-selection optimization or with full isolation requirements, balloon to L+ (~15-20d).

**Verdict — Phase 5: Re-scope.** Add an explicit "non-goals" paragraph (no new stages, no schema changes), document the limited-safety guarantee around `adapters/` modification, and add a per-candidate test-selection rule.

---

### Phase 6 — Promotion + sleep-time operation

**RFC claim:** `meta/promote.py` emits a `stropha.yaml` diff for human review (never auto-applies); sleep-time mode via the existing post-commit hook detached background; `stropha meta report` leaderboard + regression alert. Effort: M (~3-4d).

**Current code reality:**

| Sub-feature | Status | Evidence |
|---|---|---|
| `stropha.yaml` writer in `src/` | **NONE** — zero matches for `stropha.yaml.*write`, `with open(.../stropha.yaml.*'w'`, or any yaml.dump targeting `stropha.yaml`. Only YAML write site is `ingest/glossary.py:323` for the glossary file. | grep |
| Config-diff renderer | **NONE** — `pipeline show` (`cli.py:652-705`) renders state, not diffs. Zero `difflib` references in `src/`. Rich rendering layer is mature. | grep |
| Hook foreground ≤100 ms invariant | **EXTANT** — already protected (`tools/hook_install.py:228-313`). | `CLAUDE.md §3` invariant #5 |
| Detached `nohup` + `flock` in hook | **EXTANT** — `tools/hook_install.py:313-336`. | `tools/hook_install.py:313-336` |
| Hook version bump pattern | **EXTANT** — `HOOK_VERSION = 4` at `tools/hook_install.py:49`, `_BLOCK_RE.sub` replaces atomically. | `tools/hook_install.py:49,402-405` |
| `STROPHA_HOOK_*` env vars | **EXTANT** — 8 documented vars. Adding `STROPHA_HOOK_META=1` follows the existing template. | `tools/hook_install.py:206-219` |
| `WatchController` reusability | **PARTIAL** — `watch.py:212-297` provides start/stop/daemon-thread/event template, but the loop body at `watch.py:258-297` fires only on file mtime changes (`_snapshot()` + `_Snapshot.diff()`). Idle-time periodic triggering would need a new loop body. | `watch.py:212-297` |

**Gap.** Three pieces:

1. **Diff renderer.** stdlib `difflib.unified_diff` over `yaml.safe_dump(current)` vs `yaml.safe_dump(candidate)`. The Rich rendering layer (e.g. `rich.syntax.Syntax(diff, "diff", theme=...)`) gives a colored display essentially for free.
2. **Hook template surgery** to add a `stropha meta sleep --budget X` step inside the existing `nohup sh -c "..."` heredoc. Bump `HOOK_VERSION` to 5 or 6 (depending on Phase 0's bump). The env-var toggle `STROPHA_HOOK_META=1` should be off by default.
3. **`meta/promote.py`** — given a candidate id, produce a diff against the current `stropha.yaml`. Refuse if the candidate has held-out fitness below the current production config.

**Risks / contradictions:**

1. **Sleep-time iterations can stall a slow commit.** The `STROPHA_HOOK_TIMEOUT` default of 600s already caps the existing hook background; the meta-sleep step needs its own sub-budget (e.g., 60s of bandit iteration) so it doesn't consume the whole 600s and starve the index step. Add `STROPHA_HOOK_META_BUDGET` (default 60s).
2. **Hook template coupling.** Phase 0 also edits the template (outcome backfill). The two changes should land in the same `HOOK_VERSION` bump if they ship close together; if Phase 0 ships first (v=5) and Phase 6 later (v=6), upgrade discipline costs nothing because `_VERSION_RE` (`hook_install.py:61`) re-installs cleanly.
3. **Regression alert** (RFC line 227) needs an explicit threshold. "When the live `stropha.yaml` falls behind a discovered candidate by a margin on held-out" — what margin? Recommend a percentile-based rule: alert if any discovered candidate exceeds production by >2σ of the holdout-score noise floor (which Phase 1 needs to measure).
4. **Sleep-time concurrency with `watch`.** If `stropha watch` is running (e.g., MCP server background), it will trigger re-indexing every time the meta-sleep step writes traces. The two daemons must be aware of each other or one must defer to the other. Recommend: meta-sleep writes only to `.stropha/meta/` (which `.strophaignore` should exclude from indexing).
5. **Promotion is human-gated** — the RFC commits to "never auto-applies" (line 226), which is the correct posture and avoids any new "config writer" code path. Good.
6. **`stropha meta report` overlaps with `stropha cost`.** `cost.py` already aggregates per-adapter metrics. Phase 6 should extend `cost.py` rather than ship a parallel reporter — same logfile, same aggregation primitives.

**Effort calibration:** M (~3-4d) holds, splitting roughly 1d for diff renderer + promotion gate, 1d for hook surgery + env var + tests, 0.5-1d for the report leaderboard, 0.5d for the regression alert thresholding.

**Verdict — Phase 6: Accept with two clarifications** (sub-budget for the meta-sleep step inside `STROPHA_HOOK_TIMEOUT`; merge `stropha meta report` into `stropha cost` extension rather than parallel command).

---

## 3. Invariant compatibility audit (`CLAUDE.md §3`)

| # | Invariant | RFC compliance | Notes |
|---|---|---|---|
| 1 | **Drift detection** — every config change baked into adapter_id | ✓ | Phase 5 candidate adapters must implement `adapter_id` correctly. RFC line 35 already requires this. |
| 2 | **Forward-only schema migrations** | ✓ | Phases 0-6 add no schema. Phase 5 explicitly defers schema to a future v8. |
| 3 | **`Pipeline.run()` owns post-processing order** | ✓ | `meta/evaluator.py` calls `Pipeline.run()` (or equivalent) per candidate; does not reorder graph/FTS post-processing. |
| 4 | **Graph-gated MCP tools never return silent empty** | n/a | The RFC adds no new graph-gated tools. |
| 5 | **Hook ≤ 100 ms before fork** | ✓ | Both Phase 0 outcome backfill and Phase 6 sleep-time live inside the existing `nohup sh -c "..."` body (`tools/hook_install.py:313-336`). The pre-fork section is untouched. |
| 6 | **Local-only by default** | ✓ for MVP. Phase 4 agentic proposer is opt-in, gated like `VOYAGE_API_KEY`. Concerns: (a) the RFC could be sharper that the bandit MVP requires zero new deps; (b) recommended addition of an `inference-helper` proposer mode (see Phase 4 §2 above) strengthens this. |
| 7 | **`fts_chunks` is contentless-with-external-content** | n/a | The RFC does not touch FTS. |
| 8 | **Search-time toggles do not require re-index** | ✓ | Critical for Phase 2's cache key: HyDE / recursive merge / multi-query are NOT in the indexing-relevant hash. RFC line 169 implicitly respects this by listing "walker + chunker + enricher + embedder + storage" as cache-key inputs. Worth making explicit. |

All 8 invariants survive. The RFC is internally consistent with `CLAUDE.md §3`.

---

## 4. Critical findings the RFC understates or contradicts

### 4.1 Factual error: NDCG/Precision are already implemented

The RFC's Phase 1 deliverables list (line 156) reads:
> NDCG / Precision@K added to `eval/harness.py` (option #106).

This is **incorrect at HEAD**. Both metrics exist:
- `_dcg`/`_ndcg` helpers at `eval/harness.py:47-92`
- `EvalResult.ndcg` at `:182`, `EvalResult.precision` at `:189`
- `EvalReport.ndcg_at_k` at `:220`, `EvalReport.precision_at_k` at `:227`
- Graded-relevance support via `EvalCase.relevance_map` at `:113-123`

The actual remaining work is rendering them in the CLI table (`cli.py:947-965` ignores them in summary + by-tag), supporting graded relevance in the golden files (zero rows use it today), and adding MAP + per-query latency.

**Action:** Update RFC Phase 1 deliverable to "Add MAP + per-query latency; render NDCG/Precision per-tag in `cli.py`".

### 4.2 Phase 0 has a sub-day milestone the RFC misses

`retrieval/savings.py:116-130` is a fully-built per-search producer; `cost.py:211-217` is a fully-built consumer. **No producer call site exists** in the codebase (grep verified). Three lines in `server.py:458` activate the entire path:

```python
from .retrieval.savings import track_search_savings  # at top
...
track_search_savings(query, response.results)  # before return SearchResponse(...)
```

This satisfies a non-trivial subset of Phase 0's exit criterion (`stropha meta trace stats` reports call counts) via the existing `stropha cost` table. **Action:** Add as the first sub-milestone of Phase 0.

### 4.3 The unified inference helper is not acknowledged

The RFC's Phase 4 (line 198) frames the agentic proposer as "Claude Code via CLI, or an Anthropic API call". The codebase has a third option already: `stropha.inference.generate(prompt, ..., backend=None)` (`inference/backend.py:173-201`) with MLX + Ollama backends, already used by HyDE (`retrieval/hyde.py:182-189`).

**Implication:** A local-only agentic proposer (MLX/Ollama backed) is feasible WITHOUT touching the `subprocess` surface area. This keeps the local-only invariant intact even when the proposer is "agentic". **Action:** Add a third proposer mode (`inference-helper`) to Phase 4.

### 4.4 The "manifest is generated from the adapter registry" claim is partial

The RFC line 184 ("it cannot drift from reality") is aspirational. Registry knows:
- (stage, name) pairs
- Config class per adapter
- Per-field types, defaults, numeric ranges, descriptions (via Pydantic)

Registry does NOT know:
- Indexing-relevant vs query-time split
- Categorical choice sets for unbounded string fields
- Cross-adapter compatibility constraints

These are unavoidable manual annotations.

**Action:** Reframe as "validated against the registry; the validator fails on drift".

### 4.5 WAL DB copy needs `sqlite3.Connection.backup`

RFC line 170 ("copy-on-write clone of the cached index") is a one-line implementation pitfall: `cp` of a live WAL file is unsafe. `Storage` doesn't expose `backup()` today.

**Action:** Add `Storage.backup_to(target_path: Path)` to Phase 2 deliverables.

### 4.6 `load_pipeline_config` is unconditionally filesystem-touching

Per-candidate isolation must either bypass `load_pipeline_config` (calling `build_stages(dict)` directly — which works) or add an `ignore_user_yaml=True` flag. The cleaner path is the former.

**Action:** Document in Phase 2 that the evaluator constructs candidate configs as in-memory dicts and feeds them to `build_stages` directly, NOT through `load_pipeline_config`.

### 4.7 No combined indexing hash exists

The RFC speaks of "indexing-`adapter_id`s" (plural). The hash function combining them is missing; zero grep matches for `indexing_hash`/`pipeline_hash`. Adding it next to `BuiltStages` in `pipeline/builder.py:32-46` is the obvious home.

**Action:** Add a one-line method (`BuiltStages.indexing_hash() -> str` returning `sha256` over the five relevant `adapter_id`s).

### 4.8 `watch.py` is event-driven, not periodic

The RFC Phase 6 says sleep-time uses the existing detached hook background, which is correct for *post-commit* sleep. But the RFC also references "between sessions, the index reasons in background" (`vs-meta-harness.md:294`, option #116). That requires a periodic / idle daemon. `WatchController` (`watch.py:212-297`) fires only on file events.

**Action:** Clarify Phase 6 — sleep-time as it lands is *post-commit-triggered bandit iterations*, not a periodic background daemon. The daemon variant is a Phase 7+ topic.

### 4.9 Zero precedent for LLM CLI subprocess

The RFC's agentic-proposer plan assumes Claude Code (or another agent CLI) is invocable from `subprocess.run`. The codebase has zero such precedent — the seven subprocess sites are all `git`. The `anthropic_api_key` declared at `config/models.py:131,293` is unused.

**Action:** Acknowledge that Phase 4's agentic proposer is **100% greenfield surface area** (subprocess wrapper or SDK, prompt construction, retry, credentials fallback) — and that prototyping against `inference.generate()` is a strictly cheaper first step (see §4.3).

### 4.10 Test count drift

`CLAUDE.md §2.6` says 806 tests; `CLAUDE.md §6` (history bottom) says "Total suite: 383 tests, all green". The merge introduces no new tests but the inherited inconsistency is worth resolving before Phase 0 lands its 25 new tests on top.

**Action:** Reconcile the snapshot test count (pre-existing issue, unrelated to the RFC, but worth fixing).

### 4.11 Hook template has no plugin slots

Both Phase 0 (outcome backfill) and Phase 6 (sleep-time) edit the same `_render_hook_block` f-string at `tools/hook_install.py:172-339`. No formal extension point exists. The `HOOK_VERSION` bump pattern lets them coexist, but two phases touching the same template is a soft coupling.

**Action:** Consider a small refactor of `_render_hook_block` to accept an `extra_steps: list[str]` argument that gets joined into the `nohup sh -c "..."` body. Phase 0 contributes "outcome backfill"; Phase 6 contributes "meta sleep step". Same string, more orderly.

### 4.12 Disk growth / pruning rule needs a concrete threshold

RFC line 262 says "prune candidates below a score percentile after N iterations, keeping the best + worst". Concrete numbers absent. Without a rule the artifact store grows monotonically.

**Action:** Pin "keep top-10 + bottom-10 + last-50, prune the middle, monthly compaction" or similar.

---

## 5. Open questions the RFC should resolve before Phase 4 lands

The RFC's own §10 (line 289) lists three open questions. Augmenting with the audit:

1. **Fitness ground truth** (RFC §10.1). Proxy-metric overfitting is real because NDCG≈Recall on binary-relevance golden sets. A retrieval-conditioned end-to-end task eval (`vs-meta-harness.md:281`, option #110) needs an LLM judge or a programmatic check — neither of which is local. Decide whether the held-out composite is enough for v1 OR whether `inference.generate()` can do the judging locally (model quality cap is real).
2. **Agentic proposer host** (RFC §10.2). The audit clarifies: prototype against `inference.generate()` *first*, then escalate to Claude Code CLI in v2 if needed. The Anthropic API path is a niche fallback.
3. **Promotion autonomy** (RFC §10.3). The current "local diff only" stance is correct; do not revisit until Phase 6 lands.
4. **NEW: Where does `search_space.yaml` live?** Per-project (`.stropha/meta/search_space.yaml`) or shared (`~/.stropha/search_space.yaml`)? Cross-project reuse is a real ergonomic question.
5. **NEW: How are stale snapshots cleaned up?** `git worktree prune` discipline + `.stropha/meta/snapshot/` TTL.
6. **NEW: How is fitness weight tuning bootstrapped?** Hand-set `(w₁, w₂, w₃, w₄)` for the first 100 candidates; revisit after.
7. **NEW: How are concurrent `stropha watch` + meta-sleep daemons coordinated?** `.strophaignore` excludes `.stropha/meta/` from indexing; meta-sleep writes only there. Document this.

---

## 6. Recommended adjustments to the plan

| Adjustment | Rationale | Effort cost |
|---|---|---|
| **Phase 0**: split into "wire `track_search_savings`" S milestone + full `TraceRecorder` work | De-risks producer/consumer pattern, gives a 4-hour demo point | None — addition only |
| **Phase 0**: refactor `_search_single` to preserve per-stream lists past `streams = [...]` | RFC's `results[].streams` trace field needs this | +0.2d |
| **Phase 0**: `Config.resolve_meta_path()` mirrors `resolve_index_path()` | Per-target-repo `.stropha/meta/` discovery | +0.2d |
| **Phase 1**: demote from M (3-4d) to S (~1-1.5d); replace "add NDCG/Precision" with "render NDCG/Precision per-tag + add MAP + latency" | NDCG/Precision already exist | -2d |
| **Phase 1**: document weight-tuning bootstrap | Avoids weights becoming a free parameter | None — documentation only |
| **Phase 2**: add `Storage.backup_to(target_path)` using `sqlite3.Connection.backup` | Safe WAL DB clone | +0.3d |
| **Phase 2**: explicit "indexing-relevant `adapter_id`s = walker + chunker + enricher + embedder; storage and retrieval excluded" rule | Cache-key correctness | None — documentation only |
| **Phase 2**: explicit "candidates use `build_stages(dict)`, bypassing `load_pipeline_config`" | Per-candidate isolation without filesystem reads | None — documentation only |
| **Phase 2**: add `BuiltStages.indexing_hash()` | Missing aggregator | +0.2d |
| **Phase 3**: bump from S-M (~2d) to M (~3d) | Manifest content curation effort underbudgeted | +1d |
| **Phase 3**: replace "manifest generated from registry" with "manifest validated against registry; field metadata partly generated; choice sets + indexing/query split hand-annotated" | Factual accuracy | None — documentation only |
| **Phase 4**: add `inference-helper` as a third proposer mode (alongside `bandit` and `agentic`) | Local-only agentic proposer feasible NOW via existing `inference.generate()` | Net cost ambiguous — replaces "subprocess wiring" with "prompt design"; likely neutral or -2d |
| **Phase 4**: per-candidate `evaluate_candidate(...)` timeout in addition to wall-clock budget | Stall protection | +0.1d |
| **Phase 4**: atomic `scores.json.tmp` → `scores.json` write | Crash consistency | +0.1d |
| **Phase 5**: explicit non-goals paragraph (no new stages, no schema changes) | Scope discipline | None |
| **Phase 5**: limited-safety guarantee around `adapters/` modification, with `git status --porcelain` post-candidate check | Truth in advertising | +0.2d |
| **Phase 5**: per-candidate test-selection map | Cuts pytest gate cost by ~10× | +0.5d |
| **Phase 6**: `STROPHA_HOOK_META_BUDGET` sub-budget inside `STROPHA_HOOK_TIMEOUT` | Sleep-time can't starve index step | +0.1d |
| **Phase 6**: clarify "post-commit-triggered" vs "idle-time daemon" | The latter is Phase 7+, not Phase 6 | None |
| **Phase 6**: extend `stropha cost` rather than a parallel `stropha meta report` | Reuse aggregation primitives | -0.5d |
| **Cross-phase**: refactor `_render_hook_block` to accept `extra_steps: list[str]` | Phase 0 + Phase 6 both modify the hook template; one signature avoids merge churn | +0.3d (paid once, twice in returns) |

Total net change to the original RFC budget: **roughly –1 to –2 days** (Phase 1 savings exceed Phase 2/3/4/6 additions).

---

## 7. Bottom line

The RFC describes a system that can be built on stropha as-is without breaking any invariant, and the order of phases is correct. The strongest pieces of the plan are the artifact-store layout (matches the paper's filesystem-first design), the bandit-MVP scoping (genuinely local, zero new deps), the index-cache idea (correctly identifies the dominant efficiency lever), and the human-gated promotion stance.

The weakest pieces are the **factual error about NDCG/Precision being unimplemented** (Phase 1), the **understated greenfield risk of the agentic proposer** (Phase 4), the **overstated registry-driven manifest generation** (Phase 3), and a handful of small mechanical pitfalls (WAL copy, hook template coupling, watch.py reuse).

None of these are blocking. With the §6 adjustments — most of which are documentation tightenings rather than scope changes — the plan ships. The MVP timeline (3-4 weeks for Phases 0-4 bandit) is realistic; the full plan timeline (7-9 weeks) is realistic *if* the agentic proposer ships as the `inference-helper` variant first and Claude-Code-via-subprocess is treated as v2.

**Recommended next step:** if the team wants to start, ship the "wire `track_search_savings`" sub-milestone of §4.2 immediately (3 lines of code, lights up an existing dead-rendering path in `stropha cost`, costs 30 minutes, demonstrates the producer/consumer wiring) — then commit to Phase 0 in full, then re-evaluate the RFC text against this audit before Phase 1.
