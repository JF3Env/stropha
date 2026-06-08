# Persisting Problems Report — Stropha tooling + mimoria Stripe model

> **Date:** 2026-06-07
> **Author:** investigation session (OpenCode)
> **Jira (system of record):** stropha tooling → epic **WIN-421** (WIN-422 / WIN-423 / WIN-424); mimoria → **MIM-33** (Fase 1 hardening) + epic **MIM-34** (Fase 2 conceptual fix). This file is now a root-cause reference; status lives in Jira.
> **Scope:** Problems that still persist after the Stropha engine change
> `46ec16b → 5dab77c`, discovered while benchmarking the tenant↔Stripe
> investigation against the mimoria codebase.
> **Repos involved (one shared index):**
> - `github.com/jonatasfischer/stropha` — served at `/Users/jonatas/sources/rag` (6,908 chunks)
> - `github.com/jonatasfischer/mimoria` — `/Users/jonatas/sources/mimoria` (5,500 chunks), index head `2468903c`

---

## Executive summary

| # | Problem | Jira | Where | Severity | Effort |
|---|---------|------|-------|----------|--------|
| 1 | `trace_id` exposed on only 2 of ~40 tool responses | WIN-422 | stropha | High (breaks feedback loop) | ~10 min |
| 2 | All project memory bound to `target_repo`, not the repo being reasoned about | WIN-423 | stropha | High (memory unusable in multi-repo) | ~half day |
| 3 | `assert_snippet_freshness` compares against the outline card, not source → false drift | WIN-424 | stropha | Medium (erodes agent trust) | ~1–2 h |
| 4 | Tenant wired to Stripe only as merchant (payee), never as customer (payer) | MIM-33 + MIM-34 | mimoria | Medium (product/architecture) | feature |

**Structural root of #1–#2:** the MCP serves **one shared index over two repos**, but
several subsystems were written assuming a single repo (`app.config.target_repo`).

---

## Problem 1 — `trace_id` is exposed on only 2 of ~40 tools

> **Jira:** WIN-422 (epic WIN-421)

### Symptom
`search_code` and `smart_search` now return a top-level `trace_id` (the
`5dab77c` fix). But `execute_investigation` / `assemble_context` — the tools
most likely to return a weak bundle — return **no** `trace_id`, so
`report_bad_result` cannot be called for them. The feedback loop the
Meta-Harness depends on is dead for everything except the two search tools.

### Root cause — stamping is opt-in per response model, almost nothing opted in
The `@traced` decorator already mints an id for **every** tool and writes it to
the JSONL trace:
- `src/stropha/meta/decorator.py:191` — `trace_id = uuid.uuid4().hex`
- `src/stropha/meta/decorator.py:238` — `_attach_trace_id(result, trace_id)`

But `_attach_trace_id` only stamps the response if the model **declares** the field:
- `src/stropha/meta/decorator.py:40-58` — sets it only `if fields is not None and "trace_id" in fields`; otherwise silently skips (best-effort, swallows).

Only two models declare it:
- `src/stropha/server_models.py:93` — `SearchResponse.trace_id`
- `src/stropha/server_models.py:194` — `SmartSearchResponse.trace_id`

`AssembleContextResponse` has none — `src/stropha/server.py:1365-1375` (declares
`task`, `bundle`, `sections`, … but not `trace_id`). It is returned by both:
- `src/stropha/server.py:1417` — `assemble_context`
- `src/stropha/server.py:1526` — `execute_investigation`

Same gap for `FreshnessResponse`, memory responses, etc.

### Fix
Add `trace_id: str | None = None` to `AssembleContextResponse` (and any other
response you want reportable). The decorator auto-stamps — no per-tool change.
For full coverage: put `trace_id` on a shared base model and inherit, **or**
relax `_attach_trace_id` to attach to any `BaseModel` (e.g. via `model_config`
extra) instead of requiring a pre-declared field.

---

## Problem 2 — All project memory is bound to `target_repo`, not the repo being reasoned about

> **Jira:** WIN-423 (epic WIN-421)

### Symptom
Mimoria findings (`remember` / `record_episode`) were filed under the
**stropha** namespace. `check_staleness` on a mimoria memory compared against
**stropha's** commits (`46ec16b → 5dab77c`) and stropha's changed files
(`src/stropha/*.py`), returning a meaningless "stale=false, none touch the anchor".

### Root cause — one resolver hard-wired to the server's configured repo
- `src/stropha/server_memory_tools.py:32-42` — `resolve_target_repo_id(app)`
  resolves the repo from `app.config.target_repo` only; there is no per-call
  repo argument.
- Called by every memory impl: `remember_impl` (`:101`), `recall_impl` (`:131`),
  `forget_impl` (`:155`), `record_episode_impl` (`:320`), `recall_episodes_impl`
  (`:349`), `co_occurrence_impl` (`:374`), `cache_stats/prune` (`:169/:190`),
  `check_staleness_impl` (`:253`).

Three downstream bugs fall out:

1. **Write anchor wrong.** `remember` / `record_episode` always store under
   `target_repo`'s `repo_id`. In a multi-repo index, mimoria knowledge lands in
   the stropha bucket.
2. **Provenance commit wrong.** `src/stropha/server_memory_tools.py:45-50` —
   `current_head_commit(app)` reads `app.config.target_repo`'s HEAD, so a
   mimoria memory got stamped with a stropha commit (`5dab77c`).
3. **Staleness diff wrong.** `src/stropha/server_memory_tools.py:267-271` —
   `check_staleness_impl` calls
   `staleness_signal(app.config.target_repo, since_commit=entry.provenance_commit, anchor_files=...)`.
   It runs `git diff` in the **stropha** tree, from a **stropha** commit, and
   intersects **mimoria** `anchor_files` against stropha's changed files — they
   can never intersect, so it structurally cannot detect mimoria drift.

### Fix
- Add an explicit `repo` parameter to the memory tools (normalized key or local
  path), defaulting to `target_repo` for back-compat; resolve `repo_id` from it.
- For staleness, resolve the memory entry's **own** repo. The `repos` table
  already stores `local_path` per repo (visible as `repo.local_path` in every
  search result). `check_staleness_impl` already loads the row
  (`src/stropha/server_memory_tools.py:256`) — join to its `repo_id`, fetch that
  repo's `local_path`, and run `staleness_signal` against **that** tree + HEAD,
  not `app.config.target_repo`.

---

## Problem 3 — `assert_snippet_freshness` compares against the outline card, not source

> **Jira:** WIN-424 (epic WIN-421)

### Symptom
Reported `fresh=false` ("drifted") for `TenantSettingsEntity` even though
`git diff` proves the file is byte-identical to the indexed commit. The
`current_snippet` returned was the **outline**
(`class TenantSettingsEntity / Members: hasAnthropicKey, …`), not source.

### Root cause — compares cached text against whatever single representation storage returns as `.snippet`; for container chunks that is a structural card
- `src/stropha/retrieval/freshness.py:127` —
  `current = str(getattr(hit, "snippet", "") or "")`, where `hit = lookup_fn(chunk_id)`.
- Wired `lookup_fn` is `Storage.lookup_by_chunk_id`
  (tool `src/stropha/server.py:2257-2271` → `assert_snippet_freshness_impl`).
  Confirmed in all backends, e.g. `PostgresStorage.lookup_by_chunk_id`
  (`src/stropha/adapters/storage/postgres.py:1085`) → `self._hit(row, …)`.
- For a `kind=class` chunk, that hit's `.snippet` is the **outline** (empirically:
  the tool's own `current_snippet` was the `class … / Members: …` text). A
  class/file chunk's "snippet" is never the literal source body.

Compounding factors:
- A single `chunk_id` renders at multiple `card_level`s — `expand_chunk` takes
  `line` / `outline` / `full` (`src/stropha/server.py:~603`). But
  `lookup_by_chunk_id` returns one fixed rendering, so freshness has no way to
  compare like-for-like against the level the agent cached.
- Comparison is line-equality after whitespace-normalisation
  (`src/stropha/retrieval/freshness.py:50-58, 129-132`). A cached **method body**
  vs a class **outline** differs on line 1 → instant false drift with zero real change.

### Fix (increasing fidelity)
- **Minimum:** when `hit.kind` is a container (`class` / `file` / `section`),
  return `fresh=None` with a reason ("freshness only verifies leaf chunks")
  instead of a false `False`.
- **Better:** compare against the **full source slice** — the hit carries `path`,
  `start_line`, `end_line` and the repo `local_path`; read those lines from disk
  (or store the raw body) and compare against that, not the card.
- **Best:** default to **containment** (`fresh=True` if normalised
  `expected_snippet` is a substring of the chunk's full source) and accept an
  optional `card_level` so strict callers compare exact renderings.

---

## Problem 4 — Tenant↔Stripe conceptual error still persists (mimoria)

> **Jira:** MIM-33 (Fase 1 hardening) + epic MIM-34 (Fase 2 conceptual fix)

Unchanged in code. `git diff` on the four Stripe files is empty and the working
tree still has **no** `stripeCustomerId` / `MimoriaPlanSubscriptionEntity` field
usages (only doc-comments saying "forthcoming").

- Tenant is wired to Stripe **only** as a Connect Standard merchant (payee):
  `TenantSettingsEntity.stripeAccountId`
  (`backend/src/main/java/com/mimoria/infrastructure/persistence/entity/TenantSettingsEntity.java:101`).
- It is never modeled as a paying **customer** (`cus_xxx` + subscription) — the
  inverse relationship the SaaS business needs.
- `StripeCheckoutService.createSession` sets **no `application_fee_amount`**
  (`backend/src/main/java/com/mimoria/service/StripeCheckoutService.java:145-195`)
  → the platform earns nothing.
- `getStripeConnected()` (`TenantSettingsEntity.java:195`) means "can charge
  students," not "pays Mimoria" — misleading semantics.

This is a product/architecture task: implement the Fase 2 customer side per
`docs/architecture/B2B_SAAS_PLATFORM.md` (the design already separates it into
`MimoriaPlanSubscriptionEntity.stripe_customer_id`). Not a Stropha bug.

---

## Appendix — Benchmark that motivated this report

Clean A/B: mimoria code index held constant at `2468903c` (verified — `git diff`
on the 4 Stripe files empty); only the Stropha engine moved
(`46ec16b → 5dab77c`: `a72e948` dedup parent-promoted hits #4, `5dab77c`
surface trace_id #5).

Identical query `"TenantSettings entity stripe account id charges enabled fields columns"`:

| Metric | Before | After |
|--------|--------|-------|
| Results returned | 5 | 2 |
| `TenantResource` chunk `180b1397` | duplicated at ranks 1, 2, 3, 6 | once (rank 1) |
| `total_candidates` | 5 | 2 |
| `trace_id` in response | absent | present (`34add13c`) |

`execute_investigation`: byte-identical (5 spans, 330 tokens), still **no**
`trace_id` (Problem 1).

Index/config snapshot: shared index 12,408 chunks (stropha 6,908 + mimoria 5,500);
embedder `voyage-4-large@512`; reranker off; graphify graph not loaded
(`graph_traversal_available: false`).

---

## Suggested order

1. **Problem 1** (~10 min) — one field on `AssembleContextResponse`; unblocks the
   feedback loop for the other ~38 tools.
2. **Problem 3** (~1–2 h) — self-contained in `freshness.py` +
   `assert_snippet_freshness_impl`; high agent-trust impact.
3. **Problem 2** (~half day) — API param + per-repo staleness resolution; makes a
   multi-repo index usable for memory.
4. **Problem 4** — separate mimoria feature track (Fase 2 B2B SaaS).
