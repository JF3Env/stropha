# stropha MCP tool layer — structural audit

**Status:** Proposed (analysis only; no code changes shipped with this doc)
**Date:** 2026-05-25
**Scope:** `src/stropha/server.py` (~4500 LOC, 48 `@mcp.tool` registrations)
**Audience:** maintainers considering a tool-layer refactor

---

## 1. Why this document exists

Every change to a stropha tool currently has to absorb the friction of an
inconsistent layer. The 48 MCP tools were grown one ticket at a time and
the shapes have drifted. Some tools return raw `dict`, some return typed
Pydantic models. Some gate on the graph mirror via a shared helper,
others hand-roll the same gate three different ways. Some catch
`StrophaError`, most rely on `@traced` to surface it.

None of these divergences is wrong in isolation, but together they cost
maintainers time on every read and every diff: "is this the right
no-repo signal for this tool?", "does this graph tool return Pydantic
or `dict`?", "where is the canonical `_graph_unavailable` message?"

This audit catalogues what's actually in the codebase, groups it by
implementation pattern, and lists the divergences that are worth
unifying — and the ones that should be left alone.

Every claim is line-cited against the HEAD of `main` at write time.

---

## 2. Methodology

1. Enumerated every `@mcp.tool` decoration in `server.py`
   (`grep -c "^@mcp.tool" src/stropha/server.py` → **48**).
2. Read each tool body and categorised by:
   - Where the real logic lives (inline vs delegated to a helper module).
   - How it acquires `AppContext` (always via `_ctx(ctx)`, `server.py:494`).
   - How it talks to storage (typed methods vs raw `_conn.execute`).
   - Whether it gates on `graph_loaded(app.storage)`.
   - Error handling: `@traced` only, or catches & rewrites.
   - Return type: typed Pydantic vs raw `dict`.
3. Verified specific findings (counts, helper names, line numbers, the
   audit_citations bug) by direct file reads.

The current `CLAUDE.md` does not give an exact tool count — earlier
snapshots in the project's history referenced smaller counts (the
codebase has grown a lot), so any documentation that names a number
should be verified against `grep -c "^@mcp.tool" src/stropha/server.py`
before being trusted.

---

## 3. Tool inventory

48 tools, grouped by implementation pattern (see §4 for what each
pattern means).

| Pattern | Count | Tools |
|---|---:|---|
| Pure pass-through | 4 | `get_symbol`, `list_repos`, `get_file_outline`, `list_task_recipes` |
| Glue + delegate | 16 | `list_pins`, `smart_search`, `expand_chunk`, `plan_investigation`, `cache_stats`, `cache_prune`, `bootstrap_environment`, `assemble_context`, `execute_investigation`, `apply_task_recipe`, `assert_snippet_freshness`, `working_set`, `audit_citations`, `spec_consistency`, `check_symbols`, `expand_chunks` |
| Graph-gated, raw-dict return | 6 | `find_callers`, `find_related`, `get_community`, `find_rationale`, `find_tests_for`, `trace_feature` |
| Graph-gated, typed return | 4 | `blast_radius`, `test_impact`, `contract_tests`, `verify_claim` |
| Session mutator | 7 | `reset_dedup`, `pin_chunks`, `unpin_chunks`, `scratchpad_read`, `scratchpad_write`, `scratchpad_append`, `scratchpad_clear` |
| Memory writer | 3 | `remember`, `forget`, `record_episode` |
| Memory reader | 4 | `recall`, `recall_episodes`, `co_occurrence`, `check_staleness` |
| Inline-heavy orchestrator | 3 | `search_code`, `get_config`, `suggest_next_tool` |
| Inline + try/except (unique) | 1 | `report_bad_result` |

Total: 4 + 16 + 6 + 4 + 7 + 3 + 4 + 3 + 1 = 48. `smart_search` exhibits
the inline + try/except shape too but is counted under "Glue + delegate"
to avoid double-counting; §5.7 covers the exception-handling
divergences across all three tools that mask exceptions.

---

## 4. Pattern catalogue

### 4.1 Pure pass-through (4 tools)

Two- or three-line body. Resolve `AppContext`, ask storage or a
registry, map rows into a Pydantic response.

Example — `get_file_outline` (`server.py:1236`):
```python
@traced("get_file_outline")
def get_file_outline(path: str, *, ctx: Context) -> list[OutlineEntry]:
    app = _ctx(ctx)
    return [OutlineEntry(**row) for row in app.storage.file_outline(path)]
```

These are healthy. They demonstrate the platonic ideal of a stropha
tool: a thin protocol adapter over a typed storage / retrieval API.

### 4.2 Glue + delegate (16 tools)

The dominant shape. Body resolves AppContext, lazy-imports a function
from `retrieval/*`, calls it, maps the dataclass result into a
Pydantic response.

The lazy import is intentional: it keeps `server.py` cold-start fast
and avoids import cycles. The Pydantic mapping is unavoidable because
MCP tool outputs must serialise.

### 4.3 Graph-gated, raw-dict family (6 tools)

`server.py:2247-2401`. All six share the same five-line skeleton:

```python
app = _ctx(ctx)
if not graph_loaded(app.storage):
    return _graph_unavailable() | {"symbol": symbol}
return graph_<operation>(app.storage, symbol, ...)
```

Returns a raw `dict`. No Pydantic schema. This is the *original* graph
tool shape — `find_callers`, `find_related`, `get_community`,
`find_rationale`, `find_tests_for`, `trace_feature`.

### 4.4 Graph-gated, typed family (4 tools)

Newer graph tools (`blast_radius_tool`, `test_impact_tool`,
`contract_tests_tool`, `verify_claim_tool`). Same conceptual gate,
but:

1. They **don't use `_graph_unavailable`**.
2. They construct a typed Pydantic response with `graph_loaded=False`.
3. The "graph is missing" message is hand-rolled per tool (see §5.1).

### 4.5 Session mutator (7 tools)

`reset_dedup`, `pin_chunks`, `unpin_chunks`, scratchpad family.
Mutate fields on `AppContext` (`seen_chunk_ids`, `pinned_chunk_ids`,
`scratchpad`) and call `_persist_session_state(app)` (`server.py:511`).

Body shapes vary because the mutations differ (sets vs strings). Each
tool body is already 5–15 lines. A "generic mutator" abstraction would
be premature — there are no obvious shared invariants beyond the
persist call.

### 4.6 Memory writers (3) / readers (4)

Both kinds resolve a target repo via `_resolve_target_repo_id(app)`
(`server.py:1399`). Writers: `remember`, `forget`, `record_episode`.
Readers: `recall`, `recall_episodes`, `co_occurrence`, `check_staleness`.

This is where the asymmetry findings of §5 are concentrated.

### 4.7 Inline-heavy orchestrator (3 tools)

`search_code` (~300 lines, `server.py:623`) is the obvious outlier. It
orchestrates filters, dedup, pins, packing, compression, decay, savings
labelling, conversation-context resolution, hits-to-results, and
response packing. Every block has a load-bearing comment tying it to
a ticket.

`get_config_tool` (`server.py:2969`) builds 6 snapshots from
`AppContext` and the runtime config. It's inherently long because
the response surface is wide; there's no extractable kernel.

`suggest_next_tool_mcp` (`server.py:3077`) has an inline
hits-lookup loop that *could* be hoisted, but the loop is the only
deviation from "glue + delegate".

---

## 5. Findings — concrete divergences

Each finding includes file:line for verification, the actual asymmetry,
and an estimate of severity (`!` = bug; `~` = correctness/UX hazard;
`.` = stylistic only).

### 5.1 Three different "graph unavailable" messages — `~`

The shared helper at `server.py:1386`:
```
"graphify graph not loaded. Run `graphify .` in the repo root to ..."
```

`blast_radius_tool` (`server.py:3634`) hand-rolls:
```
"graphify mirror not loaded — cannot compute blast ..."
```

`verify_claim_tool` (`server.py:3377`):
```
"graphify mirror is not loaded — cannot verify structural claims"
```

`contract_tests_tool` (`server.py:4170`) doesn't include a
human-readable message at all — only `notes=["graph_loaded=False"]`.

Same conceptual signal, four different phrasings. MCP clients see
inconsistent UX for the same failure mode. CLAUDE.md invariant #4
requires graph-gated tools to "never return silent empty" and to
return an actionable message — `contract_tests_tool` is arguably out
of compliance.

### 5.2 Raw `dict` vs Pydantic return for graph tools — `~`

`find_callers`/`find_related`/`get_community`/`find_rationale`/`find_tests_for`/`trace_feature`
return raw `dict`. `blast_radius`/`test_impact`/`contract_tests`/`verify_claim`
return Pydantic models with a `graph_loaded: bool` field.

The MCP client sees two different envelope conventions for tools that
are conceptually a family. Schema introspection (`tools/list`)
distinguishes them, but a model using both has to handle two failure
shapes.

### 5.3 `find_rationale` double-gate has no equivalent elsewhere — `.`

`server.py:2338`:
```python
if not has_rationale_edges(app.storage):
    return {"symbol": symbol, "rationale": [], "message": "..."}
```

`blast_radius_tool` (`server.py:3650`) uses the same
`has_rationale_edges` predicate to conditionally omit a rationale
sub-call rather than short-circuit. Same edge-state signal, two
behaviours; both are defensible, but a `_rationale_unavailable()`
helper would make the asymmetry explicit.

### 5.4 Three copies of the same `SELECT normalized_key` — `~`

```
server.py:1587   recall          "SELECT normalized_key FROM repos WHERE id = ?"
server.py:2175   recall_episodes "SELECT normalized_key FROM repos WHERE id = ?"
server.py:2224   co_occurrence   "SELECT normalized_key FROM repos WHERE id = ?"
```

(The corresponding `app.storage._conn.execute(...)` call sites are
one line above each SELECT — at 1586 / 2174 / 2223.)

Three tools reach into `storage._conn` directly. This:

- Violates the spirit of CLAUDE.md invariant #3 (storage writes must
  live in `storage/sqlite.py`) — the rule is about writes, but the
  same encapsulation principle covers reads that hardwire schema.
- Hardwires the `repos` table layout in three places. A future schema
  migration that renames the column has to grep server.py.

### 5.5 Memory-tool no-repo signals are asymmetric — `~`

| Tool | Function def | No-repo return | Signal shape |
|---|---|---|---|
| `remember` | s1517 | s1529 | `{ok: False, error: "no target repo indexed"}` |
| `recall` | s1561 | s1573 | `{entries: [], repo: None}` (silent empty) |
| `forget` | s1609 | s1614 | `{removed: False}` — indistinguishable from "entry not found" |

Three different conventions for the same conceptual error. A client
that wants to react to "no repo" specifically has to special-case
each tool.

### 5.6 `expand_chunk` and `expand_chunks` duplicate body logic — `.`

`expand_chunk` (`server.py:1283`) and `expand_chunks` (`server.py:1344`)
have identical bodies modulo a loop. `expand_chunks` re-imports
`apply_level_to_snippets`. The batch version could trivially be:
```python
return ExpandChunksResponse(items=[_expand_one(cid) for cid in chunk_ids])
```

This is a cosmetic redundancy but the kind of place where divergence
creeps in (someone fixes a bug in one but not the other).

A related find: the comment at `server.py:1306-1307` inside
`expand_chunk` says "we convert it to a SearchHit-shaped dict via the
same path `lookup_by_symbol` uses". `lookup_by_chunk_id` does not
actually use `lookup_by_symbol`. The comment is stale and should be
fixed alongside the dedup.

### 5.7 Inconsistent exception handling — `~`

Only two tools mask `StrophaError`:

- `search_code` (`server.py:729`) catches and returns
  `SearchResponse(total_candidates=0, ...)`.
- `smart_search_tool` (`server.py:1161`) catches and returns
  `SmartSearchResponse(intent="search_code", confidence=0.0, ...)`.

Every other tool relies on `@traced` to surface failures (which then
get wrapped by FastMCP as `CallToolResult(isError=True)`). The two
search tools fail soft and *differently*. The choice may have been
deliberate (search degrading gracefully is desirable), but a comment
or shared helper would make the intent legible.

`report_bad_result` (`server.py:2452`) catches a *different* error
class (`FeedbackError`) and returns `{ok: False, error: ...}`.

A fourth, broader convention: `suggest_next_tool_mcp` (`server.py:3101`)
swallows bare `Exception` defensively (`except Exception:` inside its
hits-lookup loop). That gives four distinct exception-handling
conventions across 48 tools: rely on `@traced`; mask `StrophaError`
and return a typed empty response (two flavours); catch a domain
error and return `{ok: False}`; catch any `Exception` silently.

### 5.8 `_ctx` discarded for "signature parity" — `.`

`server.py:3205` (`list_task_recipes`) and `server.py:3247`
(`apply_task_recipe`):
```python
_ = _ctx(ctx)  # signature parity with the rest of the surface
```

`_ctx` has a side effect (binds `session_id` into structlog). Calling
it just for the side effect and discarding the return reads as a
code smell. The comment is honest about the intent — but the helper
should expose a separate `_bind_session(ctx)` for the side-effect
case so the call site can express what it's actually doing.

### 5.9 `audit_citations_tool` passes a file path into `lookup_by_symbol` — `!`

`server.py:3941`:
```python
if cit.kind == "file_line":
    hits = app.storage.lookup_by_symbol(cit.path, limit=20)
```

`lookup_by_symbol` (`storage/sqlite.py:2314`) queries
`WHERE c.symbol = ? OR c.symbol LIKE ?`. Passing `src/foo/bar.py` will
only match a chunk whose **symbol column** equals or fuzzy-matches
that path — which is essentially never true.

The downstream loop `for hit in hits or []:` then iterates an empty
list, and the function returns `False` for every `file_line` citation.
This is a bug: file-line citation audit silently never resolves any
citation.

The right fix is a `storage.chunks_overlapping_lines(rel_path, start, end)`
method (which doesn't exist yet) called from this branch.

### 5.10 `spec_consistency_tool` resolves the repo differently — `.`

`server.py:4063-4074`:
```python
app = _ctx(ctx)
# Use the first registered repo's local path as the spec root.
try:
    repos = app.storage.stats().get("repos") or []
except Exception:
    repos = []
repo_root = ""
for r in repos:
    path = r.get("local_path") or r.get("path") or ""
    if path:
        repo_root = path
        break
```

Every other repo-scoped tool uses `_resolve_target_repo_id(app)`. This
one walks `stats()['repos']` and picks the first entry that has a
local_path / path — which works in single-repo deployments and
silently picks an arbitrary repo in multi-repo indexes.

---

## 6. Simplification proposals

Ranked by impact / cost ratio.

### 6.1 Introduce a `GraphGated` decorator — high impact, low risk

Covers 10 of 48 tools (§4.3 + §4.4). Idea:

```python
def graph_gated(typed_response_class=None):
    """Short-circuit the wrapped tool when the graphify mirror is empty.

    If `typed_response_class` is None, returns the raw _graph_unavailable() dict.
    Otherwise constructs `typed_response_class(graph_loaded=False, notes=[...])`.
    """
    def decorator(fn):
        @functools.wraps(fn)
        def wrapper(*args, ctx, **kwargs):
            app = _ctx(ctx)
            if not graph_loaded(app.storage):
                if typed_response_class is None:
                    return _graph_unavailable()
                return typed_response_class(
                    graph_loaded=False,
                    notes=[_GRAPH_UNAVAILABLE_MESSAGE],
                )
            return fn(*args, ctx=ctx, app=app, **kwargs)
        return wrapper
    return decorator
```

Benefits: one canonical message; the gate check stops being copy-pasted;
the four typed-return tools stop hand-rolling the response.

Cost: each tool body needs `app` injected. Stylistic break with the
current "tools never take `app`" convention.

**Risk:** none — the four typed-return tools already return a
"graph_loaded=False" envelope; the change is internal.

### 6.2 `Storage.resolve_repo_name(repo_id)` method — high impact, trivial

Replaces three copies of inline SQL (§5.4). Single 4-line method on
`Storage`. Pure win.

### 6.3 Unified `repo_unavailable` envelope for memory tools — medium impact, breaking

Pick one convention for the three memory writers and four memory
readers (§5.5). Recommended: typed `RepoUnavailable` envelope on
every response, with `repo: str | None` and `error: str | None`.

**This is a breaking API change** for any MCP client currently
checking `ok` vs `removed` vs empty `entries`. Worth coordinating
with the HTTP-only cutover messaging — the breaking-change budget is
already spent there.

### 6.4 Fix `audit_citations` `file_line` lookup — must-fix bug

§5.9. Add `Storage.chunks_overlapping_lines(rel_path, start, end)`,
swap the `lookup_by_symbol` call. Independent of the broader refactor.

### 6.5 Migrate raw-dict graph tools to Pydantic — medium impact, breaking

§5.2. Pair with 6.1 — the `GraphGated` decorator already needs the
typed_response_class. The six raw-dict tools (`find_callers` et al)
gain a typed envelope.

Breaking change for any agent that parses the raw `dict` shape (which
includes Claude Code's current tool consumption — it normalises both
shapes, but the JSON Schema delta is observable).

### 6.6 Split `_ctx` into `_app(ctx)` and `_bind_session(ctx)` — low impact, trivial

§5.8. Rename so the side-effect-only call sites can be explicit.
Three-line change, zero risk.

### 6.7 De-duplicate `expand_chunk` / `expand_chunks` bodies — low impact, trivial

§5.6. Extract `_expand_one(...)` once; call from both. Cosmetic but
cheap.

### 6.8 `spec_consistency_tool` switches to `_resolve_target_repo_id` — bug-class fix

§5.10. One-line fix. Removes a silent misbehaviour in multi-repo
indexes.

---

## 7. What NOT to refactor

- **`search_code`'s inline orchestration.** The 300-line body
  (`server.py:623`) reads as a list of WIN-* tickets layered over one
  another. Every block has comments explaining sequencing constraints
  (dedup before pins, pins before compression, etc.). A premature
  refactor will lose the rationale ordering and re-introduce bugs
  that the layering was designed to fix.
- **Session-mutator family unification.** The six session mutators
  (§4.5) are already cheap. The mutations differ enough (sets vs
  strings vs scalar) that a generic `mutate(state, fn)` would obscure
  intent without saving code.
- **Pure pass-through tools.** Already minimal; nothing to extract.

---

## 8. Open questions

1. **Public API breakage budget.** Are we willing to break the raw-dict
   graph tools (§5.2 / §6.5) in the same release as the HTTP-only
   cutover? If yes, doing them together amortises the client-side
   pain. If no, the typed migration waits a release cycle.

2. **`_ctx` side effect.** Should `_ctx` continue to bind structlog,
   or should that move to an MCP-server middleware that runs once per
   request? Middleware would let pure pass-through tools skip the
   helper entirely.

3. **Schema drift detection for the tool layer.** We have `adapter_id`
   drift detection for the pipeline. No equivalent exists for tool
   response shapes. Worth a small registry test that asserts every
   tool's JSON Schema is a stable function of its return type?

4. **`audit_citations` bug severity.** §5.9 is a real bug — silent
   `False` for every `file_line` citation against a real SQLite
   storage. Tests at `tests/unit/test_citation.py:327, 338, 349, 369`
   do exercise the `file_line` branch, but they stub
   `storage.lookup_by_symbol` with a path-keyed dict
   (`side_effect = lambda path, limit: symbols.get(path, [])[:limit]`),
   which masks the actual SQL behaviour. The fix needs both the
   storage method change *and* a real-SQLite fixture so the test
   covers the new path.

---

## 9. Recommendation

Phase the work to match risk:

1. **Now, low-risk:** §6.2 (storage method), §6.4 (audit_citations bug),
   §6.6 (_ctx split), §6.7 (expand_chunks de-dup), §6.8 (spec_consistency
   repo resolution). Five small PRs, no breaking changes.
2. **Next, medium-risk:** §6.1 (`GraphGated` decorator). Internal
   refactor; the typed-return tools change their internals but their
   external shape is unchanged.
3. **Bundled with next breaking release:** §6.3 (memory envelope) and
   §6.5 (raw-dict → typed graph tools). Coordinate with whatever the
   next BREAKING entry is in the changelog.

A single follow-on PR can land phases 1 and 2 cleanly. Phase 3 is the
one that needs an architecture-level go/no-go decision.
