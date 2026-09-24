# DeepSeek Harness (Cordis) as a structure for a stropha-enforced agent

Status: **investigation / proposal** — not a spec. No code changes.
Date: 2026-08-25. Verified against the cited sources on this date; local claims
refer to stropha at HEAD `335d57d`.

## 1. Question

DeepSeek recently released **DeepSeek Harness (dsh)** — an open-source agent
harness whose tagline is "Everything is a Plugin". The composition model it
implements is formalized in a separate Cordiverse paper, co-authored by a
DeepSeek researcher [R1][R2][R5]. We want to know:

1. What is the structure of dsh and the underlying paper?
2. How can that structure be reused to build a stropha agent that **enforces**
   work execution through stropha MCP tools, instead of letting the model fall
   back to host tools (raw `Grep`/`Glob`/`Read`) or its own memory?

Key constraint from this repo: **stropha stays context infrastructure and never
becomes an agent runtime** [R21]. So the "agent" lives in a host runtime
(Claude Code, opencode, a future thin loop); dsh-shaped **enforcement** is what
we adopt, not dsh's runtime.

## 2. What DeepSeek shipped

### 2.1 DeepSeek Harness (dsh)

Repository facts (checked via GitHub API on 2026-08-25) [R1]:

| Field | Value |
|---|---|
| Repo | `deepseek-ai/deepseek-harness` |
| Description | "DeepSeek Harness: Everything is a Plugin." |
| Created | 2026-08-13 |
| Last push | 2026-08-21 |
| License | MIT |
| Stars | 194,040 |

DeepSeek's own news feed announces the harness as a **developer preview**
(news item 2026-08-21) [R5]. A repository created 12 days before this
investigation, with 194k stars, is a viral-release signal — not a maturity
signal. Copy the *patterns*, not the code.

The enforcement-relevant machinery, from dsh's own docs [R3][R4]:

| dsh mechanism | Behavior |
|---|---|
| Tool registry (`ToolDefinition`) | Tools are first-class registrations; no implicit tool surface |
| `ToolRestriction` | Scoped allow/deny filters; multiple restrictions compose by intersection |
| `ToolGuard` | **Monotonic, deny-only**: once a tool is denied it stays denied regardless of ordering |
| `tools/pre-execute` | Waterfall of handlers returning `allow` / `deny` / `ask` (human checkpoint) |
| `tools/post-execute` | Waterfall returning `accept` / `replace` / `block`, with corrective feedback returned to the model |
| Profiles / bundles / patches | Layered capability sets; `dsh-base` is the first layer; patches layer on top |
| Agent presets | Preconfigured role = scoped registration + isolation realm |
| Reversible effects | Tool side effects wrapped so agent state can roll back |
| Session invariant | "Model-visible ⇒ logged" — anything the model sees is recorded |

The design lesson we take: **capability is enforced at the registry boundary**
(what exists, what is allowed, what can never be allowed), with prompt text as
the weakest and last layer.

### 2.2 The paper

The design paper behind the structure is:

> Yifan Shi, Wei Zhang (Peking University), Tianyi Cui (DeepSeek-AI).
> "A Programming Paradigm for Spatiotemporal Composability."
> https://github.com/cordiverse/paper (`paper.pdf`, 88 pages) [R2]

Two framing notes:

- The paper is a **Cordiverse** (academic) publication; DeepSeek's formal
  involvement is one co-author plus the dsh implementation. It is not a
  DeepSeek-lab technical report.
- Its content (revertible effects, reactive coeffects, a unified context
  paradigm, a dynamic-composition calculus; implemented as **Cordis**, with
  **Koishi** as case study) is a *programming model* for composable agent
  plugins. For our purposes the load-bearing part is the same one dsh
  documents: scoped registration + policy layers over a tool registry.

### 2.3 Independent validation of the harness approach

dsh did not invent the pattern. Three independent 2026 engineering write-ups
converge on the same architecture:

- **OpenAI** (Feb 2026): a product built with "0 lines of manually-written
  code" by Codex agents; ~1M LOC, ~1,500 merged PRs, 3.5 PRs/engineer/day.
  Core principle: **"By enforcing invariants, not micromanaging
  implementations, we let agents ship fast without undermining the
  foundation"** — enforced with custom linters whose error messages carry
  remediation instructions. Their "one big `AGENTS.md`" attempt failed for
  four documented reasons: context is scarce, too much guidance becomes
  non-guidance, the file rots instantly, and it can't be mechanically
  verified [R6].
- **Anthropic** (Mar 2026): three-agent **planner / generator / evaluator**
  harness for long-running builds; "every component in a harness encodes an
  assumption about what the model can't do on its own, and those assumptions
  are worth stress testing" [R7].
- **MAST** (arXiv preprint, v3, 2025): 1,600+ annotated
  traces across 7 multi-agent frameworks yield 14 failure modes in 3
  categories — system design issues, inter-agent misalignment, and task
  verification — κ = 0.88 inter-annotator agreement [R9].

All three say the same thing our question asks: reliability comes from
structure (roles, boundaries, mechanical checks), not from better prompting.

## 3. Why "use stropha tools" cannot be a prompt instruction

Evidence that prompt-level tool guidance fails or erodes:

1. **Instruction overload degrades performance.** Chroma's *Context Rot* study
   (18 LLMs, Jul 2025) shows performance degradation as context grows is
   non-uniform and is amplified by distractors — a long "please always prefer
   `search_code` over grep" preamble is exactly such a distractor [R10].
   Anthropic's context-engineering post reaches the same conclusion [R19].
2. **Agents re-optimize for the path of least resistance.** OpenAI's team
   found agents "pattern-match locally instead of navigating intentionally"
   when guidance is undifferentiated; the fix was mechanical enforcement, not
   more words [R6].
3. **Verification failures are a first-class failure class.** MAST's third
   category (task verification) covers agents reporting completion without
   evidence — the exact failure mode "the model answered from memory instead
   of querying the index" [R9].
4. **Self-evaluation is biased.** Anthropic observed agents "confidently
   praising the work" of their own generation; separating a tuned, skeptical
   evaluator from the generator "turns out to be far more tractable than
   making a generator critical of its own work" [R7].
5. **Tool metadata itself is an attack surface.** If tool *descriptions* can
   steer behavior, then guidance embedded in descriptions is both fragile and
   dangerous (Section 6.1) [R11][R12].

Conclusion: "must use stropha tools" belongs in the **registry + pre/post
execution policy**, with the model-facing description kept minimal and
truthful.

## 4. Proposed design: stropha agent on the dsh structure

### 4.0 Ground rules (local invariants)

- Stropha never becomes the agent runtime [R21].
- MCP is HTTP-only; one shared daemon per instance; `mcp.profile` is
  **daemon-wide and static** (`core` | `standard` | `full`); no per-session
  profile overrides [R20, invariant 2].
- `src/stropha/server_tool_manifest.py` is the canonical 55-tool inventory
  (13 core, +27 standard, +15 full) [R22].
- Registration is all-or-nothing: startup fails unless every manifest tool is
  registered in manifest order [R24]. Capability restriction therefore happens
  at discovery and invocation, never by withholding registration.
- No global state; new behavior goes through the adapter framework or an
  explicit new dispatch stage; spec drift requires a `docs/architecture/`
  update in the same commit [R20].

### 4.1 Roles (dsh "agent presets")

Four roles, each a scope + allowlist + guards + hooks:

| Role | Job | Profile floor |
|---|---|---|
| `planner` | Turn a request into a scoped plan; pick recipes; never read raw files | `full` |
| `retriever` | All codebase questions answered via stropha index only | `standard` |
| `coder` | Edits, grounded on stropha-sourced context; pre-flight symbol checks | `full` |
| `verifier` | Independent pass: citations, claims, spec, tests; blocks with feedback | `full` |

"Profile floor" is the minimum daemon profile that makes the role's allowlist
callable, derived mechanically from the manifest [R22]. Three of the four
roles land on `full` because memory (`remember`/`recall`), scratchpad,
`co_occurrence`, `check_staleness` and `report_bad_result` are full-only
tools; only `retriever` fits inside `standard`.

That has a consequence worth stating plainly, because it is easy to get
backwards: since `mcp.profile` is daemon-wide and static [R20, invariant 2],
a single-daemon deployment must run `full`, and therefore **the profile is not
what separates the roles — the layer-2 allowlist is**. The profile only sets
the ceiling. Per-role daemons (Section 5, option (a)) buy isolation only for
`retriever`; every other role would still be started on a `full` daemon. This
is the strongest argument for building layer 2 rather than treating profiles
as the isolation mechanism.

### 4.2 Tool allowlist matrix

Derived from the canonical manifest [R22]. A role sees `profile ∩
role-allowlist`.

One divergence from dsh has to be stated up front, because it rules out the
obvious implementation: stropha **cannot** enforce this by not registering
tools. `validate_tool_manifest()` fails daemon startup unless *every* manifest
entry is registered, in exact manifest order [R24]. dsh's scoped-registration
semantics [R3] are therefore unavailable. The allowlist must instead reuse the
two seams profiles already flow through — **discovery**
(`visible_tool_names()` → `list_tools()`) and **invocation**
(`tool_is_available()` → `call_tool()`, which already raises `ToolError`
before dispatch) [R24]. Practically this is good news: layer 2 is an extra
predicate on two existing functions, not a new registry.

| Role | Allowed stropha tools |
|---|---|
| `planner` | `bootstrap_environment`, `list_repos`, `index_health`, `recall`, `remember`, `plan_investigation`, `assemble_agent_context`, `list_task_recipes`, `apply_task_recipe`, `scratchpad_read`, `scratchpad_write`, `scratchpad_append` |
| `retriever` | `search_code`, `get_symbol`, `get_file_outline`, `smart_search_tool`, `find_callers`, `find_related`, `find_tests_for`, `find_rationale`, `trace_feature`, `get_community`, `expand_chunk`, `expand_chunks`, `pin_chunks`, `unpin_chunks`, `list_pins`, `reset_dedup`, `assemble_context`, `execute_investigation`, `assemble_agent_context`, `suggest_next_tool_mcp`, `working_set_tool`, `list_repos`, `index_health` |
| `coder` | `get_symbol`, `get_file_outline`, `expand_chunk`, `expand_chunks`, `resolve_agent_instructions`, `check_symbols_tool`, `verify_claim_tool`, `blast_radius_tool`, `working_set_tool`, `scratchpad_read`, `scratchpad_append`, `recall`, `co_occurrence` |
| `verifier` | `audit_citations_tool`, `verify_claim_tool`, `assert_snippet_freshness_tool`, `spec_consistency_tool`, `test_impact_tool`, `blast_radius_tool`, `contract_tests_tool`, `check_staleness`, `index_health`, `record_runtime_artifact`, `report_bad_result` |

Notes:

- **Host tools are the real fallback.** An MCP server cannot unregister the
  host's own `Grep`/`Glob`/`Read`/`Bash`. Closing that hole requires the host
  client's permission settings (per-role client config, shipped with the
  integration guide [R23]). The server-side layers below are necessary but not
  sufficient; this split is documented as a known gap, not hidden.
- `retriever` deliberately has **no** host file-read access: "how does X
  work?" may only be answered from `search_code` / `get_symbol` / graph tools,
  so a cold or stale graph surfaces as `{"graph_loaded": false}` [R20,
  invariant 6] or `index_health` drift instead of silently being answered
  from a stale grep.
- Memory tools (`remember`/`recall`, episodes, `check_staleness`) give the
  coder a durable store, but a recalled fact must be re-validated with
  `check_staleness` before it is acted on [R22]. No equivalent
  provenance-drift check surfaced in the dsh documents reviewed for this
  investigation [R3][R4] — record that as "not found", not "does not exist".

### 4.3 Enforcement layers (strongest first)

Mirrors dsh's pipeline [R3][R4], mapped onto stropha's dispatch:

| # | Layer | dsh analog | stropha implementation point |
|---|---|---|---|
| 1 | Profile membership | profiles | exists: `tool_is_available()` + `visible_tool_names()` [R24] |
| 2 | Role allowlist (invisible **and** uninvocable) | `ToolRestriction` | extra predicate on those same two functions [R24]; **not** achievable via non-registration |
| 3 | Deny-only monotonic guard | `ToolGuard` | new module; pure function; e.g. "retriever: no state writes", "never `cache_prune` without an explicit cutoff" |
| 4 | Pre-execute waterfall (`allow`/`deny`/`ask`) | `tools/pre-execute` | hook chain inside `call_tool()` before `super().call_tool()`; the deny path already exists as `ToolError` [R24] |
| 5 | Post-execute gate (`accept`/`replace`/`block` + corrective feedback) | `tools/post-execute` | wrap the `super().call_tool()` return value in the same override [R24] |
| 6 | Prompt text | system prompt | weakest; last resort only |

The guard is deny-only and monotonic on purpose: a later, more permissive
policy can never re-open a deny — this is the property dsh's docs highlight
and the one that makes the stack auditable [R3][R4].

### 4.4 The verification loop (evaluator–optimizer)

The shape is Anthropic's **evaluator–optimizer** workflow [R8], instantiated
with stropha's verification tools as the measuring instrument rather than an
LLM's opinion. `verifier` is a **separate session/role**, never sharing
context with `coder`:

- After a coding turn, the verifier runs: `audit_citations_tool` (every claim
  carries a resolvable `[chunk_id]`/`[file:line]`), `verify_claim_tool`
  (structural claims checked against the graph),
  `assert_snippet_freshness_tool` (cached snippets vs live index),
  `spec_consistency_tool` (changed symbols vs `CLAUDE.md`/`docs/architecture/`
  invariants), `test_impact_tool` (minimal test set) [R22].
- A failed check is a post-execute **block with corrective feedback** — the
  exact dsh mechanism [R4] — so the coder receives an actionable reason, not a
  silent rejection.
- Anthropic's evidence: out-of-the-box LLM QA "talked itself into deciding
  issues weren't a big deal"; the evaluator had to be tuned to be skeptical
  over several rounds [R7]. Expect the same tuning loop here.

### 4.5 Recipes ≈ bundles/patches

Stropha already has the declarative layer: `list_task_recipes` /
`apply_task_recipe` materialize named workflow templates with placeholder
substitution [R22]. In dsh terms these are bundles/patches [R3]. Proposed
usage: planner's first move on a known task shape (bug-fix, add-feature,
rename) is `apply_task_recipe`, and the resulting step list *is* the
allowlist-scoped work order for the turn.

### 4.6 Context handling

- Per-turn context is budgeted: `assemble_context` / `working_set_tool` under
  explicit token budgets, pinned chunks for the working set [R22] —
  countering context rot [R10].
- Cross-session handoff uses `scratchpad_write` + `remember` (durable,
  key-anchored) instead of ever-growing chat context [R22]. This mirrors the
  structured-handoff pattern Anthropic found necessary for long-running
  builds, including context *resets* with a handoff artifact rather than
  compaction alone [R7][R18].
- Retrieval results are traced: `search_code` and `smart_search_tool` return a
  top-level `trace_id` matching the JSONL trace record that
  `report_bad_result` refers to [R22]. That is **partial** coverage of dsh's
  "model-visible ⇒ logged" invariant [R3] — graph, memory and verification
  tools expose no comparable handle. Full coverage is a prerequisite for the
  post-incident forensics Section 6.2 depends on; it is not something we can
  claim to have today.

## 5. Gap analysis against stropha's invariants

| dsh feature | Available in stropha today? | Gap / decision needed |
|---|---|---|
| Static profiles | Yes [R22] | — |
| Per-role capability sets | **No** | `mcp.profile` is daemon-wide and static [R20, invariant 2]. Options: (a) one daemon instance per role on distinct ports (fits the singleton-per-daemon model, zero invariant changes); (b) a future *session-scoped* registry filter enforced server-side that still derives from the static manifest (needs an explicit design decision + spec update per hygiene rule 6). Recommend (a) first. |
| Scoped registration (per-agent registry) | **No — and blocked** | `validate_tool_manifest()` requires complete, exactly-ordered registration at startup [R24]. Do not attempt; enforce at discovery + invocation instead. |
| Pre/post-execute waterfalls | **Partial** | The single-point deny seam already exists (`call_tool()` raises `ToolError` when a tool is unavailable [R24]). Missing: the ordered hook chain, the `ask` verdict, and any post-execute gate. Extend that override; adapter-registered per hygiene rule 2 [R20]. |
| Monotonic deny-only guard | **No** | Small new module; pure function over (role, tool, args) ⇒ decision. |
| Reversible effects | **No** | Stropha tools are read-mostly, but the persistent-state surface is wider than it looks: `remember`, `forget`, `record_episode`, `cache_prune`, `scratchpad_write`/`_append`/`_clear`, `pin_chunks`/`unpin_chunks`, `reset_dedup` [R22]. None is transactional, so layer 3 must **prevent** rather than roll back. |
| Bundles/patches | Yes (recipes) [R22] | Extend recipe coverage. |
| Context objects (unified context) | Partial (scratchpad + pins + working set) | No change proposed. |
| Host-tool denial | **Impossible server-side** | Client-side permission templates per role [R23]. |

## 6. Risks and open problems (each with its reference)

### 6.1 Tool metadata is an attack surface

Tool poisoning — malicious instructions embedded in tool *descriptions* —
measured across 45 live MCP servers, 353 authentic tools, 1,312 malicious
cases: o1-mini hit **72.8% attack success rate** [R11]. Variants and
follow-ups:

- **Implicit poisoning** (MCP-ITP): the poisoned tool is never invoked; its
  description steers the agent to invoke a *legitimate* high-privilege tool
  [R12].
- **Defenses** must derive descriptions from the implementation, not trust
  prose (TRUSTDESC: static slice → description synthesis → dynamic
  verification) [R13]; OAuth-anchored tool definitions for squatting/rug-pull
  (ETDI) [R14]; broader MCP-ecosystem attack surveys [R15][R16].

Design consequence for stropha: the manifest is static and hand-owned
[R22]; **no** tool description may ever be assembled from user content,
index content, or fetched metadata. The verifier role's own tool descriptions
get the same treatment — a poisoned `audit_citations_tool` description would
defeat the loop.

### 6.2 Even dsh's own enforcement is penetrable

An independent security assessment of DSH (14,560 controlled executions, 16
indirect-content channels, 35 payload objectives, 12 attack methods) found
indirect prompt injection still succeeds at meaningful rates: **25.5%**
attack success (rule judge, hidden-Unicode payloads in file mode), **17.0%**
(LLM judge, fake-completion in text mode), **16.0%** (rule judge, skills
channel in file mode) [R17]. Its conclusion matches this investigation's:
the controls that matter sit **between untrusted content and the agent's
tool-call policy** — i.e., exactly layers 2–5 of Section 4.3. Implications:

- Enforcement at the registry is necessary but **not sufficient**;
  untrusted-content taint tracking is the open research gap [R17].
- "Model-visible ⇒ logged" (dsh) / stropha's `trace_id` [R22] is what makes
  such attacks forensically recoverable; keep it intact.
- The **fake-completion** attack (17.0%, LLM judge, text mode) is the one that
  directly threatens our verifier: it induces the agent to *claim* the work is
  done. Mitigation: a verdict must be the stropha tool's structured output,
  never the model's prose summary of it — the orchestrator reads the JSON, not
  the narration [R17][R22]. Note also that the two judges disagree
  systematically (partial compliance: 7.3% under the LLM judge vs 2.0% under
  the rule judge [R17]), so any single ASR figure is judge-dependent and
  should not be quoted as *the* number.

### 6.3 Self-evaluation bias

Separate the evaluator; tune it to be skeptical; expect several tuning rounds
[R7]. The verifier role must not share the coder's session state beyond the
structured artifacts (plan, changed files, test run).

### 6.4 Cost

Anthropic's benchmark: solo run 20 min / **$9**; full planner+generator+
evaluator harness 6 h / **$200** (>20×); after removing the sprint construct
on a stronger model, 3 h 50 min / **$124.70** [R7]. Design consequence:
**tiered escalation**. Default path is a single role on the `core`/`standard`
profile; the full role team is engaged only when the task exceeds what the
model does reliably solo — which is precisely the load-bearing test Anthropic
applies to every harness component [R7].

### 6.5 Harness staleness

"Every component in a harness encodes an assumption about what the model can't
do on its own" — and those assumptions go stale as models improve; Opus 4.6
made their sprint construct unnecessary [R7]. Each allowlist entry should
therefore carry a reason ("model still falls back to raw grep for X"), and
the matrix must be re-derived on every model generation, with unused layers
deleted rather than accumulated.

### 6.6 Entropy / drift

Fully agent-driven repositories accumulate "AI slop" and pattern-drift;
OpenAI's fix is recurring background cleanup agents enforcing encoded "golden
principles" [R6]. Stropha analog: periodic `spec_consistency_tool` +
`index_health` sweeps over `docs/architecture/` and the manifest [R20][R22].

### 6.7 dsh-specific risk

Developer-preview status, 12 days old at time of writing, TypeScript runtime,
academic paper with a single DeepSeek co-author [R1][R2][R5]. Treat dsh as a
*reference implementation of the pattern set*, not a dependency. If/when dsh
stabilizes, the interesting integration is a **stropha plugin for dsh**
(scoped registration of the stropha MCP surface into dsh's registry), not
vendoring dsh into stropha.

## 7. Phased plan

| Phase | Deliverable | Code? |
|---|---|---|
| 0 | This document + role×tool matrix ratification; per-tool "reason" column filled in | No |
| 1 | Role-allowlist predicate added to `visible_tool_names()` + `tool_is_available()`, plus the deny-only guard module [R24]; per-role daemon instances (Section 5, option (a)); `uv run pytest -q` green; spec doc updated same commit per hygiene rule 6 | Yes |
| 2 | Pre/post-execute waterfall dispatch stage; block-with-correction feedback | Yes |
| 3 | Declarative role presets in `~/.stropha/config.yaml` (new `agent:` section; config remains single source of truth [R20]); recipe coverage for bug-fix/add-feature/rename flows | Yes |
| 4 | Verifier loop wiring (machine-JSON verdicts only) + per-role host-client permission templates in `docs/guides/mcp-integration.md` | Yes |

Exit criteria per phase: invariants [R20] intact, tests green, spec drift
rule satisfied, and — per Section 6.5 — each added layer has a written
"model-fails-without-this" rationale that gets re-checked on model upgrades.

## 8. How we would know it works

Enforcement without measurement is theatre. Each phase has to move a number,
and none of these numbers is interpretable without a **pre-enforcement
baseline captured on the same task set** — so Phase 0 includes recording that
baseline. Shipping layers without it would reproduce, on this project, the
failure class MAST calls "task verification" [R9].

| Signal | Measurement | Wanted |
|---|---|---|
| Host-tool fallback rate | share of task turns where the host's `Grep`/`Glob`/`Read` fires before any stropha retrieval call (host transcript) | down |
| Grounding rate | share of edits preceded by ≥1 `get_symbol`/`search_code`/`expand_chunk` touching the edited file | up |
| Citation hygiene | `audit_citations_tool` uncited + broken counts per verifier run [R22] | down |
| Snippet staleness | `assert_snippet_freshness_tool` drift hits per run [R22] | down |
| Spec violations caught pre-merge | `spec_consistency_tool` HIGH findings per change set [R22] | up, then down |
| Cost per completed task | tokens/USD vs the solo baseline [R7] | inside budget |

The first two are the ones that actually answer the question this
investigation started from; the rest measure whether the verifier loop is
worth its cost.

## 9. References

External (all fetched/verified 2026-08-25):

- [R1] DeepSeek Harness repository — https://github.com/deepseek-ai/deepseek-harness (facts via GitHub API, 2026-08-25).
- [R2] Shi, Zhang, Cui. "A Programming Paradigm for Spatiotemporal Composability." https://github.com/cordiverse/paper (paper.pdf).
- [R3] dsh architecture docs — https://github.com/deepseek-ai/deepseek-harness/blob/master/docs/architecture.md (ToolRestriction, ToolGuard, profiles/bundles/patches, scoped registration, reversible effects).
- [R4] dsh tools subsystem docs — https://github.com/deepseek-ai/deepseek-harness/blob/master/docs/subsystems/tools.md (pre-execute allow/deny/ask; post-execute accept/replace/block).
- [R5] DeepSeek news (harness developer preview, item 2026-08-21) — https://api-docs.deepseek.com/news.
- [R6] Lopopolo, R. (OpenAI). "Harness engineering: leveraging Codex in an agent-first world." 2026-02-11. https://openai.com/index/harness-engineering/
- [R7] Rajasekaran, P. (Anthropic). "Harness design for long-running application development." 2026-03-24. https://www.anthropic.com/engineering/harness-design-long-running-apps
- [R8] Anthropic. "Building effective agents." 2024-12-19. https://www.anthropic.com/research/building-effective-agents
- [R9] "Why Do Multi-Agent LLM Systems Fail?" (MAST). arXiv:2503.13657.
- [R10] Chroma. "Context Rot." 2025-07-14. https://www.trychroma.com/research/context-rot
- [R11] "MCPTox: A Benchmark for Tool Poisoning Attack on Real-World MCP Servers." arXiv:2508.14925.
- [R12] "MCP-ITP: An Automated Framework for Implicit Tool Poisoning in MCP." arXiv:2601.07395.
- [R13] "TRUSTDESC: Preventing Tool Poisoning in LLM Applications via Trusted Description Generation." arXiv:2604.07536.
- [R14] "ETDI: Mitigating Tool Squatting and Rug Pull Attacks in MCP by using OAuth-Enhanced Tool Definitions and Policy-Based Access Control." arXiv:2506.01333.
- [R15] "Beyond the Protocol: Unveiling Attack Vectors in the Model Context Protocol (MCP) Ecosystem." arXiv:2506.02040.
- [R16] "Systematization of Knowledge: Security and Safety in the Model Context Protocol Ecosystem." arXiv:2512.08290.
- [R17] "Security Assessment of DeepSeek Harness with A.I.G: Evaluating Resistance to Indirect Prompt Injection." arXiv:2608.16393.
- [R18] Anthropic. "Effective harnesses for long-running agents." https://www.anthropic.com/engineering/effective-harnesses-for-long-running-agents (linked from [R7]).
- [R19] Anthropic. "Effective context engineering for AI agents." https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents (linked from [R7]).

Local (stropha, at HEAD `335d57d`):

- [R20] `CLAUDE.md` — invariants (HTTP-only MCP, static profiles, canonical manifest, forward-only schema, graph-gated tools, hygiene rules).
- [R21] `docs/architecture/stropha-agent-exoskeleton.md` — "Stropha never becomes the agent runtime."
- [R22] `src/stropha/server_tool_manifest.py` — 55-tool canonical inventory; core 13, standard +27, full +15. Where cited for the *behaviour* of an individual tool (e.g. `search_code`'s `trace_id`, recipe placeholder substitution), the claim refers to that tool's registered MCP description and handler in `src/stropha/server.py`.
- [R23] `docs/guides/mcp-integration.md` — host-client integration guide (target of the per-role permission templates, Phase 4).
- [R24] `src/stropha/server_http.py` — `RestartTolerantFastMCP`: `validate_tool_manifest()` (complete, exactly-ordered registration enforced at startup), `visible_tool_names()`/`list_tools()` (discovery filter), `tool_is_available()`/`call_tool()` (invocation filter, raises `ToolError` pre-dispatch). These are the insertion points for enforcement layers 2, 4 and 5.
