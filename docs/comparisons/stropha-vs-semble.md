# stropha vs. semble — comparação detalhada

**Status:** Historical technical comparison (revision 2; not current inventory)
**Original date:** 2026-05-18
**History audit:** 2026-08-11 at stropha commit
`b0d391d7d67606a04666d8e9a48cfbef24c1726d`
**Scope:** the 2026-05-18 stropha/semble snapshots described below. `HEAD`,
"today", counts, defaults, and roadmap statements in the body refer to that
snapshot, not current HEAD. Use live registry/config introspection for current
stropha facts.
**Objetivo:** preservar a comparação histórica sem sacrificar os invariantes
nomeados em `CLAUDE.md` **Invariants (do NOT break)**.

**Fontes verificadas neste documento:**
- `pyproject.toml` (entry-points, deps, extras)
- `src/stropha/adapters/**/__init__.py` + `grep register_adapter` para contagem real de adapters
- `grep STROPHA_` no código-fonte para contagem real de env vars
- `grep @mcp.tool` em `server.py` para contagem real de tools
- Model card de `mixedbread-ai/mxbai-embed-large-v1` (HuggingFace)
- Model card de `minishlab/potion-code-16M` (HuggingFace)
- README do semble + página do repositório

> **Historical snapshot note:** the then-current `CLAUDE.md` inventory differed
> from the registry counts listed here. Current `CLAUDE.md` deliberately avoids
> copied totals; use `stropha adapters list`.

---

## 1. Sumário executivo

stropha e semble compartilham o mesmo *fit*: code-search local exposto a agentes via MCP, com chunking por tree-sitter e fusão de retrieval denso + esparso. Mas as duas ferramentas otimizam para metas diferentes:

| Dimensão | stropha | semble |
|---|---|---|
| Filosofia | RAG **completo + estrutural**: 12 ferramentas MCP, grafo, comunidades, rationale, trace_feature | Code-search **mínimo + rápido**: 2 ferramentas MCP (`search` + `find_related`) |
| Embedder padrão | `mxbai-embed-large-v1` 1024-d (fastembed/ONNX, ~0.3 B params) | `potion-code-16M` (Model2Vec, **embeddings estáticos** 256-d, ~16 M params) |
| Inferência em query | sim (forward pass ONNX no modelo denso, ou MLX em Apple Silicon via extra `[mlx-embed]`) | **não** — lookup direto numa tabela de embeddings estáticos |
| Latência de indexação típica | segundos a minutos (depende do enricher; ainda assim os defaults locais não precisam de rede após o download dos pesos) | **~250 ms** para um repo médio (README) |
| Latência de query típica | dezenas a centenas de ms (depende de quantos streams + reranker + HyDE) | **~1,5 ms** (README) |
| Install | `git clone` + `uv sync` (sem publicação em PyPI no HEAD atual) + (opcional) graphify + hook install | `pip install semble` ou `uvx --from "semble[mcp]" semble` |
| Setup MCP | edição manual de `.mcp.json`, env vars, `STROPHA_TARGET_REPO`. O script `stropha-mcp` **existe** em `pyproject.toml`, mas como o pacote não está publicado, `uvx stropha-mcp` puro não funciona — só `uvx --from <git-url> stropha-mcp`. | `claude mcp add semble -s user -- uvx --from "semble[mcp]" semble` |
| Reranking | RRF (k=60) + opcional cross-encoder `BAAI/bge-reranker-base` via fastembed | RRF + **heurísticas code-aware** (5 sinais: adaptive weighting, definition boost, identifier stem, file coherence, noise penalty) |
| Quality benchmark | golden set local (`tests/eval/golden.jsonl`, ~43 queries em 1 repo), Recall@K + MRR | NDCG@10 = 0.854 em ~1.250 queries / 63 repos (paridade com CodeRankEmbed 137M, conforme README) |
| Multi-repo / monorepo | nativo (schema v7, manifest YAML, prefixo `{repo_id}:` em node IDs) | indexa por path local ou git URL on-the-fly |
| Graph traversal | sim (graphify mirror: callers, related, community, rationale, trace_feature) | não — apenas `find_related` por similaridade |
| Aware of git history | hook v=4 incremental + adapter `git-diff` walker + rename-resilient `chunk_id` | file-watcher (README: "local paths are watched for file changes and re-indexed automatically", sem detalhes de implementação) |
| Persistência de índice | SQLite + sqlite-vec + FTS5 (visível, debuggável) | cache local (não exposto no README) |
| Tooling acessório | `eval`, `cost`, `watch`, `glossary`, `pipeline {show,validate}`, `adapters list`, `hook {install,uninstall,status}` | `savings` (com `--verbose`), `init` (gera sub-agent para Claude Code) |

semble vence em **time-to-first-query, footprint, latência e benchmarks publicados**.
stropha vence em **profundidade estrutural, observabilidade, extensibilidade e correção em monorepos**.

---

## 2. Onde semble bate stropha — desvantagens documentadas

### 2.1 Latência e footprint do embedder

- stropha embute via `mxbai-embed-large-v1` (1024-d, ~0.3 B params, F16 nas safetensors; rodado via fastembed/ONNX em CPU). Cada query roda um forward pass; cada chunk re-indexado também. Em CI/laptop sem GPU, isso domina o tempo de indexação.
- O extra `[mlx-embed]` (Apple Silicon only) usa `mlx-embedding-models` e roda o mesmo modelo na GPU integrada, prometendo 3–5× sobre fastembed ONNX. **Mas ainda é forward pass de transformer** — não é embedding estático.
- semble usa **Model2Vec / potion-code-16M** (256-d, ~16 M params, vocab ~62.500): a "inferência" é um *table lookup* + pooling — sem multiplicação de matrizes em runtime. Resultado (segundo README): indexação 250 ms, query 1,5 ms.
- O stropha **não tem adapter de embeddings estáticos** (Model2Vec, GloVe, fastText). Os 4 adapters de embedder existentes (`local`, `voyage`, `bge-m3`, `mlx`) são todos transformer-based. Não existe a opção "rápida e suficientemente boa".

**Impacto prático:** quando um agente chama `search_code` várias vezes seguidas,
o stropha paga um forward-pass por chamada na dense lane; o cache semântico
existente ainda não está integrado ao caminho de retrieval. Semble paga cerca
de 1 ms por construção no ambiente medido por sua própria avaliação.

**Footprint de índice:** mxbai 1024-d × 4 bytes/dim = 4 KB por chunk só no vetor denso. Em um repo de 50k chunks: ~195 MB só de embeddings densos. Com potion 256-d × 4 bytes: ~49 MB. Stropha não tem quantização binária / int8 — guarda float32 sempre.

### 2.2 Reranking sem sinais code-aware

stropha funde três streams por padrão via RRF (dense, BM25 e símbolo), com
GraphVec e Model2Vec opcionais. Também pode executar rerankers opt-in, incluindo
heurísticas code-aware e cross-encoder. A composição efetiva deve ser consultada
com `stropha pipeline show`, não inferida de um total estático.

semble aplica heurísticas explícitas após o RRF (README enumera 5 sinais):

| Heurística (semble) | O que faz | stropha tem? |
|---|---|---|
| Adaptive weighting | Detecta query "tipo símbolo" (CamelCase, snake_case, `module.foo`) e aumenta peso lexical; query natural mantém balanceado | Parcial: o stream `like-tokens` existe e contribui via RRF, mas o peso de cada stream no `rrf_fuse` é fixo (não muda com a forma da query) |
| Definition boost | Chunks que **definem** o símbolo da query (kind=class/function/method com `symbol` matching) sobem | **não** |
| Identifier stemming | `getUserById` ↔ `get_user_by_id` ↔ `GetUserByID` (semble README cita `parse config` → `parseConfig`/`ConfigParser`) | **não** (existe split CamelCase no documento FTS via `_fts_text()`, mas não há stemming na query) |
| File coherence | Reordena hits do mesmo arquivo para preservar contexto adjacente | Parcial: `recursive_retrieval` faz auto-merge de chunks adjacentes do mesmo arquivo, mas é off por default e foca em merging, não em reordenação |
| Noise penalty | Down-weight para `test_*`, `*.spec.*`, `compat/`, `legacy/`, exemplos, `.d.ts` | **não**: existe filtro `exclude_tests` opt-in no `search_code`, mas é binário (exclui), não atenua |

**Impacto:** em queries de símbolo, o stropha frequentemente devolve o teste antes da definição quando o teste tem mais menções textuais do nome. semble corrige isso por construção.

### 2.3 Friction de install e onboarding

| Passo | stropha | semble |
|---|---|---|
| Install | `git clone` + `uv sync` (fastembed + onnxruntime + tree-sitter + voyageai + …). Modelo ONNX (~ 1 GB para mxbai) baixado on-demand pelo fastembed na primeira execução. | `pip install semble` |
| MCP wire-up | configurar o endpoint HTTP compartilhado; `STROPHA_TARGET_REPO` seleciona o alvo e `secrets.voyage_api_key` no YAML é opcional | `claude mcp add semble -s user -- uvx --from "semble[mcp]" semble` (uma linha) |
| Primeiro index | `uv run stropha index` (depende do repo + enricher escolhido) | implícito ao chamar `search` (auto-detecta path; também aceita URL git) |
| Graph stack | graphify (binário Rust separado) + hook install | n/a |

O entry-point `stropha-mcp = "stropha.server:main"` **existe** em `pyproject.toml`, ou seja, a parte de empacotamento já está pronta — o que falta é a **publicação em PyPI**. Enquanto não acontecer, o equivalente do "one-liner" do semble é `uvx --from "git+https://github.com/JonatasFischer/stropha" stropha-mcp`, que ainda assim:

1. Baixa o repo inteiro (incluindo testes, docs, fixtures) em vez do wheel mínimo.
2. Não tem extra `[mcp]` separado — instala todas as deps mesmo quando o usuário só quer rodar o servidor.

Isso bloqueia o caso de uso "agente novo, repo novo, quero search agora".

### 2.4 Falta de benchmark público comparável

- stropha publica Recall@10 e MRR num golden set local de 43 queries sobre 1 repo (mimoria).
- semble publica **NDCG@10 = 0.854** em ~1.250 queries sobre 63 repos, com comparação direta vs. CodeRankEmbed (transformer de 137M).

Sem NDCG nem cobertura cross-repo no harness, stropha não consegue afirmar paridade com baselines da literatura — só "passou no nosso teste".

### 2.5 Tracking de economia de tokens

semble tem o conceito de "savings" como cidadão de primeira classe: cada chamada de `search` registra quantos tokens **não foram lidos** vs. o grep+read equivalente, persistido em `~/.semble/savings.jsonl`. Claim: 98% fewer tokens, 94% recall a 2k tokens.

stropha tem `stropha cost` mas é orientado a **custo de indexação** (per-adapter / per-repo), não a **valor entregue ao agente**. O ângulo "quanto context window economizei" é o que vende a ferramenta para usuários de LLM, e stropha não o mede.

### 2.6 Indexação ad-hoc de URLs git

semble: `semble search "auth flow" https://github.com/user/repo` clona, indexa e responde numa só chamada.
stropha: precisa de `stropha index --repo <path>` separado, e o path tem de ser local.

Esse atrito mata o caso "Claude, busque no repo X que eu nem clonei ainda".

### 2.7 Quantidade de configuração exposta

`grep -oE 'STROPHA_[A-Z_]+' src/stropha/` retorna **42 env vars distintas** referenciadas no código, das quais ao menos 9 afetam diretamente qualidade de retrieval (HyDE, code-tuned HyDE, recursive merge, multi-query, decomposition, rewrite, cache semântico, FTS augment, query router LLM). semble tem **zero env vars** documentados no README — defaults funcionam.

Mais flexibilidade ≠ melhor produto. Cada flag é uma combinação não testada no
benchmark. This historical mismatch is why current `CLAUDE.md` **Key
architecture** points to runtime introspection instead of copying the full
configuration surface.

### 2.8 Modelo padrão pesado em disco / memória

`mxbai-embed-large-v1` (1024-d float32) ⇒ 4 KB por chunk só no vetor denso. Para um repo com 50k chunks, ~195 MB só de embeddings. `potion-code-16M` (256-d) ocuparia ~49 MB para o mesmo número de chunks — 4× menos. Stropha **não tem** suporte a quantização binária/int8 nem a Matryoshka truncation (mesmo o mxbai suportando truncate_dim=512 nativamente).

### 2.9 Acoplamento com graphify

6 das 12 ferramentas MCP do stropha **só funcionam se o usuário rodar graphify**. Isso é uma dependência rust externa, não-publicada-em-crates.io na época deste documento, com overhead próprio de indexação. Quando o grafo está vazio, as ferramentas retornam `{graph_loaded: false}` (current named invariant **Graph-gated tools never return silent empty**) mas a *promessa* de "trace_feature, find_callers, etc." só se cumpre depois de mais setup.

semble não tem essa promessa, mas também não cobra o setup.

---

## 3. Onde stropha continua melhor

Não para inverter a comparação, mas para não jogar fora o diferencial ao tentar "ser mais semble":

1. **Estrutura semântica via grafo** — `find_callers`, `trace_feature`, `get_community`, `find_rationale` não têm equivalente no semble.
2. **Multi-repo nativo** — schema v7 + manifest YAML lidam com monorepos vendorizados.
3. **Adapter framework + drift detection** — trocar enricher / embedder / chunker dispara re-embed automático dos chunks afetados (sem rebuild manual).
4. **Hook incremental git-aware** — re-indexa só os arquivos do diff, com rename-resilient chunk_id.
5. **Eval harness com Recall@K + MRR + tags** — exit code não-zero em CI se Recall < 0.85.
6. **Observabilidade** — `pipeline show/validate`, `adapters list`, `cost`, `stats`, logs estruturados via structlog.
7. **Query intelligence avançada** — HyDE, decomposition, multi-query, rewrite, cache semântico, query router (LLM-fallback).
8. **Glossary** — `stropha glossary` injeta definições de domínio no contexto de embeddings.

---

## 4. Plano de melhoria — pegar o que semble faz melhor sem destruir o que stropha já entrega

Ordenado por **ROI vs. esforço**. Cada item respeita `CLAUDE.md` **Invariants
(do NOT break)** (drift detection, forward-only migrations, graph lifecycle,
explicit graph gating, non-blocking hook, local-only default, SQLite FTS5 rules).

### 4.1 Quick wins (1–3 dias cada)

#### A. Embedder estático Model2Vec / potion-code-16M
**Arquivo:** `src/stropha/adapters/embedder/model2vec.py` (novo)
**Plano:**
- Novo adapter `@register_adapter(stage="embedder", name="model2vec")` carregando `model2vec` via `pip install model2vec` (extra opcional `[fast]`).
- Default model: `minishlab/potion-code-16M` (256-d, ~16 M params; vocab ~62.500; max-seq 1 M tokens — confirmado no model card).
- `adapter_id = f"model2vec:{model_name}:{dim}"` — protected by the named
  **Drift detection** invariant.
- `EmbedderStage.embed_chunks()` chama `model.encode(texts)` (lookup + pooling, sem forward pass de transformer).
- Expose via adapter introspection and `CLAUDE.md` **Key architecture** rather
  than adding another hard-coded total.
- **Resultado esperado (a confirmar empiricamente via golden set):** dense lane substancialmente mais rápida em CPU; possível regressão de Recall@10 que precisa ser medida — semble reporta NDCG@10 = 0.854, mas isso é com **todas** as heurísticas code-aware (item B), não só o embedder.

#### B. Code-aware reranker como stage opcional
**Arquivo:** `src/stropha/adapters/reranker/code_aware.py` (novo)
**Plano:**
- Aproveita o stage `reranker` que já existe (`noop`, `cross-encoder`).
- Aplica em ordem, antes do top-K final:
  1. **Definition boost**: se `hit.chunk.symbol` e `hit.chunk.kind in {class, function, method, interface}` e query tokens ⊆ symbol tokens → `score *= 1.5`.
  2. **Identifier stem match**: stemizar `getUserById` → `[get, user, by, id]`, comparar com query stems. Boost proporcional ao overlap.
  3. **Noise penalty**: `path` matches `test_*|*_test.*|*.spec.*|*.test.*|/tests/|/legacy/|/deprecated/` → `score *= 0.7` (a menos que a query contenha "test" como token).
  4. **File coherence**: agrupa hits por arquivo, reordena dentro do grupo por proximidade de linha ao "best hit".
- `adapter_id = "code-aware:v1:flags=<hash>"` — flags configuráveis via YAML, mudança força re-rank (não re-embed).
- **Resultado esperado:** +10–15 pp em queries de símbolo no golden set.

#### C. Adaptive query weighting no hybrid-rrf
**Arquivo:** `src/stropha/adapters/retrieval/hybrid_rrf.py` (edit)
**Plano:**
- Heurística pura no `search()`: classificar query como `symbol_like` se ≥80% dos tokens batem `[A-Z][a-z]+([A-Z][a-z]*)+` (CamelCase) ou `[a-z]+(_[a-z]+)+` (snake_case) ou contém `.` qualificado.
- Quando `symbol_like`: pesos RRF dos streams ficam `{vec: 0.5, fts: 1.0, like: 1.5, graph_vec: 0.5}`.
- Quando natural-language: pesos `{1.0, 1.0, 0.5, 1.0}`.
- Pesos entram no RRF como multiplicador de `1/(k+rank)`.
- `adapter_id` digere os pesos → drift detection funciona.

#### D. Tracking de "tokens saved"
**Arquivo:** `src/stropha/retrieval/search.py` (edit) + `src/stropha/cost.py` (edit)
**Plano:**
- Para cada query, calcular `baseline_tokens = file_size_estimate(repo)` (cache simples por repo).
- Logar via structlog: `event="search.served", query, returned_tokens, baseline_tokens, savings_pct`.
- `stropha cost --savings` agrega no JSONL e mostra dashboard "tokens saved per query / per session".
- Sem mudança de schema, sem invariante quebrado.

### 4.2 Mid-term (1 semana cada)

#### E. UX de install single-command — `stropha init`
**Arquivos:** `src/stropha/cli.py` (novo subcommand) + `src/stropha/tools/init.py` (novo)
**Plano:**
- `uv run stropha init [--target <path>] [--mcp-client claude-code|cursor|opencode]`
- Faz, em ordem:
  1. Cria `.stropha/` no target, com `.strophaignore` default.
  2. Detecta cliente MCP no `$HOME` e injeta entry em `.mcp.json` / `~/.cursor/mcp.json` / `~/.opencode/config.json`.
  3. Roda primeiro `index` com defaults.
  4. Imprime próximos passos (hook install opcional, graphify opcional).
- Não substitui o fluxo `uvx`; complementa.

#### F. Distribuição via PyPI + `uvx stropha-mcp`
**Plano:**
- O entry-point já existe em `pyproject.toml` (`stropha-mcp = "stropha.server:main"`); o que falta é publicação no PyPI.
- Mover `fastembed`, `tree-sitter-language-pack`, `voyageai` para extras opcionais (`stropha[fastembed]`, `stropha[chunkers-extra]`, `stropha[voyage]`) para enxugar o wheel default.
- Adicionar extra `[mcp]` com apenas o mínimo para rodar o servidor (mcp + pydantic + sqlite-vec + numpy + structlog + Model2Vec).
- Usuário final: `claude mcp add stropha -s user -- uvx --from "stropha[mcp]" stropha-mcp`.
- Modelo Model2Vec (item A) vira o **default novo** justamente porque cabe em wheel leve.

#### G. Aceitar git URL em `stropha search`
**Plano:**
- `stropha search "auth" https://github.com/x/y` clona em `~/.cache/stropha/repos/<sha>/`, indexa se faltar, responde. Cache invalida por TTL (24h) ou por `--refresh`.
- MCP tool `search_code` ganha param opcional `repo_url`.

#### H. NDCG@10 + golden set expandido
**Arquivos:** `src/stropha/eval/harness.py` (edit) + `tests/eval/golden.jsonl` (expand)
**Plano:**
- Implementar NDCG@10 em `EvalReport` (Recall@K + MRR + NDCG@K).
- Expandir golden set para ≥200 queries balanceadas (symbol / conceitual / multi-hop / natural-language), cobrindo ≥3 repos open-source distintos.
- `stropha eval --benchmark-suite` roda contra todos os repos em paralelo.

### 4.3 Long-term (>1 semana)

#### I. Modelos maiores opt-in para HyDE
At the historical snapshot this was pending in the project history. Reassess
against the live registry before prioritizing.

#### J. Late chunking + ColBERT
Historically pending. Reassess against the live registry; the effort comparison
is preserved from the original analysis.

#### K. LanceDB storage adapter
Historically pending. PostgreSQL storage has since shipped; this LanceDB option
remains only a comparison idea.

---

## 5. Roadmap proposto (3 sprints)

| Sprint | Itens | Métrica de sucesso |
|---|---|---|
| 1 | A (Model2Vec) + C (adaptive weighting) + D (tokens-saved) | Indexação <2s para mimoria; query <20 ms; Recall@10 ≥ 0.83 |
| 2 | B (code-aware reranker) + H (NDCG + golden expandido) | NDCG@10 ≥ 0.80 em golden expandido; +10 pp em queries de símbolo |
| 3 | E (`stropha init`) + F (PyPI + uvx) + G (git URL) | `claude mcp add stropha …` em uma linha; primeira query <30s do zero |

Itens I/J/K ficam fora do roadmap inicial — entram quando o benchmark estabilizar.

---

## 6. Invariantes preservados

Todas as mudanças propostas respeitam os invariantes nomeados de `CLAUDE.md`:

| Invariante | Como é preservado |
|---|---|
| Drift detection | Cada novo adapter (Model2Vec, code-aware reranker) digere config no `adapter_id`. |
| Forward-only schema migrations | Itens A–H não tocam schema. Se eventualmente precisar, cada storage adapter recebe migration idempotente equivalente. |
| `Pipeline.run()` owns graphify lifecycle | Reranker code-aware roda dentro do `RetrievalStage.search()`, não no pipeline de indexação. Order intocada. |
| Graph-gated tools never return silent empty | Nenhuma mudança nas ferramentas de grafo. |
| Hook <= 100 ms before fork | Mudanças no fluxo de query/index, não no hook. |
| Local-only by default | Model2Vec e code-aware reranker são 100% locais. PyPI install não exige rede em runtime. |
| SQLite FTS5 writes are explicit | Não alterado; a regra é específica do adapter SQLite. |
| Search-time toggles don't require re-index | Adaptive weighting + tokens-saved são puramente de query — sem invalidação de cache. |

---

## 7. Conclusão

semble entrega uma promessa **mais estreita** com qualidade publicada e UX excepcional. stropha entrega uma promessa **mais larga** (grafo + multi-repo + adapter framework) ao custo de friction e latência.

As cinco maiores desvantagens do stropha são, em ordem:

1. **Embedder pesado por padrão** (sem opção estática Model2Vec).
2. **Reranker sem sinais code-aware** (definition boost, identifier stem, noise penalty).
3. **Friction de install** (sem `uvx stropha-mcp`, sem `stropha init`).
4. **Sem indexação ad-hoc por URL git**.
5. **Benchmarks limitados a um repo** (sem NDCG@10, sem cross-repo).

Os itens A, B, C, D, E, F, G, H acima endereçam todos esses pontos sem quebrar invariantes nem desistir das vantagens estruturais do stropha. O roadmap de 3 sprints é o caminho mais curto para **paridade de UX e latência com semble, mantendo a profundidade estrutural que semble não tem**.

---

## 8. Errata da revisão 1 (auto-revisão)

A primeira versão deste documento tinha erros factuais identificados em re-leitura contra o código. Listados aqui por transparência:

| # | Onde | Erro v1 | Correção verificada |
|---|---|---|---|
| 1 | §1 tabela / §2 | "3 embedders" no stropha | **4 embedders**: `local`, `voyage`, `bge-m3`, `mlx` (confirmado em `grep '@register_adapter.*embedder' src/stropha/adapters/embedder/`) |
| 2 | menção implícita | "5 enrichers" | **6 enrichers**: `noop`, `hierarchical`, `graph-aware`, `ollama`, `mlx`, `contextual` |
| 3 | menção implícita | "3 walkers" | **4 walkers**: `git-ls-files`, `filesystem`, `nested-git`, `git-diff` |
| 4 | §2.1 | "mxbai-embed-large-v1 ~335M params" | **~0.3 B params** (model card oficial; 300 M, F16 safetensors) |
| 5 | §2.3 | "stropha não tem equivalente a `uvx stropha-mcp`" | O **script existe** em `pyproject.toml`; o que falta é publicação em PyPI. `uvx --from <git-url>` já funciona, com caveats. |
| 6 | §2.3 | "fastembed pesa ~200 MB de wheel" | **Inventado** — não verifiquei. Removido. |
| 7 | §2.7 | "38 env vars" | **42 env vars distintas** referenciadas no código (`grep -oE 'STROPHA_[A-Z_]+'`) |
| 8 | §2.8 | "vetores muito menores (256-d typical)" para potion | Confirmado 256-d (model card), mas era chute na v1 — agora citado com fonte. |
| 9 | §2 (omissão) | extra `[mlx-embed]` não mencionado | Adicionado: 3–5× sobre fastembed via mlx-embedding-models (Apple Silicon GPU). Continua sendo transformer-based, não estático. |
| 10 | §2 (omissão) | reranker default não nomeado | Cross-encoder `BAAI/bge-reranker-base` via fastembed `TextCrossEncoder` (confirmado em `adapters/retrieval/reranker/cross_encoder.py`). |
| 11 | §2.2 | "4 sinais code-aware" no semble | README enumera **5 sinais** (adaptive weighting + definition boost + identifier stem + file coherence + noise penalty). |

O que continua valendo:
- Direção geral das 5 maiores desvantagens (§7) — embedder pesado, falta de reranking code-aware, friction de install, ausência de indexação por URL git, benchmark limitado.
- Roadmap em 3 sprints (§5) — itens A–H, ROI vs. esforço, invariantes preservados (§6).
- Comparação de filosofia (§1, §3) — stropha é deep + estrutural, semble é raso + rápido.

**Métodos de verificação usados na revisão 2:**
- `grep register_adapter src/stropha/adapters/` para contagem de adapters por stage
- `grep -oE 'STROPHA_[A-Z_]+' src/stropha/ -r | sort -u` para env vars
- `grep '@mcp.tool' src/stropha/server.py` para tools MCP (confirmado 12)
- Leitura de `pyproject.toml` para entry-points e extras
- Model cards no HuggingFace para mxbai e potion-code-16M
- README de semble (2 passes) para reranking signals e CLI commands
