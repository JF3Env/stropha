# stropha vs. semble — comparação detalhada

**Status:** Análise técnica · 2026-05-18
**Escopo:** stropha @ HEAD vs. [MinishLab/semble](https://github.com/MinishLab/semble) (`main`, leitura via README + página do repositório)
**Objetivo:** mapear desvantagens do stropha em relação ao semble e propor melhorias acionáveis sem sacrificar invariantes do stropha (§3 do CLAUDE.md).

---

## 1. Sumário executivo

stropha e semble compartilham o mesmo *fit*: code-search local exposto a agentes via MCP, com chunking por tree-sitter e fusão de retrieval denso + esparso. Mas as duas ferramentas otimizam para metas diferentes:

| Dimensão | stropha | semble |
|---|---|---|
| Filosofia | RAG **completo + estrutural**: 12 ferramentas MCP, grafo, comunidades, rationale, trace_feature | Code-search **mínimo + rápido**: 2 ferramentas MCP (`search` + `find_related`) |
| Embedder padrão | `mxbai-embed-large-v1` 1024-d (fastembed/ONNX) | `potion-code-16M` (Model2Vec, **embeddings estáticos**) |
| Inferência em query | sim (forward pass ONNX no modelo denso) | **não** — lookup direto numa tabela de embeddings estáticos |
| Latência de indexação típica | segundos a minutos (depende do enricher) | **~250 ms** para um repo médio |
| Latência de query típica | dezenas a centenas de ms | **~1,5 ms** |
| Install | `uv sync` + clone do repo + download de pesos fastembed + (opcional) graphify + hook install | `pip install semble` (ou `uvx --from "semble[mcp]" semble`) |
| Setup MCP | edição manual de `.mcp.json`, env vars, `STROPHA_TARGET_REPO` | `claude mcp add semble -s user -- uvx --from "semble[mcp]" semble` |
| Reranking | RRF (k=60) + opcional cross-encoder BGE | RRF + **heurísticas code-aware** (definition boost, identifier stem, file coherence, noise penalty) |
| Quality benchmark | golden set local de 43 queries (mimoria), Recall@K + MRR | NDCG@10 = 0.854 em ~1.250 queries / 63 repos (paridade com CodeRankEmbed 137M) |
| Multi-repo / monorepo | nativo (schema v7, manifest YAML) | indexa por repo ou git URL on-the-fly |
| Graph traversal | sim (graphify mirror: callers, related, community, rationale, trace_feature) | não — apenas `find_related` por similaridade |
| Aware of git history | hook v=4 incremental, diff-walker, rename-resilient | file-watcher por mtime |
| Persistência de índice | SQLite + sqlite-vec + FTS5 | cache local (não exposto), in-memory após load |
| Tooling acessório | `eval`, `cost`, `watch`, `glossary`, `pipeline {show,validate}`, `adapters list` | `savings` |

semble vence em **time-to-first-query, footprint, latência e benchmarks publicados**.
stropha vence em **profundidade estrutural, observabilidade, extensibilidade e correção em monorepos**.

---

## 2. Onde semble bate stropha — desvantagens documentadas

### 2.1 Latência e footprint do embedder

- stropha embute via `mxbai-embed-large-v1` (1024-d, ~335M params via ONNX). Cada query roda um forward pass; cada chunk re-indexado também. Em CI/laptop sem GPU, isso domina o tempo de indexação.
- semble usa **Model2Vec / potion-code-16M**: a "inferência" é um *table lookup* + pooling — sem multiplicação de matrizes em runtime. Resultado: indexação 250 ms, query 1,5 ms.
- O stropha **não tem adapter de embeddings estáticos** (Model2Vec, GloVe, fastText). Mesmo o `bge-m3` é fastembed/ONNX. Não existe escolha "rápida e suficientemente boa".

**Impacto prático:** quando um agente do Claude Code chama `search_code` várias vezes seguidas (típico em loops de planning), o stropha paga ~50–200 ms por chamada só pela dense lane; semble paga ~1 ms.

### 2.2 Reranking sem sinais code-aware

stropha funde 4 streams via RRF (k=60) e opcionalmente roda um cross-encoder. **Nenhum dos streams nem o reranker conhece código** — todos tratam o texto como linguagem natural.

semble aplica heurísticas explícitas após o RRF:

| Heurística | O que faz | stropha tem? |
|---|---|---|
| Adaptive weighting | Detecta query "tipo símbolo" (CamelCase, snake_case, `module.foo`) e aumenta peso lexical; query natural mantém balanceado | Parcial: `like-tokens` stream existe, mas o peso é fixo |
| Definition boost | Chunks que **definem** o símbolo da query (kind=class/function/method com `symbol` matching) sobem | **não** |
| Identifier stemming | `getUserById` ↔ `get_user_by_id` ↔ `GetUserByID` | **não** (apenas split CamelCase no FTS) |
| File coherence | Reordena hits do mesmo arquivo para preservar contexto adjacente | Parcial: recursive merge faz isso, mas só com toggle `STROPHA_RECURSIVE_RETRIEVAL=1` |
| Noise penalty | Down-weight para `test_*`, `*.spec.*`, código gerado, stubs | **não** (existe filtro `exclude_tests` opt-in, não penalidade) |

**Impacto:** em queries de símbolo ("FSRSCalculator") o stropha frequentemente devolve o teste antes da definição. semble corrige isso por construção.

### 2.3 Friction de install e onboarding

| Passo | stropha | semble |
|---|---|---|
| Install | `git clone` + `uv sync` (>30 deps, fastembed pesa ~200 MB de wheel + modelo ONNX baixado on-demand) | `pip install semble` (alguns MB) |
| MCP wire-up | criar `.mcp.json`, ajustar `STROPHA_TARGET_REPO`, opcionalmente `VOYAGE_API_KEY` | `claude mcp add semble -s user -- uvx --from "semble[mcp]" semble` (uma linha) |
| Primeiro index | `uv run stropha index` (segundos a minutos) | implícito ao chamar `search` (auto-detecta path) |
| Graph stack | `cargo install graphify` + `graphify .` + hook install | n/a |

stropha **não tem** equivalente a `uvx stropha-mcp` — o servidor MCP só roda dentro do projeto clonado. Isso impede o caso de uso "agente novo, repo novo, quero search agora".

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

stropha tem **38 env vars** documentados (§2.5 do CLAUDE.md) e 9 toggles que afetam qualidade (HyDE, recursive, multi-query, decomposition, rewrite, cache, FTS augment, router LLM, etc.). semble tem **zero env vars** documentados no README — defaults funcionam.

Mais flexibilidade ≠ melhor produto. Cada flag é uma combinação não testada no benchmark.

### 2.8 Modelo padrão pesado em disco / memória

`mxbai-embed-large-v1` (1024-d float32) ⇒ 4 KB por chunk só no vetor denso. Para um repo com 50k chunks, ~200 MB só de embeddings. semble com `potion-code-16M` armazena vetores muito menores (256-d typical), com fração do footprint.

### 2.9 Acoplamento com graphify

6 das 12 ferramentas MCP do stropha **só funcionam se o usuário rodar graphify**. Isso é uma dependência rust externa, não-publicada-em-crates.io na época deste documento, com overhead próprio de indexação. Quando o grafo está vazio, as ferramentas retornam `{graph_loaded: false}` (cumpre o invariante §3.4) mas a *promessa* de "trace_feature, find_callers, etc." só se cumpre depois de mais setup.

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

Ordenado por **ROI vs. esforço**. Cada item respeita os invariantes §3 do `CLAUDE.md` (drift detection, migrations forward-only, `Pipeline.run()` como composition root, graph-gated tools nunca silenciosos, hook ≤100 ms, local-first, FTS5 contentless-external).

### 4.1 Quick wins (1–3 dias cada)

#### A. Embedder estático Model2Vec / potion-code-16M
**Arquivo:** `src/stropha/adapters/embedder/model2vec.py` (novo)
**Plano:**
- Novo adapter `@register_adapter(stage="embedder", name="model2vec")` carregando `model2vec` via `pip install model2vec` (extra opcional `[fast]`).
- Default model: `minishlab/potion-code-16M`.
- `adapter_id = f"model2vec:{model_name}:{dim}"` — drift detection liga sozinha.
- `EmbedderStage.embed_chunks()` chama `model.encode(texts)` (lookup puro).
- Documentar em §2.3 do `CLAUDE.md`.
- **Resultado esperado:** indexação 10–100× mais rápida na lane densa, query 50× mais rápida. Recall@10 esperado: -2 a -5 pp (a confirmar via golden set).

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
- Publicar `stropha` no PyPI (entrada `stropha-mcp` no `[project.scripts]`).
- Usuário: `claude mcp add stropha -s user -- uvx --from "stropha[mcp]" stropha-mcp`.
- fastembed pode permanecer extra opcional para evitar wheel pesado por default; entrar via `stropha[heavy]`.
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
Já está pendente no §6 do CLAUDE.md (`deepseek-coder-v2`, `codestral`). Manter prioridade média.

#### J. Late chunking + ColBERT
Pendente no §6. Maior esforço, ganho marginal sobre as melhorias de retrieval acima.

#### K. LanceDB storage adapter
Pendente no §6. Alternativa embarcada ao sqlite-vec, mesmo `StorageStage`. Útil quando o repo passa de ~500k chunks.

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

Todas as mudanças propostas respeitam §3 do `CLAUDE.md`:

| Invariante | Como é preservado |
|---|---|
| §3.1 Drift detection | Cada novo adapter (Model2Vec, code-aware reranker) digere config no `adapter_id`. |
| §3.2 Schema migrations forward-only | Itens A–H não tocam schema. Se eventualmente precisar (e.g. armazenar `definition_score` por chunk), bump `SCHEMA_VERSION` + `_add_column_if_missing`. |
| §3.3 `Pipeline.run()` order | Reranker code-aware roda dentro do `RetrievalStage.search()`, não no pipeline de indexação. Order intocada. |
| §3.4 Graph-gated tools never silent | Nenhuma mudança nas ferramentas de grafo. |
| §3.5 Hook ≤ 100 ms | Mudanças no fluxo de query/index, não no hook. |
| §3.6 Local-only por default | Model2Vec e code-aware reranker são 100% locais. PyPI install não exige rede em runtime. |
| §3.7 FTS5 contentless-external | Não alterado. |
| §3.8 Search-time toggles sem re-index | Adaptive weighting + tokens-saved são puramente de query — sem invalidação de cache. |

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
