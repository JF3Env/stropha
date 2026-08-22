# RAG System para o Codebase Mimoria

Documento de design técnico para um sistema de Retrieval-Augmented Generation (RAG) cuja função é servir como fonte de verdade sobre o codebase Mimoria, exposto ao Claude Code (e outros clientes LLM) via Model Context Protocol (MCP).

> **Status**: este documento é o **mapa do espaço de soluções v1.0** — captura
> escolhas de design, alternativas e milestones históricos. Para o estado vivo,
> leia `CLAUDE.md` **Key architecture** e interrogue
> `stropha pipeline show` / `stropha adapters list`; quando houver divergência,
> essas fontes vencem.
>
> **Implementado (cumulativo, conforme §16):**
> - Phase 0 (spike) ✓ — walker / chunker dummy / sqlite-vec / CLI.
> - Phase 1 (MVP MCP) ✓ — tree-sitter chunking, hybrid retrieval configurável,
>   MCP server `stropha_rag` (the original stdio milestone was later superseded
>   by the shared HTTP daemon).
> - Pipeline-adapters Phase 1–4 (spec `stropha-pipeline-adapters.md`) ✓ —
>   framework Stage/registry; use registry introspection for the current inventory.
> - Graphify integration Fase 1.5a/b/c/e/f (spec
>   `stropha-graphify-integration.md`) ✓ — storage-adapter graph mirror, graph
>   tools, detached hook, lexical augmentation and graph-vector lane.
> - Phase 2 parcial ✓ — `find_callers` / `find_tests_for`, Recall@K + MRR eval (RAGAS deferred).
> - Phase 3 parcial ✓ — `trace_feature`, HyDE (via Ollama local), recursive retrieval, soft-index `stropha watch`, cost dashboard.
> - Phase 4 parcial ✓ — multi-repo `--manifest`, walker `nested-git`, embedder `bge-m3`, hook cross-repo (`--project-dir`).
>
> **Historical deferrals:** this list records the v1 planning decision, not the
> current adapter inventory. PostgreSQL storage and contextual enrichment have
> since shipped; cloud providers remain optional.
>
> **Current schema:** v14 on SQLite and PostgreSQL. Each adapter owns an
> idempotent, forward-only migration path. Run `stropha stats` for live metadata;
> the v2-v7 notes later in this document remain delivery history.

---

## 1. Objetivos e não-objetivos

### 1.1 Objetivos

1. **Recuperação semântica de código**: dado uma query em linguagem natural ("onde está a lógica que valida resposta de exercício de Speech?"), retornar trechos relevantes do código com path, linhas e contexto.
2. **Recuperação semântica de artefatos não-código**: feature files Gherkin, YAML de seed, documentação Markdown, comentários de domínio.
3. **Navegação simbólica**: dado um nome de classe/método, retornar definição, callers, callees, tipos relacionados.
4. **Atualização incremental** integrada ao fluxo Git, sem reindexação completa a cada commit.
5. **Exposição via MCP** com contrato estável, latência baixa (<500 ms p95) e respostas que economizam o context window do cliente.
6. **Avaliação contínua** com golden dataset versionado e métricas reproduzíveis.

### 1.2 Não-objetivos (escopo explicitamente fora)

- **Geração de código**: o RAG fornece contexto, não escreve código. O Claude Code (ou outro cliente) é o gerador.
- **Substituir LSP/IDE**: análise estática profunda (refactor cross-file, type-checking) continua sendo do IntelliJ/IDE.
- **Indexar dependências externas** (Quarkus, Vue, etc.) numa primeira fase. Isso é explosão de escopo; entra como Phase 4 opcional.
- **Multi-repositório**: foco no monorepo Mimoria. Multi-repo é generalização posterior.

### 1.3 Critérios de sucesso

| Métrica | Alvo |
|---|---|
| Recall@10 em queries semânticas (golden set) | ≥ 0,85 |
| MRR (Mean Reciprocal Rank) em queries simbólicas | ≥ 0,90 |
| Latência p95 do tool `search_code` | ≤ 500 ms |
| Latência p95 da reindexação incremental por commit | ≤ 5 s |
| Custo mensal de embeddings (após bootstrap) | ≤ US$ 5 |
| Cobertura do índice (arquivos relevantes) | ≥ 95 % |

---

## 2. Arquitetura de alto nível

```
┌──────────────────────────────────────────────────────────────────────┐
│                         CAMADA DE INGESTÃO                           │
│  ┌──────────┐  ┌──────────────┐  ┌───────────┐  ┌────────────────┐  │
│  │ Source   │→ │ Tree-sitter  │→ │ Chunker   │→ │ Enricher (LLM) │  │
│  │ Walker   │  │ Parser (AST) │  │ semântico │  │  + metadata    │  │
│  └──────────┘  └──────────────┘  └───────────┘  └────────────────┘  │
│       │              │                  │                │           │
│       ▼              ▼                  ▼                ▼           │
│  .strophaignore Symbol Graph       Chunk Store      Summary Cache    │
│  + git diff    (call/import)        (SQLite)         (SQLite)        │
└──────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌──────────────────────────────────────────────────────────────────────┐
│                      CAMADA DE EMBEDDING                             │
│  ┌────────────────────┐  ┌────────────────────┐                      │
│  │ Dense embedder     │  │ Sparse embedder    │                      │
│  │ (Voyage code-3)    │  │ (BM25 / SPLADE)    │                      │
│  └────────────────────┘  └────────────────────┘                      │
│            │                       │                                 │
│            ▼                       ▼                                 │
│   Vector Index (HNSW)      Inverted Index                            │
│   sqlite-vec / LanceDB     Tantivy / SQLite FTS5                     │
└──────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌──────────────────────────────────────────────────────────────────────┐
│                       CAMADA DE RECUPERAÇÃO                          │
│  Query → [Rewriter / HyDE] → [Hybrid Search] → [RRF Fusion]          │
│        → [Reranker (Cohere/Voyage)] → [Context Packer]               │
└──────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌──────────────────────────────────────────────────────────────────────┐
│                          MCP SERVER                                  │
│  Tools: search_code · get_symbol · find_callers · get_file_outline   │
│         find_tests_for · explain_module · trace_feature              │
│  Resources: index_stats · indexed_files                              │
│  Transport: streamable HTTP only (singleton at /mcp)                 │
└──────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
                    Claude Code / IDE / Agent
```

### 2.1 Princípios de design

1. **Local-first**: tudo roda na máquina do dev por padrão. Sem dependência de serviços hospedados para uso individual.
2. **Single binary preferível**: minimizar processos longos. Servidor MCP em Python com `uv` ou em Rust embedando tudo.
3. **Append-only com compactação**: writes idempotentes; reorganização periódica em background.
4. **Observabilidade desde o dia 0**: tracing OpenTelemetry, métricas Prometheus-compatible, logs estruturados (JSON).
5. **Custo previsível**: embeddings só rodam em chunks novos/modificados; reranking apenas quando hybrid search devolve >K candidatos.
6. **Falha graciosa**: se o reranker estiver fora, devolva resultados de hybrid puro com aviso. Se o índice vetorial falhar, caia para BM25.

---

## 3. Pipeline de indexação

A qualidade do RAG é dominada pela **qualidade da indexação**. Embedding bom não compensa chunking ruim.

### 3.1 Source walker e exclusões

**Descoberta de arquivos**:

- Caminhar a partir da raiz do repo (`git ls-files` é mais confiável que `find` — respeita `.gitignore` automaticamente).
- Aplicar `.strophaignore` adicional (formato igual a `.gitignore`) para excluir lockfiles, builds, fixtures binários:
  ```
  **/target/**
  **/node_modules/**
  **/dist/**
  **/.next/**
  **/*.lock
  **/*.min.js
  **/*.snap
  ```
- Detecção de binários: `file --mime-encoding` ou heurística (presença de bytes nulos nos primeiros 8 KB).
- Limite de tamanho por arquivo: 500 KB. Acima disso, indexar apenas o "outline" (símbolos de topo).

### 3.2 Parsing AST com Tree-sitter

**Tree-sitter** (https://tree-sitter.github.io) é o padrão de facto para parsing incremental de múltiplas linguagens, usado por GitHub, Neovim, Zed, Helix e ferramentas como `ast-grep` e `tree-sitter-graph`.

Linguagens necessárias para Mimoria:

| Linguagem | Grammar |
|---|---|
| Java | `tree-sitter-java` |
| Vue (SFC) | `tree-sitter-vue` (com sub-parsing para `<script>` em TS/JS, `<template>` HTML, `<style>` CSS) |
| TypeScript | `tree-sitter-typescript` |
| JavaScript | `tree-sitter-javascript` |
| YAML | `tree-sitter-yaml` |
| Markdown | `tree-sitter-markdown` |
| Gherkin | `tree-sitter-gherkin` |
| SQL | `tree-sitter-sql` |
| Properties / TOML / JSON | grammars correspondentes |

Para cada arquivo:

1. Detectar linguagem por extensão + shebang.
2. Carregar grammar correspondente.
3. Gerar AST.
4. Aplicar **query S-expression** específica por linguagem para extrair nós-alvo (classes, métodos, componentes Vue, scenarios Gherkin).

Exemplo de query Tree-sitter para Java (extrair métodos com docstring):

```scheme
(method_declaration
  name: (identifier) @method.name
  body: (block) @method.body
  (#set! priority "high")) @method.full
```

### 3.3 Estratégia de chunking

**Princípio fundamental**: um chunk = uma unidade semântica completa que faz sentido isoladamente. Não corte no meio de um método.

#### 3.3.1 Chunking semântico hierárquico (recomendado)

Modelo de **três níveis**:

| Nível | Granularidade Java | Granularidade Vue | Granularidade Markdown |
|---|---|---|---|
| **Macro** | Pacote (sumário das classes) | Diretório de feature (sumário) | Documento inteiro (sumário) |
| **Meso** | Classe (assinatura + javadoc + lista de métodos) | Componente SFC completo | Seção H2/H3 |
| **Micro** | Método individual | `<script setup>` + `<template>` separados, ou função composable | Parágrafo / bloco de código |

Cada chunk no nível N referencia seu pai no nível N+1 via `parent_chunk_id`. Isso permite **recursive retrieval** (recuperar pelo método e expandir para a classe se faltar contexto).

**Regras de quebra**:

- **Java**: nunca quebrar dentro de um método. Se um método > 200 linhas (raro, mas acontece), quebrar em "header" (assinatura + primeiras 30 linhas) e "tail" (resto), ambos referenciando a classe.
- **Vue SFC**: três chunks por componente — `script setup`, `template`, `style`. Em componentes pequenos (<150 linhas totais), um único chunk para o SFC inteiro economiza embeddings.
- **Gherkin**: um chunk por `Scenario`/`Scenario Outline`. Um chunk-pai por `Feature` com Background incluído.
- **Markdown**: split por heading. Se uma seção tem code block grande, ele vira sub-chunk próprio (linkado).
- **YAML de seed**: um chunk por entidade de topo (curso, capítulo, lição) — preserva estrutura aninhada.

#### 3.3.2 Janelas de contexto

Cada chunk tem dois "raios" de contexto adicional, **não embeddados** mas retornáveis:

- `neighbor_above`: 5 linhas anteriores (imports relevantes, decoradores).
- `neighbor_below`: 5 linhas posteriores.

Útil para o consumidor reconstruir o trecho com contexto sem segundo round-trip.

#### 3.3.3 Late chunking (Jina, 2024)

Técnica em que se embedda o documento inteiro com um modelo de contexto longo, depois se "fatia" o vetor em sub-vetores correspondentes a cada chunk, preservando contexto cross-chunk. Recomendado para documentação longa onde referências cruzadas importam (ex.: `CLAUDE.md`, `study-flow.md`). Não compensa para código (chunks já são autocontidos).

#### 3.3.4 Anti-padrões a evitar

- **Fixed-size chunking** (cortar a cada 512 tokens): destrói estruturas, péssimo recall em código.
- **Overlap exagerado** (>20 %): infla custo de armazenamento e degrada precisão (mesma informação em múltiplos chunks confunde rerank).
- **Chunks de 1 linha**: ruído, sem contexto suficiente para embedding.
- **Chunks > 8K tokens**: excede contexto da maioria dos modelos de reranking; trunca silenciosamente.

### 3.4 Extração de metadados

Cada chunk persistido tem schema rico:

```jsonc
{
  "chunk_id": "sha256:abc123...",                     // determinístico (path + start + end + content_hash)
  "path": "backend/src/main/java/com/mimoria/service/StudyService.java",
  "language": "java",
  "kind": "method",                                   // file | class | method | component | scenario | section
  "symbol": "StudyService.submitAnswer",
  "parent_symbol": "StudyService",
  "parent_chunk_id": "sha256:def456...",
  "start_line": 145,
  "end_line": 187,
  "byte_range": [4321, 6543],
  "imports": ["com.mimoria.domain.enrollment.Enrollment", ...],
  "annotations": ["@Transactional", "@Mutation"],
  "calls_out": ["EnrollmentRepository.findByUser", "FsrsCalculator.update"],
  "called_by": [...],                                 // preenchido na 2ª passada
  "tests_referencing": ["StudyServiceTest.shouldRecordCorrectAnswer"],
  "git": {
    "last_commit": "2d22f0a",
    "last_modified": "2026-05-13T10:24:11Z",
    "main_authors": ["jonatas"]                       // top-3 por linhas
  },
  "summary": "Submete a resposta de um exercício, atualiza streak e dispara FSRS quando atinge mastery.",
  "summary_model": "claude-haiku-4-5-20251001",
  "summary_generated_at": "2026-05-14T08:00:00Z",
  "content_hash": "sha256:...",                       // p/ detectar mudança real vs git noise
  "embedding_model": "voyage-code-3",
  "embedding_version": 1
}
```

### 3.5 Symbol graph (grafo de código)

Em paralelo aos chunks, construir um **grafo dirigido**:

- **Nós**: símbolos (classes, métodos, funções, componentes).
- **Arestas**:
  - `defined_in` → arquivo.
  - `calls` → método chamado.
  - `imports` → módulo importado.
  - `extends` / `implements` → herança/interface.
  - `tests` → step definition para feature, teste para método.
  - `references` → uso de tipo/constante.

**Por quê?** Habilita queries que embedding sozinho não resolve bem:

- "Quem chama `submitAnswer`?" → travessia direta do grafo, custo O(1) de lookup.
- "Quais testes cobrem `FsrsCalculator`?" → seguir aresta `tests` reversa.
- "Cadeia de prerequisitos de um Subject" → query sobre dados de domínio (não código), mas mesma estrutura.

**Implementação**:

- Extração: combinar tree-sitter (resolve nomes locais) + análise heurística de imports (resolve cross-file). Para precisão máxima, usar **JavaParser** (Java) e **typescript-eslint** (TS/Vue).
- Persistência: tabelas `nodes(id, kind, name, ...)` e `edges(from_id, to_id, kind, weight)` em SQLite. Para grafos > 1M arestas, considerar Kuzu ou DuckDB com extensão grafo.
- **GraphRAG** (Microsoft, 2024) e **LightRAG** (2024) extraem grafos textualmente via LLM — desnecessário para código onde grafos são derivados deterministicamente do AST. Reservar GraphRAG para documentação não-estruturada se necessário.

### 3.6 Enriquecimento via LLM (summary generation)

Para cada chunk de granularidade média/macro, gerar uma `summary` curta (1-2 sentenças) via LLM barato:

- **Modelo**: `claude-haiku-4-5` (custo ~US$ 1/M input tokens). Prompt-cache as instruções de sistema.
- **Conteúdo embeddado**: `summary + "\n\n" + chunk_text`. O summary aproxima o chunk de queries em linguagem natural (que raramente usam vocabulário do código).
- **Regeneração**: apenas quando `content_hash` muda, não a cada reindex.
- **Custo estimado** (Mimoria, ~5K chunks): ~US$ 0,50 inicial, ~US$ 0,02/dia em incremental.

**Cuidado**: se a summary for muito genérica ("este método faz X"), prejudica mais que ajuda. Prompt deve forçar especificidade com exemplos few-shot e proibir frases vazias.

### 3.7 Pipeline de execução (orquestração)

Frameworks state-of-the-art em 2026:

| Framework | Pontos fortes | Quando escolher |
|---|---|---|
| **LlamaIndex** | Maior ecossistema RAG; primitives prontas (NodeParser, IngestionPipeline, transformations). | Prototipagem rápida, equipe Python. |
| **Haystack 2.x** | Pipelines declarativos, ótima observabilidade, deploy serverless. | Time já familiar com Haystack. |
| **txtai** | Single-binary, embedded, baixa cerimônia. | MVP local ultraleve. |
| **Custom (DIY)** | Sem dependência transitiva, controle total. | Quando os frameworks acima atrapalham mais que ajudam. |

Recomendação: começar com **LlamaIndex `IngestionPipeline`** (com cache de transformações), migrar para custom se gargalo aparecer.

---

## 4. Embeddings

### 4.1 Modelos densos — comparação 2026

| Modelo | Dim | Contexto | MTEB Code | Custo (USD/M tokens) | Notas |
|---|---|---|---|---|---|
| **Voyage `voyage-code-3`** | 1024 (Matryoshka 256/512/1024/2048) | 32K | **SOTA em código (CodeSearchNet, CoIR)** | 0,18 | Melhor qualidade absoluta para código. |
| **Voyage `voyage-3-large`** | 1024 (Matryoshka) | 32K | Excelente geral | 0,18 | Bom se misturar muito texto e código. |
| OpenAI `text-embedding-3-large` | 3072 (truncável) | 8K | Bom | 0,13 | Conhecido, ecossistema enorme. |
| Cohere `embed-v4.0` | 1536 (Matryoshka 256/512/1024/1536) | 128K | Muito bom; multimodal (imagens) | 0,12 | Suporta imagens (diagramas). |
| Jina `jina-embeddings-v3` | 1024 | 8K | Bom; suporta late chunking nativo | 0,02 | Excelente custo-benefício; open weights disponíveis. |
| **NV-Embed-v2** (NVIDIA) | 4096 | 32K | Top MTEB geral | self-host | Open weights; precisa GPU. |
| **Stella-en-1.5B-v5** | 1024 (Matryoshka) | 8K | Top open-source | self-host | 1.5B params; roda em CPU lenta, GPU recomendada. |
| **bge-m3** (BAAI) | 1024 | 8K | Bom; multi-funcional (dense + sparse + multi-vec) | self-host | Único modelo que produz dense + sparse + ColBERT no mesmo forward pass. |

**Recomendação**: `voyage-code-3` para code chunks; `voyage-3-large` para markdown/docs. Configurar dimensão Matryoshka em **512** para economizar storage/latência sem perda significativa.

**Por que não OpenAI por padrão?** Em benchmarks específicos de código (CoIR, CodeSearchNet 2024), Voyage code-3 supera `text-embedding-3-large` por 5-12 pontos de NDCG@10. A diferença é maior em queries que misturam intenção com nome de símbolo.

### 4.2 Embeddings esparsos

Embeddings densos têm um ponto cego: queries com **nomes exatos** ("EnrollmentRepository", "submitExerciseAnswer"). BM25 resolve trivialmente. Hybrid search é mandatório.

Opções:

- **BM25 clássico** (Okapi BM25): implementação trivial via Tantivy ou SQLite FTS5 (BM25 ranking). Zero custo, zero dependência externa.
- **SPLADE** (Sparse Lexical and Expansion): aprende expansões neurais. Melhor que BM25 puro, mas mais complexo (requer modelo).
- **BM42** (Qdrant, 2024): híbrido entre BM25 e atenção de transformer. Ainda controverso (artigo recolhido por inflar resultados); evitar até comunidade decidir.

**Recomendação**: BM25 via SQLite FTS5 (single-file, zero infra). SPLADE só se evaluations mostrarem ganho > 5 pontos NDCG.

### 4.3 Embeddings multi-vetoriais

**ColBERT v2** representa cada token como vetor independente; matching é via "MaxSim" — para cada token da query, achar o token mais similar do doc, somar. Muito mais preciso que dense single-vector, mas storage 50-100x maior.

**ColPali** (2024): variante para documentos visuais (PDFs com diagramas). Não relevante para código puro; relevante se indexar diagramas de arquitetura.

Para Mimoria: **não recomendado na fase inicial**. Single-vector + rerank cobre a necessidade.

### 4.4 Quantização

- **Float16 → Int8**: redução 4x de storage, perda <1% NDCG. Suportado nativo em FAISS, sqlite-vec, Qdrant.
- **Binary quantization** (1-bit): redução 32x, perda 5-10% (recuperável com re-rank). Útil para >10M vetores; overkill para Mimoria.

---

## 5. Camada de armazenamento

### 5.1 Vector stores — comparação

| Store | Tipo | Pros | Contras | Indicação |
|---|---|---|---|---|
| **sqlite-vec** | Embedded SQLite extension | Single-file, zero infra, ACID | Sem replicação, single-writer | **Default para Mimoria.** |
| **LanceDB** | Embedded (Lance format) | Versionamento de dados, multi-modal, columnar | Maturidade média | Quando precisar de versioning/time-travel. |
| **DuckDB + VSS** | Embedded analytical | SQL completo, joins poderosos | Vector search ainda jovem | Análises ad-hoc + RAG na mesma base. |
| **Qdrant** | Server (Rust) | Performático, filtros ricos, payload index | Requer processo separado | Multi-tenant ou >1M vetores. |
| **Weaviate** | Server (Go) | Schema rico, modules (rerank embutido), GraphQL | Mais pesado | Quando precisa de schema strong-typed. |
| **Milvus / Zilliz** | Server (Go/C++) | Escala bilhões | Operação complexa | Escala enterprise. |
| **pgvector** | Postgres extension | Reuso da infra Postgres existente | HNSW só recente; menos features que dedicados | Quando Postgres já é stack obrigatória. |
| **Chroma** | Embedded ou server | API simples | Performance medíocre vs alternativas | Prototipagem inicial. |
| **Vespa** | Server (Java) | Tensor expressions, ranking complexo | Curva aprendizado íngreme | Casos avançados de ranking. |
| **Turbopuffer** | Cloud (serverless) | S3-backed, pay-per-query | Vendor lock-in | Cargas esporádicas. |
| **Pinecone** | Cloud managed | Zero ops | Vendor lock-in, custo alto | Quando ops é proibitivo. |

**Current implementation**: both `sqlite-vec` and PostgreSQL/pgvector are
registered storage adapters. `sqlite-vec` remains the zero-infrastructure
default; select the adapter in `~/.stropha/config.yaml`. The other rows are
design alternatives, not shipped inventory.

### 5.2 Algoritmos de indexação aproximada (ANN)

- **HNSW** (Hierarchical Navigable Small World): default da indústria. Recall alto, latência baixa, build moderado.
- **IVF-PQ** (FAISS): compressão extrema, bom para >10M. Build pesado.
- **DiskANN** (Microsoft): vetores em disco SSD. Útil quando RAM limitada.
- **ScaNN** (Google): muito rápido, mas Python-only e pouco mantido.

Para <100k vetores (caso Mimoria), **flat (brute-force)** com SIMD é suficiente e tem recall 100%. HNSW só vira necessário acima de ~500k.

No adapter PostgreSQL, vetores de até 2000 dimensões usam HNSW `vector_l2_ops` diretamente.
Entre 2001 e 4000 dimensões, o vetor continua armazenado como `vector(N)` em precisão completa,
mas o HNSW e a consulta ANN usam a expressão `embedding::halfvec(N)` com
`halfvec_l2_ops`, respeitando o limite dimensional do pgvector. Acima de 4000 dimensões,
a busca densa permanece exata sem HNSW.

### 5.3 Persistência paralela

| Dado | Store | Por quê |
|---|---|---|
| Vetores densos | storage adapter ativo | `vec_chunks`/sqlite-vec ou `chunks.embedding`/pgvector |
| Índice lexical | storage adapter ativo | SQLite `fts_chunks` FTS5 ou PostgreSQL `chunks.fts` tsvector |
| Symbol graph | storage adapter ativo | Tabelas relacionais com migrations equivalentes por adapter |
| Chunks raw + metadata | storage adapter ativo | Fonte de verdade transacional do adapter |
| Cache de enrichment | storage adapter ativo | Reuso por `content_hash` + identidade do adapter |
| Espaço Model2Vec secundário | storage adapter ativo | Lane 256-d independente do vetor primário (schema v14) |

No adapter SQLite, tudo fica em um único arquivo `.db`; no adapter PostgreSQL,
os mesmos contratos vivem no banco configurado. Regras de FTS5 contentless e
arquivo único são específicas de SQLite, não invariantes globais de storage.

O adapter PostgreSQL mantém uma conexão writer dedicada para migrations,
indexação e o session advisory lock. Retrieval tipado (`dense`, BM25, symbol e
secondary vectors) usa um `psycopg_pool.ConnectionPool` separado, bounded por
`read_pool_size` e `read_pool_timeout`. Assim, readers executam em transactions
independentes sem compartilhar rollback/commit state com indexação. Durante um
index lock, reads internos permanecem na conexão writer para preservar
read-your-writes. O `_conn` compatibility shim continua serial até seus
consumidores mistos de read/write migrarem para APIs tipadas.

### 5.4 Versionamento e migração

- Cada storage adapter migra seu próprio schema de forma forward-only e
  idempotente; nunca há `DROP` como parte do startup normal.
- SQLite usa `IF NOT EXISTS` / `_add_column_if_missing`; PostgreSQL fornece o
  caminho equivalente em `adapters/storage/postgres.py`.
- O schema corrente é v14 em ambos. Trocas que alteram output entram no
  `adapter_id`; o pipeline reprocessa apenas os espaços afetados.

---

## 6. Recuperação (retrieval)

### 6.1 Hybrid search (dense + sparse)

#### 6.1.1 Escopo de repositório

Busca semântica, lookup direto de símbolos e consultas diretas ao grafo são
isolados no repositório em trabalho por padrão. Os tools MCP `search_code`,
`smart_search_tool`, `get_symbol`, os seis tools de grafo, os analisadores derivados,
`plan_investigation`, `assemble_context` e `execute_investigation`, o comando
`stropha search` e `POST /api/query` resolvem o target corrente para `repos.id`
em cada fronteira pública. Esse ID é propagado por `RetrievalStage`, lookup de
símbolo, resolução/membros de communities, extensões de grafo, planner e DAG.
Falha de resolução retorna o modelo de resposta nativo vazio, sem consultar o
índice; tools graph-gated preservam `graph_loaded: false` com mensagem acionável.
Resultados cross-repo exigem `include_other_repos=true` no MCP/frontend ou
`--include-other-repos` no CLI. Um `repo_url` explícito sempre permanece limitado
ao clone/cache solicitado, independentemente dessa flag.
Como o daemon HTTP é compartilhado e seu target padrão é process-wide,
`search_code(repo_url=...)` e `assemble_context(repo=...)` aceitam um path local
absoluto indexado ou normalized key para selecionar explicitamente o workspace do caller;
`repo_url` continua aceitando URLs Git para clone/cache remoto.

Internamente, `repo_id=None` significa busca em todos os repositórios. Um ID
concreto é aplicado no storage antes de ranking e `LIMIT`; pins são filtrados
antes de prepend e há filtro final de defesa-em-profundidade. A montagem de
contexto repete essa validação sobre os hits primários antes de packing. GraphVec filtra
`graph_nodes` e hidrata chunks pela mesma identidade de repo. Recursive merge usa
`(repo, rel_path)` como chave para não unir arquivos homônimos de repos distintos.
`repo_id=None` chega aos tools diretos somente após opt-in explícito com
`include_other_repos=true`.

O harness de avaliação recebe a normalized key esperada. Hits sem identidade de
repo ou pertencentes a outro repo nunca satisfazem path/symbol judgments e são
reportados como `wrong_repo_hits`/`wrong_repo_rate`.

Pipeline padrão:

```
query
  ├──► dense embedding (voyage-code-3)
  │       └──► sqlite-vec ANN search (top 50)
  └──► BM25 tokenization
          └──► SQLite FTS5 (top 50)
                ↓
        Reciprocal Rank Fusion (RRF)
                ↓
            top 30 candidatos
                ↓
        Reranker (Voyage rerank-2.5)
                ↓
            top 10 final
```

#### Reciprocal Rank Fusion (RRF)

Fórmula:
```
score(d) = Σ (1 / (k + rank_i(d)))   para cada lista i, com k=60 (default)
```

Vantagens sobre weighted sum:

- Não requer normalização de scores (BM25 e cosine vivem em escalas diferentes).
- Robusto: outliers em uma lista não dominam.
- Sem hyperparameters críticos (k=60 é resiliente).

### 6.2 Reranking (cross-encoder)

Modelos top em 2026:

| Reranker | Latência (100 docs) | Qualidade | Custo |
|---|---|---|---|
| **Voyage `rerank-2.5`** | ~120 ms | SOTA em código | US$ 0,05 / M tokens |
| **Cohere `rerank-3.5`** | ~150 ms | Excelente geral | US$ 2,00 / 1k searches |
| **Jina `jina-reranker-v2`** | ~80 ms | Muito bom | US$ 0,02 / M tokens |
| **bge-reranker-v2-m3** | self-host (200 ms CPU, 30 ms GPU) | Bom | infra |
| **Mixedbread `mxbai-rerank-large-v2`** | self-host | Top open-source | infra |

**Recomendação**: Voyage rerank-2.5 (mesma família dos embeddings, menos atrito de billing).

**Cuidado com latência**: rerank é cross-encoder (encode query+doc juntos por par). Para 50 candidatos, são 50 forward passes. Sempre rerankar em batch via API.

### 6.3 Query understanding

Querys cruas raramente são ótimas. Pré-processar:

#### 6.3.1 Query rewriting

LLM reescreve query em forma "pesquisável":
- "como funciona a logica de mastery" → "transição para REVIEW após streak de 3 acertos consecutivos"
- "onde tem teste pra fsrs" → "FSRS calculator test scenarios"

Custo: ~50 tokens out via Haiku, ~10 ms.

#### 6.3.2 HyDE (Hypothetical Document Embeddings)

Em vez de embeddar a query, **gerar um documento hipotético** que responderia a query, e embeddar isso. Funciona porque o espaço de embedding está alinhado com docs, não com queries.

- "como salvar enrollment?" → LLM gera: `public void saveEnrollment(Enrollment e) { repository.persist(e); ... }` → embedda este código → busca.

Útil para queries vagas. Adiciona latência (LLM call). Skip se a query já contém termos técnicos.

#### 6.3.3 Multi-query expansion

Gerar 3-5 paráfrases da query, buscar cada uma, fundir resultados via RRF. Aumenta recall ~10% mas multiplica custo.

#### 6.3.4 Step-back prompting

Para queries específicas, gerar uma versão "step back" mais geral. Buscar ambas. Útil quando a query específica não tem chunk exato.
- "como o Hibernate mapeia @TenantId?" → step-back: "como funciona multi-tenancy no projeto?"

#### 6.3.5 Query routing

Classificar a query (regra ou LLM) em rotas:

| Tipo de query | Pipeline |
|---|---|
| **Símbolo exato** ("EnrollmentRepository") | Lookup direto no symbol graph; skip embedding. |
| **Conceitual** ("como funciona spaced repetition") | Hybrid + rerank. |
| **Estrutural** ("quem chama X") | Symbol graph traversal. |
| **Cross-cutting** ("onde tem `@Transactional` em mutations") | FTS5 + filtro de metadados. |
| **Tracing de feature** ("trace do scenario X") | Gherkin → step defs → métodos chamados. |

### 6.4 Técnicas avançadas de recuperação

#### 6.4.1 Contextual Retrieval (Anthropic, set/2024)

Antes de embeddar, prepend ao chunk uma frase de contexto gerada por LLM, do tipo:
> "Este trecho pertence à classe `StudyService` (camada de serviço de aplicação) e implementa a lógica de submissão de exercício no fluxo de Phase 1 (acquisition)."

**Resultado em benchmarks Anthropic**: redução de 35% em failed retrievals; com rerank, 67%.

Custo: 1 chamada LLM por chunk na indexação. Cache by content_hash (regenera só em mudança).

#### 6.4.2 Recursive retrieval / hierarchical retrieval

1. Buscar nível "macro" (sumários de classe).
2. Para os top 3, expandir para nível "micro" (métodos da classe).
3. Rerank no nível micro.

Reduz ruído quando codebase tem muitos métodos curtos similares.

#### 6.4.3 Auto-merging retrieval (LlamaIndex)

Se mais de N chunks-filhos do mesmo pai aparecem nos top-K, retornar o pai inteiro em vez dos filhos. Economiza context window, dá visão coerente.

#### 6.4.4 Self-RAG / Corrective RAG (CRAG)

Modelo decide se cada documento recuperado é relevante; se confiança baixa, dispara busca alternativa (web, outro índice). Útil quando o índice tem cobertura desigual. Para codebase fechado, retorno marginal.

#### 6.4.5 Agentic RAG (ReAct sobre tools)

O cliente (Claude Code) NÃO chama um tool monolítico `rag_query`. Em vez disso, chama tools atômicos (`search_code`, `get_symbol`, `find_callers`) e compõe. Esta é a abordagem MCP — favorece composição sobre encapsulamento.

### 6.5 Filtros e facetas

Toda query deve aceitar filtros estruturados:

```python
search_code(
    query="exercise scoring",
    filters={
        "language": ["java"],
        "path_glob": "backend/src/main/**",
        "kind": ["method", "class"],
        "modified_since": "2026-04-01",
        "annotations_include": ["@Transactional"],
        "exclude_tests": True,
    },
    top_k=10
)
```

Filtros são aplicados **antes** do ANN search (pre-filtering) quando o store suporta (sqlite-vec, Qdrant suportam). Caso contrário, post-filtering com `top_k * 5` candidatos para compensar.

---

## 7. Técnicas específicas para código

### 7.1 Test ↔ implementation linking

Mapear automaticamente:

- Método `StudyServiceTest.shouldRecordCorrectAnswer` → método `StudyService.submitAnswer`.

Estratégias:

1. **Heurística por nome**: `FooTest` → `Foo`. `should<Verb><Object>` → métodos com `<verb>` similar.
2. **Análise de imports**: o teste importa `StudyService` → liga.
3. **Análise de chamadas**: o teste chama `studyService.submitAnswer(...)` → liga ao método específico.

Persistir como aresta `tests` no symbol graph. Tool MCP `find_tests_for("StudyService.submitAnswer")` vira O(1).

### 7.2 Feature ↔ step linking

Para Cucumber:

- Parsear `.feature` files via tree-sitter-gherkin.
- Parsear step definitions Java/TS, extrair regex/cucumber-expression de cada step.
- Casar steps de feature com regex de step def via match exato + similaridade.
- Persistir `feature_step → step_def_method → métodos chamados`.

Resultado: tool `trace_feature("StudentStudySession", "Mastery after 3 correct answers")` devolve a cadeia completa.

### 7.3 Glossário de domínio

Extrair via LLM (Haiku) um glossário de termos do domínio:

```
Subject: atomic knowledge unit (DDD entity); has Content; has RequiredSubjects
Streak of 3: protocol exiting Phase 1 acquisition
NextStudyStep: server-driven UI navigation contract
...
```

Embeddar o glossário como chunks especiais (`kind=glossary`). Queries em linguagem natural batem aqui antes de batem em código, melhorando recall conceitual.

### 7.4 Detecção de "anchors" arquiteturais

Identificar (via análise estática + LLM) os "pontos críticos" do codebase:

- Aggregate roots (DDD).
- GraphQL resources.
- Configuration entry points.
- Security policies.

Marcar com flag `architectural_anchor: true`. Boost no ranking quando query tem termos arquiteturais.

### 7.5 Indexação de comentários e Javadoc separadamente

Comentários explicam intenção; código explica mecanismo. Indexar como chunks separados (kind=`comment`) que linkam ao chunk de código adjacente. Boost para queries em linguagem natural quando matcham comentário.

---

## 8. Atualização incremental

### 8.1 Detecção de mudança

Fonte de verdade: **hash de conteúdo** (WIN-425). O modo `auto` (default)
reconcilia o working tree salvo contra o cache `files` usando SHA-256 dos
bytes completos, independente do Git:

- Caminha o conjunto atual de arquivos (walker configurado).
- Lê um snapshot imutável por tentativa; o chunker nunca relê o path mutável.
- (Re)indexa arquivo novo, com hash completo alterado, snapshot incompleto, OU cujo par de
  drift ativo `(enricher_id, embedder_model)` difere do que o indexou
  (ver §3.1 / ADR-004 — troca de config força reprocessamento).
- Calcula enrichment e embeddings antes da transação, relê/recalcula o hash e
  descarta a tentativa se o arquivo mudou durante o processamento.
- Publica chunks, vetores, FTS e metadados de um path atomicamente via
  `replace_file_snapshot`; arquivos removidos usam `delete_file_snapshot`.
- Chunks cujo conteúdo permanece idêntico reutilizam os dois espaços vetoriais,
  mas atualizam metadados estruturais e FTS dentro do mesmo snapshot atômico.
- Publica `head_commit` e `last_indexed_at` apenas após completar o repositório.

Propriedades: uma falha expõe o snapshot completo anterior ou o novo, nunca uma
mistura; arquivos com zero chunks continuam versionados; edições de mesmo
tamanho com mtime preservado são detectadas. Um marcador de contrato força a
reconciliação automática de índices produzidos pela semântica antiga. O
marcador é scoped por repositório, portanto atualizar um checkout não promove
hashes legados de outros repositórios que compartilham o mesmo storage.
Leituras estáveis validam inode, tamanho, mtime, ctime e o target de symlink
antes/depois da leitura para não publicar bytes de duas versões diferentes.

Mudanças de código também persistem `graph_dirty:<repo_id>`. O marcador
permanece ativo em runs com graphify desabilitado, sem binário ou com falha
parcial; somente o lifecycle completo `graphify update` → mirror → graph vectors
→ FTS o limpa. Assim uma passagem rápida do watcher não perde trabalho de grafo
que deve ser retomado pela próxima indexação completa.

Git é uma **otimização opt-in** (`--incremental`), não a fronteira de
correção:

```
git diff --name-status <last_indexed_commit>..HEAD
```

Para cada arquivo (caminho git-diff):

- `A` (added): chunk + embed + index.
- `M` (modified): rechunk; comparar `content_hash` por chunk; reembed só os mudados.
- `D` (deleted): remover chunks.
- `R` (renamed): atualizar path; preservar embeddings se conteúdo idêntico.

### 8.2 Triggers de reindex

| Trigger | Mecanismo | Latência alvo |
|---|---|---|
| Commit local | Git hook `post-commit` (silencioso, em background) | <5 s |
| Pull / merge | Git hook `post-merge` | <30 s |
| Edição salva não-commitada | Polling stdlib por hash completo, debounce 2 s | Intervalo de polling + debounce |
| Deploy CI | GitHub Actions step | OK ser lento |

### 8.3 Working tree salvo

O índice persistente representa o working tree **salvo**, incluindo edições não
commitadas. `HEAD` é metadado Git, não prova dos bytes indexados. O watcher faz
reconciliação inicial e retries com backoff para não depender de um novo evento
após falha. Seu snapshot usa o path lógico relativo retornado pelo walker, mesmo
quando o path físico foi canonicalizado, e valida o target de symlink ao ler.
Buffers ainda não salvos não entram no índice persistente; overlays de sessão
explícitos são uma extensão futura e separada.

### 8.4 Compactação e rebuild

- **Compactação semanal**: VACUUM no SQLite, rebuild HNSW.
- **Rebuild completo**: trigger manual ao trocar embedding model. Run em background, swap atômico ao final.

### 8.5 Backfill de novos campos

Schema migrations + lazy backfill: adicionar coluna nullable, preencher em background sem bloquear queries.

### 8.6 Incremental indexing delivery (Phase A/B/C)

> **Status:** Implemented (2026-05-15)

| Phase | Schema | Delivery |
|---|---|---|
| **A** | v6 (`files` table) | File-level dirty cache keyed by `(repo_id, rel_path)` storing `mtime`, `size_bytes`, `content_hash`, `last_enricher_id`, `last_embedder_model`. 10x faster no-op runs (14.7s → 1.4s empirically). `Storage.file_is_fresh()` + `upsert_file_meta()`. |
| **B** | — | Git-diff aware ingestion. `git diff --name-status <last_indexed_sha>..HEAD` drives the walker; only touched files re-chunked/embedded. `FileDelta` dataclass with `action: add|modify|delete|rename`. Hook v=4 passes `--incremental`. Auto-fallback to full walk when no checkpoint. |
| **C** | — | Rename-resilient `chunk_id`. `Storage.rename_chunks()` recomputes `chunk_id` via `make_chunk_id(new_path, ...)` + regenerates FTS5 rows. Zero re-embed cost for renames. `GraphifyLoader` diff-loads (deltas only, not full DELETE+INSERT). |
| **D** | — | Hash-based reconcile (WIN-425, 2026-06-08). `auto` (default) diffs the working tree against the `files` cache via `Storage.load_file_meta_map()` — git-independent, crash-resilient (interrupted runs self-heal), sees uncommitted edits, works on non-git dirs. Git-diff (`--incremental`) demoted to an opt-in fast-path. `Pipeline._index_one_repo_reconcile()`. |
| **E** | — | Trustworthy source snapshots (2026-08-21). Full raw-byte SHA-256, immutable chunker input, post-processing rehash, adapter-owned atomic replace/delete, vector-preserving structural refresh, zero-chunk support, completion-only repository watermarks, durable per-repo graph dirtiness, startup watcher reconciliation, retry backoff, and versioned source evidence on every retrieval stream. |

CLI flags: `--full` (force full walk), `--incremental` (force git-diff fast-path, falls back to reconcile if no checkpoint), `--since <sha>` (override checkpoint). Default (no flag) = `auto` = hash reconcile.

`git.rename_threshold` in `~/.stropha/config.yaml` (default 80) controls
`git diff --find-renames=<N>` sensitivity.

---

## 9. MCP Server

### 9.1 Model Context Protocol — visão geral

MCP é o protocolo aberto da Anthropic (2024) para conectar LLMs a ferramentas/dados externos. Especificação: https://modelcontextprotocol.io.

Conceitos:

- **Tools**: funções invocáveis (ex.: `search_code`).
- **Resources**: dados expostos (ex.: estatísticas do índice via URI `stropha://stats`).
- **Prompts**: templates parametrizados (ex.: prompt "explique o módulo X" pré-formatado).
- **Sampling**: o servidor pode pedir ao cliente (LLM) para gerar texto. Útil para HyDE.

Transport:

- **HTTP streamable (único transporte suportado)**: servidor singleton
  exposto em `http://<host>:<port>/mcp`. Um único processo serve todos
  os clientes MCP concorrentemente com um único runtime aquecido de adapters
  (pipeline, embedder, storage e retrieval). `bin/stropha-mcp-ensure` é o
  launcher canônico; health probe em `GET /health`. stdio e SSE não são
  suportados.

### 9.2 Design dos tools

Princípios:

1. **Atômicos**: cada tool faz uma coisa. Cliente compõe.
2. **Idempotentes**: mesma input → mesma output (até a próxima reindexação).
3. **Output enxuto**: nunca retornar conteúdo grande. Path + lines + snippet curto. Cliente lê arquivo se quiser.
4. **Schemas estritos**: JSON Schema com `additionalProperties: false`, descrições ricas (LLM lê).

#### Catálogo implementado e perfis estáticos

`src/stropha/server_tool_manifest.py` é o inventário canônico. Cada entrada
declara o menor perfil que pode descobrir e invocar o tool; a ordem do manifest
é também a ordem de `tools/list`. O startup falha se registro e manifest
divergirem.

| Perfil | Escopo |
|---|---|
| `core` | 13 tools de seleção de repositório, orientação, busca, símbolos, grafo, contexto e saúde do índice |
| `standard` | `core` mais observações efêmeras, ergonomia de sessão, investigações compostas e inteligência de mudança |
| `full` | `standard` mais memória, scratchpad, episódios, feedback, cache e diagnóstico de runtime |

Os perfis são aninhados (`core` ⊂ `standard` ⊂ `full`) e `full` é o default de
compatibilidade. `capabilities` pertence a `core`, portanto está sempre visível,
e retorna perfil ativo, perfis aceitos e nomes ordenados dos tools disponíveis.
O filtro se aplica somente a tools; resources e prompts não mudam por perfil.

Discovery e invocation usam a mesma política: `tools/list` omite tools acima do
perfil ativo e `tools/call` rejeita explicitamente qualquer tentativa de chamar
um tool oculto. Todos os tools registrados publicam output schema estruturado.
Suggestions nunca recomendam tools ocultos, e recipes são validados contra os
schemas dos tools que referenciam. O inventário completo e a composição atual
dos três perfis estão documentados em
`docs/architecture/stropha-agent-exoskeleton.md`.

### 9.3 Formato de resposta

Padrão para `search_code`:

```jsonc
{
  "results": [
    {
      "rank": 1,
      "score": 0.87,
      "path": "backend/src/main/java/com/mimoria/service/StudyService.java",
      "lines": [145, 187],
      "symbol": "StudyService.submitAnswer",
      "kind": "method",
      "snippet": "public StudySubmissionResult submitAnswer(...) {\n    var enrollment = ...\n    // ~30 linhas\n}",
      "source": {
        "source_version": {
          "repository": "github.com/acme/mimoria",
          "path": "backend/src/main/java/com/mimoria/service/StudyService.java",
          "layer": "saved_worktree",
          "file_sha256": "...",
          "observed_at": "2026-08-21T20:00:00+00:00",
          "head_commit": "...",
          "dirty": null
        },
        "start_line": 145,
        "end_line": 187,
        "chunk_id": "sha256:...",
        "source_slice_sha256": null,
        "indexed_file_sha256": "...",
        "freshness": "unknown"
      },
      "summary": "Submete resposta, atualiza streak, dispara FSRS em mastery.",
      "context": {
        "parent": "class StudyService",
        "neighbors": { "imports": ["..."], "annotations": ["@Transactional"] }
      }
    }
  ],
  "query_metadata": {
    "rewritten_query": "...",
    "retrieval_time_ms": 142,
    "rerank_time_ms": 118,
    "total_candidates_before_rerank": 50
  }
}
```

Resultados lidos apenas do índice carregam a versão completa do arquivo, mas
deixam `source_slice_sha256` nulo porque o hash do conteúdo transformado do
chunk não prova os bytes literais da faixa. Ferramentas que relêem o working
tree salvo, como `assert_snippet_freshness_tool`, calculam esse hash da faixa e
marcam a evidência como `fresh`, `stale` ou `missing`.
Todos os streams, inclusive `graph-vec`, hidratam o resultado pelo lookup
canônico para preservar `repo.local_path` e essa mesma evidência de fonte.

### 9.4 Implementação

**Lifecycle do runtime**: o primeiro MCP session inicializa os adapters pesados
sob lock assíncrono e os associa ao `FastMCP` process-wide. Sessions seguintes
reutilizam esse runtime, inclusive no recovery de session id obsoleto. Apenas
estado mutável de cliente (`seen_chunk_ids`, pins, scratchpad, queries recentes,
trace recorder, estado observado do agente e artifacts de runtime) vive no
`AppContext` por session. Observações e artifacts têm quotas fixas, paths
validados contra a raiz local do repositório tanto na entrada quanto na leitura,
e são limpos ao encerrar a session; IDs expirados não podem ser reutilizados na
mesma session. Eles nunca entram em metadata persistente nem traces. No shutdown
ASGI, o session manager encerra as tasks antes de parar watcher e fechar storage.

O perfil MCP é selecionado de `~/.stropha/config.yaml` antes de iniciar o
transport HTTP e permanece fixo por toda a vida do daemon. Não existem
overrides por CLI, ambiente ou session.

#### 9.4.1 Frontend JSON API e live reload

O mesmo daemon HTTP serve um dashboard em `/dashboard` e expõe cinco rotas
JSON locais, separadas do protocolo MCP:

- `GET /api/status`: retorna saúde e identidade do pipeline ativo, contagens do
  índice e aliases de modelos descobertos no MoA Gateway local.
- `GET /api/config`: retorna o documento `~/.stropha/config.yaml` com secrets e
  passwords em URLs redigidos.
- `GET /api/config/schema`: deriva opções, tipos, limites e defaults dos schemas
  Pydantic dos adapters registrados, incluindo metadados tipados para sub-adapters
  aninhados como `pipeline.retrieval.config.reranker`; o dashboard renderiza esses
  controles sob o adapter pai e combina-os com a ajuda operacional dos campos.
- `PATCH /api/config`: aplica deep merge parcial, substituindo a configuração de
  um adapter quando seu nome muda, escreve via temp file + rename atômico e
  constrói um novo conjunto de adapters antes de publicar o runtime.
- `POST /api/query`: executa retrieval no índice ativo, monta evidências com IDs
  `[S1]`, `[S2]`, etc. e usa o backend de inference configurado para responder.
  Histórico recente (`user`/`assistant`) é limitado, validado e usado como
  contexto de retrieval e geração para suportar follow-ups sem estado no servidor.

O swap de runtime ocorre sob o mesmo lock assíncrono process-wide usado no
startup. Falha de validação, conexão de storage ou construção de adapter restaura
o YAML anterior e mantém o runtime anterior publicado. Sessions MCP já abertas
retêm seus adapters para não fechar uma conexão em uso; os runtimes aposentados
são fechados no shutdown ASGI. Novas sessions e requests frontend observam o
runtime substituto imediatamente.

`mcp.profile` é a exceção ao live reload. O PATCH pode persistir o novo perfil,
mas retorna `restart_required: true`; o manifest ativo continua filtrando
discovery e invocation com o perfil anterior até o restart do daemon.

O LLM nunca recebe acesso SQL nem escolhe queries arbitrárias. A rota de query
fornece somente os chunks retornados pelo `RetrievalStage`, trata snippets como
dados não confiáveis e exige citações por source ID. CORS para essas rotas aceita
somente origins HTTP(S) em `localhost`, `127.0.0.1` ou `::1`; não há wildcard.

O backend `moa` conecta ao MoA Gateway local pela API OpenAI-compatible. Modelo,
base URL, timeout, retries e API key opcional são configurados sob `inference.moa_*`
e `secrets.moa_api_key`. Seleção explícita não faz fallback silencioso se o gateway
ou alias estiver indisponível; o dashboard expõe esse estado como falha acionável.

**Stack recomendado**:

- **Linguagem**: Python 3.12 (ecossistema RAG mais maduro). Alternativa: Rust se latência crítica.
- **MCP SDK**: `mcp` (Anthropic oficial, Python). Versão ≥ 1.0.
- **Pipeline**: LlamaIndex `IngestionPipeline` + `RetrieverQueryEngine`.
- **Embeddings**: SDK Voyage (`voyageai`).
- **Storage**: `sqlite-vec` (extensão), `apsw` (bindings SQLite Python performáticos).
- **Tree-sitter**: `tree-sitter-language-pack` (todas as grammars empacotadas).
- **Async**: `asyncio` nativo; uvloop em produção.
- **CLI**: `typer` (Click moderno).
- **Empacotamento**: `uv` para gerenciamento, `hatch` para build.

**Estrutura de projeto** (sugestão):

```
stropha/
├── pyproject.toml
├── src/stropha/
│   ├── __init__.py
│   ├── server.py              # MCP server entry
│   ├── tools/
│   │   ├── search.py
│   │   ├── symbol.py
│   │   └── trace.py
│   ├── ingest/
│   │   ├── walker.py
│   │   ├── chunker.py
│   │   ├── enricher.py
│   │   └── pipeline.py
│   ├── retrieval/
│   │   ├── hybrid.py
│   │   ├── rerank.py
│   │   └── rrf.py
│   ├── storage/
│   │   ├── sqlite.py
│   │   ├── vector.py
│   │   └── graph.py
│   ├── embeddings/
│   │   ├── voyage.py
│   │   └── cache.py
│   └── cli.py
└── tests/
    ├── eval/                  # golden dataset, harness
    └── unit/
```

### 9.5 Configuração dos clientes MCP

Primeiro inicie/verifique o daemon compartilhado com
`bin/stropha-mcp-ensure`. Todos os clientes apontam para a mesma URL; eles não
devem iniciar um subprocesso por projeto.

```json
{
  "mcpServers": {
    "stropha": {
      "url": "http://127.0.0.1:9741/mcp",
      "transport": "streamable-http"
    }
  }
}
```

stdio, SSE, `command`/`args`, and per-client `env` blocks are unsupported. The
singleton owns one warm pipeline/storage runtime; client sessions isolate only
their lightweight dedup, pins, scratchpad, recent-query, and trace state. Todos
os clientes também compartilham o mesmo perfil MCP selecionado no startup;
abrir uma nova session não altera o conjunto de tools.

### 9.6 Segurança do MCP

- **Path traversal**: validar todos os paths recebidos contra raiz do repo. Rejeitar `..`.
- **Secret leakage**: além da exclusão na ingestão, `search_code` aplica uma
  barreira de egress com assinaturas de providers, assignments sensíveis e
  entropia. O snippet inteiro é substituído antes da serialização, e o log
  contém somente `chunk_id`, path e nomes das regras acionadas.
- **Rate limiting**: por cliente (token bucket). Default: 100 req/min.
- **Audit log**: cada tool call logado com timestamp, cliente, args, result_hash.
- **Sandboxing**: para deploy remoto, rodar em container sem network egress (exceto Voyage API).

---

## 10. Avaliação

Sem evaluation, qualquer mudança é palpite. Evaluation é o que separa RAG sério de demo.

### 10.1 Golden dataset

Construir manualmente (50-200 exemplos para começar):

```jsonc
{
  "id": "Q-001",
  "query": "como o sistema decide quando agendar review FSRS?",
  "category": "conceptual",
  "relevant_chunks": [
    "backend/.../FsrsCalculator.java:nextReview",
    "backend/.../StudyFlowCoordinator.java:scheduleNext",
    "docs/architecture/study-flow.md#fsrs-cycle"
  ],
  "graded_relevance": { "...:nextReview": 3, "...:scheduleNext": 2, "...#fsrs-cycle": 3 }
}
```

Categorias balanceadas: conceitual, simbólica, estrutural, multi-hop, ambígua, negativa (deve retornar nada).

### 10.2 Métricas

| Métrica | O que mede | Alvo |
|---|---|---|
| **Recall@k** | % de queries onde ≥1 chunk relevante aparece nos top-k | @10 ≥ 0,85 |
| **Precision@k** | % dos top-k que são relevantes | @5 ≥ 0,60 |
| **MRR** | 1/rank do primeiro relevante | ≥ 0,75 |
| **NDCG@k** | Recall ponderado por posição e relevância graduada | @10 ≥ 0,70 |
| **Latência p50/p95/p99** | Tempo de resposta | p95 ≤ 500 ms |
| **Cobertura** | % de chunks que aparecem em ≥1 query top-50 (detecta dead zones) | > 30% |

### 10.3 Frameworks

- **RAGAS** (https://docs.ragas.io): métricas LLM-as-judge — `faithfulness`, `answer_relevance`, `context_precision`, `context_recall`. Útil quando relevância não é binária.
- **DeepEval**: alternativa com mais integrações pytest.
- **TruLens**: bom para tracing + evaluation acoplados.
- **Phoenix (Arize)**: open-source, ótima UI para análise.
- **ARES**: evaluation com synthetic data generation.

### 10.4 LLM-as-judge

Para queries onde relevance gold é caro de produzir:

- LLM (Sonnet) recebe query + chunk + critério, retorna score 0-3 + justificativa.
- Calibração: rodar em 50 exemplos rotulados manualmente, medir kappa (>0,7 aceitável).
- Custo: ~US$ 0,01 por query × top_k.

### 10.5 A/B testing online

Quando rodar como serviço com múltiplos usuários: split traffic entre configurações (ex.: rerank on vs off), medir engagement (cliques nos resultados, tempo até resolver dúvida).

### 10.6 Regression testing em CI

Toda PR ao código do RAG roda golden set. PR bloqueada se NDCG@10 cair >2 pontos vs baseline.

---

## 11. Observabilidade

### 11.1 Tracing

**OpenTelemetry** como padrão. Spans por etapa:

```
mcp.tool.search_code
  ├─ query.rewrite
  ├─ embedding.encode
  ├─ retrieval.dense
  ├─ retrieval.sparse
  ├─ retrieval.rrf
  ├─ rerank.voyage
  └─ response.pack
```

Backends:

- **Langfuse** (open-source, self-hostable): purpose-built para LLM ops.
- **Phoenix** (Arize, OSS): foco em evaluation + tracing.
- **LangSmith**: SaaS, ótimo se já usa LangChain.
- **Honeycomb / Datadog APM**: APM tradicional, OTel-compatible.

### 11.2 Métricas Prometheus

```
rag_query_total{tool="search_code", status="ok"}
rag_query_duration_seconds{tool, quantile}
rag_chunks_indexed_total
rag_embedding_tokens_total{model}
rag_embedding_cost_usd_total{model}
rag_cache_hit_ratio{cache="embedding"}
rag_rerank_skipped_total{reason="latency_budget"}
```

### 11.3 Logging estruturado

JSON logs via `structlog` (Python) ou `tracing` (Rust). Campos obrigatórios: `request_id`, `tool`, `latency_ms`, `result_count`. Nunca logar query crua se contém path sensível.

### 11.4 Cost dashboard

Diário: tokens embeddados, tokens reranked, USD gasto por modelo. Alertas em > 2x média móvel 7d.

---

## 12. Performance e otimização

### 12.1 Caching multi-nível

| Cache | Key | TTL | Backend |
|---|---|---|---|
| **Embedding cache** | `(content_hash, model)` | infinito (até model change) | SQLite tabela |
| **Query embedding cache** | `(query_text, model)` | 1h | SQLite tabela ou Redis |
| **Rerank cache** | `(query_hash, candidates_hash, model)` | 1h | LRU em RAM |
| **Semantic cache** | embedding da query → resposta | sim. cosine > 0.95 → hit | sqlite-vec auxiliar |

Semantic cache de queries é poderoso para usuários humanos (perguntas similares retornam mesmo set). Para Claude Code, menos útil (cada invocação tem contexto novo).

### 12.2 Batching

- **Embeddings**: agrupar até 128 chunks por chamada à API Voyage (limite do provedor). Reduz overhead de RTT.
- **Reranks**: a API aceita 1 query × N docs por chamada. Sempre enviar top 50 de uma vez.

### 12.3 Connection pooling

Para vector stores remotos (Qdrant), pool de conexões HTTP/2. Para SQLite, abrir conexão por query é OK até ~100 QPS; acima, usar pool com `apsw`.

### 12.4 Prompt caching (Anthropic)

Quando o servidor MCP fizer chamadas LLM (HyDE, summary generation), usar prompt caching:

- System prompt + few-shot examples vão em block cacheado.
- Cache hit reduz custo input em 90% e latência em ~50%.
- TTL: 5 min default; 1h se contratado.

### 12.5 Quantização e precisão

- Vetores em `float32` por padrão.
- Quantizar para `int8` quando índice > 1 GB (perda <1% NDCG).
- Binary quantization apenas se índice > 10 GB.

### 12.6 Profiling

- `py-spy` para hot-paths Python.
- `tracy` para Rust se reescrever.
- SQL `EXPLAIN QUERY PLAN` em queries lentas.

---

## 13. Segurança

### 13.1 Detecção de secrets

Antes de embeddar, rodar:

- `gitleaks` ou `trufflehog` no chunk.
- Regex próprias para padrões internos (chaves Google OAuth, JWT tokens).

Chunks com secrets:

- Marcar `has_secret: true`.
- Mascarar (`****`) o secret no texto persistido.
- Excluir de qualquer response a clientes sem flag `--allow-secrets`.

Como defesa-em-profundidade, o egress de `search_code` reescaneia cada snippet
depois de compressão e packing. Assinaturas conhecidas, URLs com credenciais,
assignments sensíveis ou tokens de alta entropia substituem o snippet inteiro
por um marcador fixo. A resposta expõe apenas `redacted_count`; valores
detectados nunca entram no log.

### 13.2 RBAC (multi-usuário)

Se servir vários usuários:

- Token JWT por usuário com claims (`repo_access`, `paths_allowed`).
- Server filtra results por `paths_allowed`.

### 13.3 Auditoria

Append-only log: `(timestamp, user, tool, args_hash, result_hash, latency)`. Retenção 90d. Útil para investigar incidentes.

### 13.4 Supply chain

- Lock files committed (`uv.lock`).
- Verificar checksums de modelos baixados.
- Renovate/Dependabot para CVEs.

---

## 14. Deployment

### 14.1 Topologias

#### Local (desenvolvedor individual)

```
MCP clients ── streamable HTTP ──► shared stropha-mcp daemon
                                      ├─ warm pipeline/adapters
                                      └─ active storage adapter
```

Custo cloud é opcional; o default permanece local-only. O daemon escuta em
`http://127.0.0.1:9741/mcp` e expõe `GET /health`.

#### Self-hosted (time)

```
                       ┌─ Qdrant (Docker)
GitHub Actions ─indexa─┤
                       └─ Postgres (chunks + graph)
                              ▲
                               │ streamable HTTP
                              │
              dev ─► Claude Code ─► mcp-server (deployed)
```

Single VPS suficiente para times <50 devs. Helm chart se K8s mandatório.

#### Cloud (alta disponibilidade)

- Embedding: Voyage API.
- Vector store: Qdrant Cloud ou Turbopuffer.
- MCP server: AWS Lambda (cold start aceitável para uso esporádico) ou Fargate.
- Storage: S3 para snapshots.

### 14.2 CI/CD para o índice

Pipeline GitHub Actions:

```yaml
on:
  push:
    branches: [develop, main]

jobs:
  reindex:
    runs-on: ubuntu-latest
    steps:
      - checkout
      - uv-install
      - run: uv run stropha reindex --incremental --since=$LAST_INDEXED_SHA
      - upload index.db artifact
      - run evaluation gate (NDCG must not drop > 2 pts)
```

### 14.3 Release artifact

Distribuir como:

- PyPI package (`pip install stropha`).
- Single binary via `pyinstaller` ou `shiv` (~30 MB) para devs não-Python.
- Docker image multi-arch (`linux/amd64`, `linux/arm64`, `darwin/arm64`).

---

## 15. Stack proposto (recomendação concreta)

Decisões para a fase 1 deste projeto:

| Componente | Escolha | Justificativa |
|---|---|---|
| Linguagem servidor | Python 3.12 + `uv` | Ecossistema RAG dominante; iteração rápida. |
| MCP SDK | `mcp` (Anthropic) | Único oficial. |
| Parsing | `tree-sitter-language-pack` | Cobre Java, Vue, TS, Gherkin, YAML, MD num único pacote. |
| Chunking | Custom (semântico hierárquico) | Frameworks genéricos não conhecem Vue SFC nem Gherkin. |
| Embeddings densos | Voyage `voyage-code-3` (dim 512 Matryoshka) | SOTA em código; custo baixo. |
| Embeddings esparsos | SQLite FTS5 (BM25) | Zero infra, suficiente. |
| Vector store | sqlite-vec | Single-file; portabilidade total. |
| Symbol graph | SQLite (mesmo arquivo) | Joins SQL, transacional. |
| Reranker | Voyage `rerank-2.5` | Mesma família, billing único. |
| Query understanding | Heurística + HyDE opcional | Não complicar antes de evaluation justificar. |
| Contextual retrieval | Sim (Anthropic technique) | ROI comprovado, custo aceitável. |
| Pipeline | LlamaIndex `IngestionPipeline` (cacheada) | Reaproveita transformations, fácil swap. |
| Observability | OpenTelemetry → Langfuse local | Open-source, self-hosted, leve. |
| Evaluation | RAGAS + golden set custom | Métricas tradicionais + LLM-as-judge. |
| Deployment | `uv tool install` local; opcional Docker | Atende uso individual primário. |

Custo mensal estimado (uso individual ativo):

| Item | Estimativa |
|---|---|
| Voyage embeddings (incremental) | US$ 0,50 |
| Voyage rerank (~500 queries/mês) | US$ 1,50 |
| Anthropic Haiku (summaries + HyDE) | US$ 1,00 |
| **Total** | **~US$ 3/mês** |

Bootstrap inicial (full reindex Mimoria, ~5K chunks): ~US$ 2 one-shot.

---

## 16. Roadmap em fases

### Phase 0 — Spike (1 dia) ✓
- [x] Setup repo `stropha` com `uv`.
- [x] Walker + chunker dummy (split por arquivo).
- [x] Embedding via Voyage API (com fallback local fastembed quando sem chave).
- [x] sqlite-vec storage.
- [x] CLI `index` / `search` / `stats`.
- [x] Teste manual: "onde está o FSRS calculator?".

**Critério de saída**: query semântica retorna chunk certo no top-3 — ✓.

### Phase 1 — MVP MCP (3-5 dias) ✓ (historical milestone)
- [x] Tree-sitter parsing para Java, TypeScript, JavaScript, Python, Rust, Go, Kotlin via `tree-sitter-language-pack.process()`.
- [x] Chunkers custom para Vue SFC (script/template/style), Markdown (heading split), Gherkin (feature/scenario).
- [x] Chunking semântico hierárquico (skeleton de classe com nome qualificado + lista de membros; método como chunk filho com `parent_chunk_id`).
- [x] Hybrid search com **três streams** fundidas via RRF (k=60): dense (sqlite-vec) + sparse (FTS5 BM25 com expansão CamelCase + tokens de path + symbol) + symbol-token lookup (query routing §6.3.5).
- [x] Servidor MCP (`stropha-mcp`) inicialmente entregue sobre stdio. **Historical
  only:** stdio was later removed; the current transport is the shared
  streamable-HTTP daemon described in **MCP Server**.
- [x] Template MCP inicialmente entregue para Claude Code / Cursor; the current
  template is URL-based and never spawns a subprocess.
- [x] Freshness skip por chunk (re-rodar `index` num repo estável é quase instantâneo).
- [ ] **Diferido** Reindexação incremental via `git diff` — `index --rebuild` cobre o caso de uso para Phase 1; hooks `post-commit` e soft-index em RAM vão para Phase 3.

**Critério de saída**: Claude Code usa o RAG sem ser instruído. Critério objetivo: queries simbólicas/conceituais retornam o chunk certo no top-3. Validado com `where is the FSRS calculator` (rank 3), `calculateNewStability` (rank 1), `EnrollmentRepository findByUsername` (rank 1) usando o embedder local de fallback.

### Phase 2 — Qualidade (1 semana)
- [ ] Reranking Voyage rerank-2.5. *(cloud — deferred per local-only policy)*
- [ ] Contextual retrieval (Anthropic). *(cloud — deferred)*
- [ ] Summary generation via Haiku (cached). *(cloud — deferred; local equivalent via `enricher/ollama` + `enricher/mlx` shipped)*
- [x] Symbol graph com `find_callers`, `find_tests_for`. *(MCP tools shipped: `find_callers`, `find_tests_for`, `find_related`, `get_community`, `find_rationale`, `trace_feature`)*
- [x] Golden dataset inicial (~30 queries shipped, scaling to 50 is incremental).
- [x] Evaluation harness com Recall@K + MRR. *(RAGAS deferred — requires LLM judge; local Recall@K/MRR shipped via `stropha eval`)*
- [ ] Tracing OpenTelemetry → Langfuse local. *(deferred — structlog covers the immediate observability need)*

**Critério de saída**: Recall@10 ≥ 0,85 no golden set.

### Phase 3 — Sofisticação (2 semanas)
- [x] HyDE opcional, query routing. *(local-only via Ollama enricher pipeline)*
- [x] Recursive retrieval / auto-merging. *(via parent_chunk_id + adjacent-line merge)*
- [x] Tool `trace_feature` (Gherkin → step → método). *(MCP tool shipped)*
- [ ] Glossário de domínio embeddado.
- [x] Soft index para working tree (file watcher). *(`stropha watch <repo>` ships with debounce)*
- [ ] Cache semântico de queries.
- [x] Cost dashboard. *(`stropha cost` aggregates structlog by adapter_id)*

### Phase 4 — Escala / extensão
- [x] **Foundation multi-repo**: schema v2 adiciona tabela `repos` + coluna `chunks.repo_id` + tool MCP `list_repos`. Cada `SearchHit` carrega `repo` com URL, branch e HEAD para o cliente fazer `git clone`. Normalização de URL deduplica SSH/HTTPS do mesmo repo; auth tokens são removidos antes da persistência. Detalhes em `src/stropha/ingest/git_meta.py`.
- [x] **Multi-repo indexer UX**: `stropha index --repo A --repo B` (flag `-r` repetível) percorre cada repo sequencialmente compartilhando a mesma Storage e Embedder. `chunk_id` é namespaced por `normalized_key` (via `make_chunk_id(..., repo_key=...)`) — repos diferentes com arquivos idênticos não colidem. `IndexPipeline` aceita `repos: list[Path]`; `IndexStats` agrega contadores por repo (`stats.repos: list[RepoStats]`). `--rebuild` limpa chunks mas preserva tabela `repos`, mantendo FKs estáveis entre rebuilds.
- [x] Manifest YAML para listas declarativas de repos. *(`stropha index --manifest repos.yaml`)*
- [x] Auto-discovery de nested `.git` durante o walk (monorepos com submódulos / vendored deps). *(`walker/nested-git` adapter)*
- [ ] Indexação de dependências externas (Quarkus, Vue) on-demand.
- [x] Modelo de embedding self-hosted (bge-m3) como fallback offline. *(`embedder/bge-m3` runs `BAAI/bge-m3` through the native Apple Silicon `[mlx-embed]` backend; FastEmbed does not support this model.)*
- [ ] Deploy remoto com OAuth 2.1. *(deferred — local-first policy)*
- [ ] Multi-tenant com RBAC. *(deferred — local-first policy)*
- [ ] Web UI para evaluation/debug. *(deferred)*

---

## 17. Riscos e mitigações

| Risco | Probabilidade | Impacto | Mitigação |
|---|---|---|---|
| Custo de embedding explode com refactors massivos | Médio | Médio | Cache por content_hash; alerta de budget. |
| Voyage API down | Baixo | Alto | Fallback para cache + BM25 puro; opção self-host bge-m3. |
| Chunks vazam secrets | Baixo | Crítico | Scanner gitleaks na pipeline; flag `has_secret`. |
| Latência > 1s em queries complexas | Médio | Médio | Latency budget; rerank skip quando exceder. |
| Drift entre código e índice | Alto | Médio | Hooks Git; soft index; reconciliation noturna. |
| Claude Code não usa o tool quando deveria | Alto | Baixo | Documentar no CLAUDE.md quando preferir RAG vs grep. |
| Vendor lock-in Voyage | Médio | Baixo | Embedding model é hot-swappable; abstração `Embedder` interface. |
| Avaliação enviesada (golden set fraco) | Alto | Alto | Crowdsource queries reais via logs; revisão peer trimestral. |

---

## 18. Anti-padrões observados em outras implementações

Lições de RAGs públicos que falham:

1. **Chunking por tamanho fixo** sem respeitar AST → recall ruim em código.
2. **Só dense, sem BM25** → falha em busca por símbolo exato.
3. **Sem rerank** → top-3 frequentemente irrelevante apesar de top-50 relevante.
4. **Reindex full a cada commit** → custo proibitivo, latência inaceitável.
5. **Tool monolítico `query(text)`** → o LLM não consegue compor; melhor expor primitives.
6. **Retornar chunks gigantes** → estoura context window do cliente.
7. **Sem evaluation** → não dá pra saber se mudança ajudou ou piorou.
8. **Sem versionamento de embedding model** → migração quebra produção.
9. **HyDE em tudo** → latência dobra sem ganho de recall na maioria das queries.
10. **GraphRAG aplicado a código** → desperdiça LLM extraindo grafos que AST entrega de graça.

---

## 19. Referências

### Papers fundacionais
- Lewis et al., 2020 — *Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks* (paper original RAG).
- Karpukhin et al., 2020 — *Dense Passage Retrieval for Open-Domain QA* (DPR).
- Khattab & Zaharia, 2020 — *ColBERT: Efficient and Effective Passage Search via Contextualized Late Interaction*.
- Cormack et al., 2009 — *Reciprocal Rank Fusion outperforms Condorcet and individual rank learning methods*.

### Técnicas modernas (2023-2025)
- Anthropic, 2024 — *Introducing Contextual Retrieval* (https://www.anthropic.com/news/contextual-retrieval).
- Edge et al. (Microsoft), 2024 — *From Local to Global: A Graph RAG Approach to Query-Focused Summarization*.
- Gao et al., 2023 — *Precise Zero-Shot Dense Retrieval without Relevance Labels* (HyDE).
- Asai et al., 2023 — *Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection*.
- Yan et al., 2024 — *Corrective Retrieval Augmented Generation* (CRAG).
- Jina AI, 2024 — *Late Chunking: Contextual Chunk Embeddings Using Long-Context Embedding Models*.
- Kusupati et al., 2022 — *Matryoshka Representation Learning*.
- Guo et al., 2024 — *LightRAG: Simple and Fast Retrieval-Augmented Generation*.

### Evaluation
- Es et al., 2023 — *RAGAS: Automated Evaluation of Retrieval Augmented Generation*.
- Saad-Falcon et al., 2023 — *ARES: An Automated Evaluation Framework for RAG Systems*.

### Standards e specs
- Model Context Protocol — https://modelcontextprotocol.io/specification
- OpenTelemetry semantic conventions for LLM — https://opentelemetry.io/docs/specs/semconv/gen-ai/

### Ferramentas referenciadas
- Tree-sitter — https://tree-sitter.github.io
- LlamaIndex — https://docs.llamaindex.ai
- Haystack — https://haystack.deepset.ai
- sqlite-vec — https://github.com/asg017/sqlite-vec
- LanceDB — https://lancedb.github.io
- Qdrant — https://qdrant.tech
- Voyage AI — https://docs.voyageai.com
- Langfuse — https://langfuse.com
- Phoenix (Arize) — https://docs.arize.com/phoenix
- gitleaks — https://github.com/gitleaks/gitleaks

---

## 20. Glossário rápido

| Termo | Definição |
|---|---|
| **AST** | Abstract Syntax Tree |
| **ANN** | Approximate Nearest Neighbor |
| **BM25** | Best Match 25 — função de ranking lexical clássica |
| **Chunk** | Unidade indexável (trecho com metadados) |
| **Cross-encoder** | Modelo que processa query+doc juntos (rerank) |
| **HNSW** | Hierarchical Navigable Small World — índice ANN |
| **HyDE** | Hypothetical Document Embeddings |
| **MCP** | Model Context Protocol |
| **MRR** | Mean Reciprocal Rank |
| **NDCG** | Normalized Discounted Cumulative Gain |
| **RAG** | Retrieval-Augmented Generation |
| **Recall@k** | % de queries com ≥1 relevante nos top-k |
| **RRF** | Reciprocal Rank Fusion |
| **SOTA** | State Of The Art |
| **SPLADE** | Sparse Lexical and Expansion model |
