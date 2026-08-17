# Pipeline & Adapters — Architecture

> **Status:** Implemented. This document preserves the 2026-05-15 RFC and its
> delivery history. Sections that describe the pre-delivery state, proposed file
> moves, env aliases, or project-local configuration are historical. The current
> configuration contract is in **Configuration** below; the live adapter inventory
> comes from `stropha adapters list` / `stropha pipeline show`, not a copied total.
> **Versão:** 1.0 (RFC) → 1.1 (delivery)
> **Data:** 2026-05-15 (RFC) — 2026-05-15 (delivery)
> **Audiência:** maintainers do projeto `stropha` e agentes LLM operando-o.
> **Documentos relacionados:**
> - `docs/architecture/stropha-system.md` (spec mestre)
> - `docs/architecture/stropha-graphify-integration.md` (integração graphify + post-commit)
> - `CLAUDE.md` **Key architecture** (authoritative current configuration and schema)

---

## 0. TL;DR

Cada **nó da pipeline** do stropha (walker, chunker, enricher, embedder, storage, retrieval) é uma **responsabilidade discreta** com **um adapter ativo por vez**, selecionado por configuração. Cada adapter:

1. Implementa o `Stage` protocol da sua responsabilidade.
2. Tem **seção própria** em `~/.stropha/config.yaml`, com CLI overrides pontuais.
3. Aparece em `stropha pipeline show` com parâmetros visíveis — **sem caixa preta**.
4. Auto-registra via decorator no import-time; descoberto por `stropha adapters list`.
5. Pode ser substituído trocando uma string na config; mismatch contra estado persistido aciona re-processamento automático.

Quando um stage tem complexidade interna (despachador + sub-implementações), ele **é** um sub-pipeline encapsulado em adapter, com sub-config aninhado mas igualmente visível. Hoje o `chunker` é o exemplo canônico (dispatcher → 6 chunkers por linguagem). O `retrieval` já é sub-pipeline (até 5 streams + RRF).

Não há composição/chain entre adapters do MESMO stage. Um adapter por stage por vez. Adapters podem ser trocados a qualquer momento; o pipeline detecta drift e re-processa o necessário, com cache.

---

## 1. Motivação

### 1.1 Estado atual

O stropha já tem 50% do padrão implementado, mas inconsistente:

| Stage | Estado adapter-shape | Observação |
|---|---|---|
| walker | ✗ classe única | `ingest/walker.py:Walker` sem protocol |
| chunker | ✓ parcial | dispatcher + `LanguageChunker` protocol + 6 implementações — mas não exposto na config |
| enricher | ✗ inexistente | Phase 2 do roadmap mestre |
| embedder | ✓ | `Embedder` protocol + 2 implementações (Voyage, local fastembed) |
| storage | ✗ classe única | `storage/sqlite.py:Storage` sem protocol |
| retrieval | ✓ sub-pipeline | `adapters/retrieval/hybrid_rrf.py` 4-stream+RRF |

### 1.2 Lacunas que justificam refatoração

1. **Inconsistência cognitiva**: cada stage tem padrão diferente; agentes (humanos e LLM) precisam aprender 6 padrões.
2. **Config dispersa (histórico pré-delivery)**: env vars sem hierarquia visível
   motivaram a fonte única atual, `~/.stropha/config.yaml`.
3. **Extensibilidade ad-hoc**: adicionar novo embedder hoje é fácil (protocol existe); adicionar novo storage backend (Qdrant, pgvector) requer reescrita não-óbvia.
4. **Falta de sub-pipeline first-class**: o chunker JÁ é sub-pipeline mas isso fica implícito; quando o retrieval virar sub-pipeline (futuro: alternar streams individualmente) o padrão não suporta.
5. **Enricher pendente**: a Phase 2 do spec mestre prevê contextual retrieval. Implementar isolado é desperdício — vale fazer dentro do padrão unificado.

### 1.3 Princípios de design

1. **Single-active**: um adapter por stage por vez. Sem chain, sem composição implícita. Composição é responsabilidade do USUÁRIO do adapter, não do adapter framework.
2. **Configuração visível**: principais parâmetros sempre expostos. Sub-pipelines NÃO escondem config interna — fazem nesting estruturado.
3. **Intercambiabilidade**: trocar adapter é trocar string. Zero touch no código.
4. **Detect-and-rebuild**: trocar adapter força re-processamento mínimo dos chunks afetados. O storage adapter ativo persiste as freshness keys.
5. **Fonte única**: settings vivem em `~/.stropha/config.yaml`; não há overlay por projeto nem aliases `STROPHA_*_*` para settings.
6. **Discovery automático**: adapters em `adapters/<stage>/<name>.py` se auto-registram. Adicionar um é adicionar um arquivo.
7. **Validação no startup**: config validada via pydantic schemas antes do pipeline executar. Erros estruturados apontam exatamente qual chave em qual adapter está errada.

---

## 2. Mapa atual → arquitetura proposta

### 2.1 Stages e suas implementações

| Stage | Responsabilidade | Adapter(s) hoje | Adapter(s) planejado(s) |
|---|---|---|---|
| **walker** | Descobrir arquivos indexáveis | `git-ls-files` (hardcoded) | `git-ls-files` (default), `nested-git` (per-file toplevel detection — futuro), `filesystem` (sem git) |
| **chunker** | Split arquivo → chunks (sub-pipeline) | dispatcher + 6 langs (Java, TS, JS, Python, Rust, Go, Kotlin, MD, Vue, Gherkin, fallback) | mesmo + extensões: `late-chunking-jina`, `contextual-prefix` |
| **enricher** | Transformar chunk antes do embed | (inexistente) | `noop`, `hierarchical`, `ollama`, `mlx`, `anthropic`, `openai` |
| **embedder** | Texto → vetor | `voyage`, `local` (fastembed) | mesmo + `mlx`, `ollama-embeddings`, `openai` |
| **storage** | Persistir + query | `sqlite-vec` (única) | mesmo + futuros `qdrant`, `pgvector`, `lancedb` |
| **retrieval** | Query → top-k (sub-pipeline) | hybrid 4-stream + RRF | `dense-only`, `bm25-only`, `hybrid-rrf` (default), `hybrid-rrf-rerank` |

### 2.2 Visão arquitetural

```
┌────────────────────────────────────────────────────────────────────┐
│                            PIPELINE                                │
│                                                                    │
│  ┌─────────┐   ┌─────────┐   ┌──────────┐   ┌──────────┐           │
│  │ walker  │ → │ chunker │ → │ enricher │ → │ embedder │ ───┐      │
│  └─────────┘   └─────────┘   └──────────┘   └──────────┘    │      │
│       │             │              │              │         │      │
│       ▼             ▼              ▼              ▼         ▼      │
│   git-ls-files  tree-sitter-   hierarchical    local /    storage  │
│   (single)      dispatch       (single)        voyage    sqlite-vec│
│                 (sub-pipeline)                                     │
│                       ▲                                            │
│                       │ per-language sub-adapters                  │
│                       │ (java, vue, markdown, …)                   │
└────────────────────────────────────────────────────────────────────┘
                                                              │
                                                              ▼
                                            ┌──────────────────────────┐
                                            │       retrieval          │
                                            │       hybrid-rrf         │
                                            │       (sub-pipeline)     │
                                            │  ┌──────┬──────┬──────┐  │
                                            │  │dense │sparse│symbol│  │
                                            │  └──────┴──────┴──────┘  │
                                            │         └─ RRF ─┘        │
                                            └──────────────────────────┘
```

---

## 3. Anatomia de um Stage

### 3.1 Protocol

```python
# src/stropha/pipeline/base.py

from typing import Protocol, TypeVar
from pydantic import BaseModel

Input = TypeVar("Input")
Output = TypeVar("Output")


class Stage(Protocol[Input, Output]):
    """One responsibility in the pipeline. Exactly one adapter active per run."""

    @property
    def stage_name(self) -> str:
        """The stage's identifier: 'walker' | 'chunker' | 'enricher' | …"""

    @property
    def adapter_name(self) -> str:
        """The adapter's identifier: 'git-ls-files' | 'ollama' | 'voyage' …"""

    @property
    def adapter_id(self) -> str:
        """Stable, fully-qualified id including model/config used for cache keys.
        Examples: 'noop', 'hierarchical', 'ollama:qwen2.5-coder:14b',
                  'voyage:voyage-code-3:512'."""

    @property
    def config_schema(self) -> type[BaseModel]:
        """Pydantic model describing this adapter's config section.
        Used for validation, CLI help, and `stropha adapters info <name>`."""

    def health(self) -> StageHealth:
        """Lightweight readiness probe. Used by `stropha pipeline validate`.
        Non-blocking, ≤2s. Returns Ready / Warning / Error with detail."""

    def run(self, input: Input, ctx: StageContext) -> Output:
        """Execute this stage's logic on `input`. Pure function ideally;
        side-effects (storage writes) declared explicitly via ctx."""
```

### 3.2 Tipos canônicos

```python
@dataclass
class StageContext:
    """Cross-cutting context every stage may consult."""
    repo: Repo | None                # current repo identity (for chunker enricher)
    parent_chunk: Chunk | None       # for hierarchical enricher
    file_content: str | None         # for LLM enrichers that read whole file
    pipeline_meta: dict[str, str]    # arbitrary key-value, set by upstream stages


@dataclass
class StageHealth:
    status: Literal["ready", "warning", "error"]
    message: str
    detail: dict[str, str] = field(default_factory=dict)
```

### 3.3 Input/Output por stage

| Stage | Input | Output |
|---|---|---|
| walker | `Path` (target repo) | `Iterable[SourceFile]` |
| chunker | `SourceFile` | `Iterable[Chunk]` |
| enricher | `Chunk` | `Chunk` (com `embedding_text` setado) |
| embedder | `Sequence[str]` | `list[list[float]]` |
| storage | `Chunk + vec + repo_id` | persisted row id |
| retrieval | `str` (query) | `list[SearchHit]` |

Tipos definidos em `src/stropha/models.py` (já existentes; serão estendidos com `embedding_text`, `enricher_id` quando enricher entrar).

---

## 4. Sub-pipelines

### 4.1 Princípio

Quando um stage tem complexidade interna (despachador + sub-adapters), ele É um sub-pipeline encapsulado num adapter. A regra:

- **Principais parâmetros sempre visíveis**, mesmo dentro de sub-pipelines.
- Sub-config aninhado em estrutura clara no YAML.
- O adapter "pai" coordena seus sub-adapters; o framework não tem nada especial pra sub-pipelines — é só convenção.

### 4.2 Chunker como sub-pipeline (existente, formaliza)

O `Chunker` atual já É um sub-pipeline (dispatcher → per-language). A formalização expõe isso na config:

```yaml
chunker:
  adapter: tree-sitter-dispatch
  config:
    max_chars_per_chunk: 24000
    languages:
      java:           { adapter: ast-generic }
      typescript:     { adapter: ast-generic }
      javascript:     { adapter: ast-generic }
      python:         { adapter: ast-generic }
      rust:           { adapter: ast-generic }
      go:             { adapter: ast-generic }
      kotlin:         { adapter: ast-generic }
      markdown:       { adapter: heading-split, levels: [1, 2, 3] }
      vue:            { adapter: sfc-split }
      gherkin:        { adapter: regex-feature-scenario }
      _fallback:      { adapter: file-level }
```

Cada entrada em `languages.*` é um sub-adapter resolvido pelo dispatcher quando ele encontra um arquivo daquela linguagem.

### 4.3 Retrieval como sub-pipeline (futuro)

```yaml
retrieval:
  adapter: hybrid-rrf
  config:
    top_k: 10
    rrf_k: 60
    streams:
      dense:
        adapter: vec-cosine
        config: { k: 50 }
      sparse:
        adapter: fts5-bm25
        config: { k: 50, expand_camelcase: true, tokenizer: unicode61 }
      symbol:
        adapter: like-tokens
        config: { k: 20, stopwords: default }
```

Habilita trocar uma stream individualmente — ex.: `sparse` de FTS5-BM25 pra SPLADE no futuro — sem refator.

### 4.4 Limite explícito: não é chain

Sub-pipeline ≠ chain. Sub-pipeline tem um adapter "pai" que CONTROLA seus filhos. Chain seria múltiplos adapters do MESMO stage aplicados em sequência. **Stropha não suporta chain por design** (decisão tomada em conjunto com o usuário). Composição é responsabilidade do criador do adapter, internamente. Exemplo: `OllamaEnricher` pode ler `parent_chunk` do `StageContext` e usar como contexto no prompt — sem precisar "compor" com `HierarchicalEnricher`.

---

## 5. Configuração

### 5.1 Fontes e precedência

Contrato atual:

```
CLI override explícito  >  ~/.stropha/config.yaml  >  built-in defaults
```

`~/.stropha/config.yaml` é a única fonte persistente de settings. Não existe
`./stropha.yaml`, overlay `.env`, nem convenção genérica `STROPHA_*_*` para
configuração. Environment variables are restricted to the process/runtime
registry: target selection, shared MCP launcher, session/tracing identity,
detached-hook escape hatches, and opt-in local PostgreSQL container management.
None participates in the settings precedence above.

### 5.2 YAML — exemplo end-to-end

```yaml
# ~/.stropha/config.yaml

pipeline:
  walker:
    adapter: git-ls-files
    config:
      max_file_bytes: 524288
      respect_strophaignore: true

  chunker:
    adapter: tree-sitter-dispatch
    config:
      languages:
        java:           { adapter: ast-generic }
        typescript:     { adapter: ast-generic }
        javascript:     { adapter: ast-generic }
        python:         { adapter: ast-generic }
        rust:           { adapter: ast-generic }
        go:             { adapter: ast-generic }
        kotlin:         { adapter: ast-generic }
        vue:            { adapter: sfc-split }
        markdown:       { adapter: heading-split }
        gherkin:        { adapter: regex-feature-scenario }
        _fallback:      { adapter: file-level }

  enricher:
    adapter: hierarchical             # inspect alternatives with adapters list
    config:
      include_parent_skeleton: true
      include_repo_url: false

  embedder:
    adapter: local                    # inspect alternatives with adapters list
    config:
      model: mixedbread-ai/mxbai-embed-large-v1
      batch_size: 32

  storage:
    adapter: sqlite-vec
    config:
      path: ~/.stropha/index.db

  retrieval:
    adapter: hybrid-rrf
    config:
      top_k: 10
      rrf_k: 60
      streams:
        dense:
          adapter: vec-cosine
          config: { k: 50 }
        sparse:
          adapter: fts5-bm25
          config: { k: 50 }
        symbol:
          adapter: like-tokens
          config: { k: 20 }
```

### 5.3 Runtime arguments

CLI flags podem substituir settings para uma invocação sem criar outra fonte
persistente. The canonical process/runtime env list is `_ENV_REGISTRY` in
`src/stropha/config/__init__.py`. Env vars for settings cited in historical
sections of this RFC are unsupported.

### 5.5 Secrets

API keys ficam no próprio YAML sob `secrets:`. O dashboard e `get_config`
redigem esses valores; exemplos e logs nunca devem conter segredos reais:

```yaml
secrets:
  voyage_api_key: null
  anthropic_api_key: null
  deepseek_api_key: null
  lms_api_key: null
  moa_api_key: null
```

---

## 6. CLI surface

### 6.1 Novos subcomandos

```bash
# Pipeline introspection
stropha pipeline show                    # YAML resolvido (com overrides aplicados)
stropha pipeline validate                # health check em todos os stages
stropha pipeline diff                    # delta vs último estado persistido (DB metadata)

# Adapter introspection
stropha adapters list                    # registry completo, agrupado por stage
stropha adapters list --stage enricher   # filtra por stage
stropha adapters info ollama             # config schema + status + docstring
```

### 6.2 Commands existentes mantêm comportamento

`stropha index`, `stropha search`, `stropha stats` continuam idênticos pela ótica do usuário; internamente usam o `Pipeline` montado via builder.

Flags ad-hoc pra override:

```bash
stropha index --enricher hierarchical --embedder voyage
stropha index --no-enricher                    # explícito disable
stropha search "..." --retrieval dense-only    # override só pra essa query
```

### 6.3 `stropha pipeline show` output esboçado

```
Pipeline composition (resolved from: ~/.stropha/config.yaml)
══════════════════════════════════════════════════════════════════

  walker     git-ls-files                                ✓ ready
             max_file_bytes:        524288
             respect_strophaignore: true

  chunker    tree-sitter-dispatch                        ✓ ready
             max_chars_per_chunk:   24000
             languages:             java, typescript, javascript, python,
                                    rust, go, kotlin, vue, markdown, gherkin,
                                    +fallback (file-level)

  enricher   hierarchical                                ✓ ready
             include_parent_skeleton: true
             include_repo_url:        false

  embedder   local (mxbai-embed-large-v1)                ✓ ready
             dim:        1024
             batch_size: 32

  storage    sqlite-vec                                  ✓ ready
             path:       ~/.stropha/index.db
             chunks:     4654, repos: 1

  retrieval  hybrid-rrf                                  ✓ ready
             top_k:      10
             rrf_k:      60
             streams:    dense(k=50) + sparse(k=50) + symbol(k=20)

Invocation overrides:
  --enricher hierarchical
  --embedder local
```

### 6.4 `stropha adapters list` output

```
Stage      Adapter                Status      Description
─────────  ─────────────────────  ──────────  ──────────────────────────────────
walker     git-ls-files           ✓ ready     Discover via `git ls-files` (default)
walker     nested-git             ⚠ planned   Per-file git toplevel detection
walker     filesystem             ✓ ready     Recursive walk (no git needed)

chunker    tree-sitter-dispatch   ✓ ready     Language-aware AST dispatcher
chunker    file-level             ✓ ready     Single chunk per file (baseline)

enricher   noop                   ✓ ready     Identity (no transformation)
enricher   hierarchical           ✓ ready     Prepend parent chunk skeleton
enricher   ollama                 ✗ down      Daemon unreachable at localhost:11434
                                              Configured model: qwen2.5-coder:14b (not pulled)
enricher   anthropic              ✗ no key    secrets.anthropic_api_key not set
enricher   mlx                    ⚠ planned   Not yet implemented
enricher   openai                 ⚠ planned   Not yet implemented

embedder   local                  ✓ ready     fastembed/ONNX local
embedder   voyage                 ✗ no key    secrets.voyage_api_key not set
embedder   mlx                    ⚠ planned   Not yet implemented

storage    sqlite-vec             ✓ ready     SQLite + sqlite-vec extension
storage    qdrant                 ⚠ planned   Qdrant server (Phase 4 scale)
storage    pgvector               ⚠ planned   Postgres + pgvector

retrieval  hybrid-rrf             ✓ ready     Dense + BM25 + symbol → RRF (default)
retrieval  dense-only             ✓ ready     Vector search only (fast)
retrieval  bm25-only              ✓ ready     Lexical only (no embedder needed)
retrieval  hybrid-rrf-rerank      ⚠ planned   Phase 2 + Voyage rerank-2.5
```

---

## 7. Estrutura de arquivos

### 7.1 Layout proposto

```
src/stropha/
├── pipeline/                            # pipeline framework
│   ├── __init__.py                       # exports Pipeline, build_pipeline
│   ├── base.py                           # Stage protocol, StageContext, StageHealth
│   ├── registry.py                       # register_stage, register_adapter, lookup
│   ├── config.py                         # YAML load, env merge, validation
│   ├── builder.py                        # config dict → instantiated Pipeline
│   └── pipeline.py                       # Pipeline class — orchestrates 6 stages
│
├── stages/                              # stage protocols + IO types
│   ├── __init__.py
│   ├── walker.py                         # WalkerStage protocol
│   ├── chunker.py                        # ChunkerStage protocol
│   ├── enricher.py                       # EnricherStage protocol
│   ├── embedder.py                       # EmbedderStage protocol
│   ├── storage.py                        # StorageStage protocol
│   └── retrieval.py                      # RetrievalStage protocol
│
├── adapters/                            # concrete adapters
│   ├── __init__.py                       # auto-imports all adapter modules
│   ├── walker/
│   │   ├── git_ls_files.py
│   │   ├── nested_git.py                 (future)
│   │   └── filesystem.py
│   ├── chunker/
│   │   ├── tree_sitter_dispatch.py
│   │   ├── languages/                    # sub-adapters of the dispatcher
│   │   │   ├── ast_generic.py
│   │   │   ├── heading_split.py          (markdown)
│   │   │   ├── sfc_split.py              (vue)
│   │   │   ├── regex_feature_scenario.py (gherkin)
│   │   │   └── file_level.py             (fallback)
│   │   └── file_level.py                 (top-level fallback adapter)
│   ├── enricher/
│   │   ├── noop.py
│   │   ├── hierarchical.py
│   │   ├── ollama.py
│   │   ├── anthropic.py                  (future)
│   │   ├── mlx.py                        (future)
│   │   └── openai.py                     (future)
│   ├── embedder/
│   │   ├── local.py
│   │   └── voyage.py
│   ├── storage/
│   │   └── sqlite_vec.py
│   └── retrieval/
│       ├── hybrid_rrf.py
│       ├── dense_only.py
│       └── bm25_only.py
│
├── models.py                            (extended with embedding_text, enricher_id)
├── errors.py
├── config.py                            (legacy shim → pipeline/config.py)
├── cli.py                               (uses Pipeline; adds `pipeline`, `adapters` cmds)
└── server.py                            (MCP; uses Pipeline)
```

### 7.2 Mapeamento código atual → adapter

| Hoje | Vira |
|---|---|
| `ingest/walker.py:Walker` | `adapters/walker/git_ls_files.py:GitLsFilesWalker` |
| `ingest/chunker.py:Chunker` (dispatcher) | `adapters/chunker/tree_sitter_dispatch.py:TreeSitterDispatchChunker` |
| `ingest/chunkers/ast_generic.py` | `adapters/chunker/languages/ast_generic.py` |
| `ingest/chunkers/markdown.py` | `adapters/chunker/languages/heading_split.py` |
| `ingest/chunkers/vue.py` | `adapters/chunker/languages/sfc_split.py` |
| `ingest/chunkers/gherkin.py` | `adapters/chunker/languages/regex_feature_scenario.py` |
| `ingest/chunkers/fallback.py` | `adapters/chunker/languages/file_level.py` |
| `embeddings/local.py` | `adapters/embedder/local.py:LocalEmbedder` |
| `embeddings/voyage.py` | `adapters/embedder/voyage.py:VoyageEmbedder` |
| `storage/sqlite.py:Storage` | `adapters/storage/sqlite_vec.py:SqliteVecStorage` |
| `retrieval/search.py:SearchEngine` | `adapters/retrieval/hybrid_rrf.py:HybridRrfRetrieval` |
| `retrieval/rrf.py:rrf_fuse` | usado internamente por `hybrid_rrf.py` |
| `ingest/pipeline.py:IndexPipeline` | `pipeline/pipeline.py:Pipeline` (genérica, parametrizada) |

---

## 8. Validação de config

### 8.1 Schema por adapter

Cada adapter declara seu schema via pydantic:

```python
# adapters/enricher/ollama.py

from pydantic import BaseModel, Field, field_validator
from pipeline.registry import register_adapter
from stages.enricher import EnricherStage


class OllamaEnricherConfig(BaseModel):
    model: str = Field(default="qwen2.5-coder:14b", description="Ollama model tag")
    host: str = Field(default="http://localhost:11434")
    timeout_s: float = Field(default=30.0, gt=0)
    num_ctx: int = Field(default=8192, ge=512)

    @field_validator("host")
    @classmethod
    def must_be_http_url(cls, v: str) -> str:
        if not v.startswith(("http://", "https://")):
            raise ValueError("host must be a full URL")
        return v


@register_adapter(stage="enricher", name="ollama")
class OllamaEnricher(EnricherStage):
    Config = OllamaEnricherConfig

    def __init__(self, config: OllamaEnricherConfig) -> None:
        self._config = config

    @property
    def adapter_id(self) -> str:
        return f"ollama:{self._config.model}"

    # ... health(), run() ...
```

### 8.2 Erro estruturado

Builder valida tudo no startup; erros apontam exatamente o problema:

```
ConfigError: pipeline.enricher.config.timeout_s
  expected: positive float
  got: -5

  Adapter: ollama
  See: stropha adapters info ollama
```

### 8.3 Validação cross-stage

Algumas constraints atravessam stages:

- `embedder.dim` (output) deve casar com `storage.embedding_dim` (input). Builder calcula e propaga.
- `retrieval.streams.dense` só faz sentido se `embedder` é instanciado. Builder valida.
- `enricher.adapter=ollama` requer Ollama daemon — health check é warning, não erro fatal.

Implementado em `pipeline/builder.py:validate_pipeline_consistency()`.

---

## 9. Adapter discovery & registry

### 9.1 Auto-registration

```python
# pipeline/registry.py

_REGISTRY: dict[tuple[str, str], type] = {}   # (stage_name, adapter_name) → class

def register_adapter(stage: str, name: str):
    def deco(cls):
        if not hasattr(cls, "Config"):
            raise TypeError(f"{cls.__name__} missing Config attribute")
        _REGISTRY[(stage, name)] = cls
        return cls
    return deco

def lookup_adapter(stage: str, name: str) -> type:
    if (stage, name) not in _REGISTRY:
        available = sorted(n for s, n in _REGISTRY if s == stage)
        raise ConfigError(
            f"Unknown {stage} adapter {name!r}. Available: {available}"
        )
    return _REGISTRY[(stage, name)]

def available_for_stage(stage: str) -> list[str]:
    return sorted(n for s, n in _REGISTRY if s == stage)
```

### 9.2 Auto-import

`adapters/__init__.py` faz import recursivo de todos os módulos no startup:

```python
# adapters/__init__.py
import importlib
import pkgutil

for _, modname, ispkg in pkgutil.walk_packages(__path__, prefix=__name__ + "."):
    importlib.import_module(modname)
```

Side-effect: cada import dispara o `@register_adapter` decorator. Após a primeira import de `stropha.adapters`, todo o registry está populado.

Trade-off: import-time side effect. Mitigação: imports são cheap (apenas registrar classes; pesados como ONNX load só ocorrem na instanciação).

---

## 10. Plano de execução em fases

### 10.1 Phase 1 — Framework + Embedder + Enricher (~2 dias)

Por que esses primeiro:
- Embedder já é adapter-shaped (`Embedder` protocol existe).
- Enricher é novo e o trigger imediato do refactor.
- Validamos o padrão sem quebrar nada.

Entregáveis:
1. `pipeline/base.py` (Stage protocol, StageContext, StageHealth)
2. `pipeline/registry.py` (register / lookup / available)
3. `pipeline/config.py` (YAML loader + env merging + legacy alias translation)
4. `pipeline/builder.py` (config dict → Pipeline com stages instanciados)
5. `pipeline/pipeline.py` (Pipeline class — chama stages na ordem)
6. `stages/embedder.py` (EmbedderStage protocol)
7. `stages/enricher.py` (EnricherStage protocol)
8. Mover `embeddings/local.py` → `adapters/embedder/local.py`
9. Mover `embeddings/voyage.py` → `adapters/embedder/voyage.py`
10. Criar `adapters/enricher/noop.py` + `adapters/enricher/hierarchical.py`
11. Schema migration v3 (chunks.embedding_text + chunks.enricher_id + enrichments cache table)
12. `models.py`: Chunk ganha embedding_text + enricher_id
13. `cli.py`: `stropha pipeline show` (parcial — apenas stages migrados); `stropha adapters list` (parcial); CLI flags `--enricher`, `--embedder` no `index`
14. Tests (>15 casos novos)
15. Docs (update CLAUDE.md, stropha-system.md)

**Critério de saída**: `stropha index` reproduz comportamento atual. `stropha pipeline show` mostra embedder/enricher novos + outros stages como "legacy". Trocar `pipeline.enricher.adapter` em `~/.stropha/config.yaml` entre `noop`/`hierarchical` aciona re-enrichment automático sem `--rebuild`.

### 10.2 Phase 2 — Walker + Storage + Retrieval (~2 dias)

Migrar os outros 3 stages pro padrão. Sem mudança funcional.

Entregáveis:
1. `stages/{walker,storage,retrieval}.py`
2. Mover code existente → `adapters/walker/git_ls_files.py`, `adapters/storage/sqlite_vec.py`, `adapters/retrieval/hybrid_rrf.py`
3. `Pipeline` class completa orquestrando todos os 6 stages
4. `IndexPipeline` antigo vira shim de back-compat
5. `stropha pipeline show` completo
6. `stropha pipeline validate` (health check em todos)
7. Tests + regressão zero

**Critério de saída**: Zero código fora de `pipeline/` e `adapters/` toca diretamente em qualquer um dos 6 stages. Tudo passa pelo registry/builder. CLI `index`/`search`/`stats` idêntico.

### 10.3 Phase 3 — Chunker como sub-pipeline (~1 dia)

Refatorar o dispatcher do chunker pro padrão sub-pipeline.

Entregáveis:
1. `adapters/chunker/tree_sitter_dispatch.py` (dispatcher como adapter, com sub-config)
2. `adapters/chunker/languages/*.py` (sub-adapters, mover de `ingest/chunkers/`)
3. YAML expõe `chunker.config.languages.{java,markdown,…}`
4. Tests

**Critério de saída**: Trocar a chunker de uma linguagem específica via YAML (ex.: markdown de `heading-split` pra hipotética `late-chunking-jina`) sem mexer em código Python.

### 10.4 Phase 4 — Retrieval como sub-pipeline (~0.5 dia)

Streams (`dense`, `sparse`, `symbol`) viram sub-adapters.

Entregáveis:
1. `adapters/retrieval/streams/{vec_cosine,fts5_bm25,like_tokens}.py`
2. `adapters/retrieval/hybrid_rrf.py` lê sub-config + RRF
3. YAML expõe params por stream
4. Tests

**Critério de saída**: Ajustar `rrf_k`, top_k por stream, ou trocar `sparse` por outro implementador (SPLADE futuro) sem refator.

### 10.5 Phase 5 — Adapters novos (~0.5 dia cada, sob demanda)

- `adapters/enricher/ollama.py`
- `adapters/enricher/anthropic.py`
- `adapters/enricher/mlx.py`
- `adapters/embedder/mlx.py`
- `adapters/retrieval/hybrid_rrf_rerank.py` (com Voyage rerank-2.5)
- `adapters/storage/qdrant.py` (escala futura)
- ...

Cada um cabe no padrão; nenhum requer mudança no core.

---

## 11. Decisões registradas (ADRs)

### ADR-001 — Single-active adapter, sem chain

**Contexto**: Strategy pattern puro vs. Composite pattern. O usuário pode querer "hierarchical + ollama em sequência".

**Decisão**: Single-active. Um adapter por stage por vez. Composição interna ao adapter (ex: `OllamaEnricher` pode usar contexto hierárquico no prompt LLM, mas não compõe com outro adapter externo).

**Justificativa**: Reduz complexidade do framework drasticamente. Cada adapter é self-contained, testável isoladamente. Falha de um adapter não compromete uma "chain" — fail isolado.

**Trade-off**: Adapters LLM duplicam lógica de "gather hierarchical context". Mitigado por helper compartilhado em `enrichers/_context.py`.

### ADR-002 — YAML como config primária

**Contexto**: env-only vs YAML vs TOML vs JSON.

**Decisão atual (supersede a formulação do RFC)**: `~/.stropha/config.yaml` é a
fonte persistente única; CLI overrides ficam no topo. Não há projeto
`stropha.yaml`, `.env` overlay ou env cascade de settings.

**Justificativa**: Config hierárquica (sub-pipelines, per-language adapters) fica ilegível em env vars planas. YAML é o padrão da indústria pra config humana (Kubernetes, GitHub Actions, dbt, etc.). Comentários inline ajudam exploração.

**Trade-off**: Nova dependência (`pyyaml`, ~200 KB). Aceito.

### ADR-003 — Auto-import + decorator registration

**Contexto**: registry centralizado (`_ADAPTERS = {...}`) vs auto-discovery via decorators.

**Decisão**: Auto-import recursivo de `adapters/` + `@register_adapter` decorator.

**Justificativa**: Adicionar adapter = adicionar 1 arquivo. Zero touch em código central. Extensibilidade limpa.

**Trade-off**: Side effects no import. Mitigado: imports são cheap; pesados só na instanciação.

### ADR-004 — Auto-detect adapter drift (sem `--rebuild` manual)

**Contexto**: Quando usuário troca de adapter (ex: `hierarchical` → `ollama`), como saber o que re-processar?

**Decisão**: `chunks.enricher_id` armazena qual adapter gerou cada chunk. Pipeline em modo incremental compara `current_adapter.adapter_id` vs `chunk.enricher_id` — mismatch dispara re-enrichment + re-embed automaticamente. Cache `enrichments(content_hash, adapter_id)` torna re-enrichment barato em repeat scenarios.

**Justificativa**: UX trivial — usuário só troca config e roda `stropha index`. Sem necessidade de lembrar de `--rebuild`. Adapter_id inclui modelo (ex: `ollama:qwen2.5-coder:14b`) — trocar de modelo também invalida.

**Trade-off**: Schema gain de 1 coluna + 1 tabela. Aceito.

### ADR-005 — Backward-compat via translation layer (histórico, superseded)

**Contexto histórico**: o RFC avaliou manter aliases de env vars de settings.

**Decisão no RFC**: manter aliases por duas versões major. Essa janela terminou;
o contrato atual usa `~/.stropha/config.yaml`, CLI overrides, e somente os
argumentos de processo/runtime enumerados por `_ENV_REGISTRY`.

**Justificativa**: Zero fricção pra usuários existentes. Documentação migra gradualmente.

**Trade-off**: ~30 linhas de translation. Trivial.

### ADR-006 — Pipeline (não chain) com sub-pipelines first-class

**Contexto**: O `chunker` JÁ é sub-pipeline (dispatcher → per-language). Formalizar como pattern reusável vs deixar implícito.

**Decisão**: Formalizar. Sub-pipeline = adapter cujo `config` declara seus sub-adapters. Sem suporte especial do framework — apenas convenção de nesting no YAML. Pai coordena filhos.

**Justificativa**: Mesmo padrão hoje (chunker) e amanhã (retrieval streams). Documentado, descobrível, expansível.

**Trade-off**: Nenhum significativo.

### ADR-007 — Validação no startup, health check no demand

**Contexto**: Quando validar config? Quando checar health de adapters?

**Decisão**:
- Config: validada via pydantic schemas no `build_pipeline()` (startup) — erro fatal antes de qualquer trabalho.
- Health: opt-in via `stropha pipeline validate` ou implícito antes de `stropha index` (warning, não fatal exceto pra storage).

**Justificativa**: Config inválida = bug do usuário, falha rápida. Health = condição transiente (Ollama down, network), não vale bloquear comandos normais.

### ADR-008 — Adapter ID inclui modelo/versão pra cache

**Contexto**: `adapter_id` é só `ollama`, ou `ollama:qwen2.5-coder:14b`?

**Decisão**: Inclui modelo/versão. Cache key = `(content_hash, adapter_id)`. Trocar modelo do mesmo adapter = cache miss = re-processamento.

**Justificativa**: Modelos diferentes produzem outputs diferentes. Cache que ignora o modelo entrega lixo.

### ADR-009 — Sub-pipeline depth limit = 1

**Contexto**: Sub-pipeline pode ter sub-sub-pipelines?

**Decisão**: Não. Stops at 1 level. Se algum dia precisar mais profundidade, é signal de over-engineering.

**Justificativa**: KISS. Profundidade ilimitada complica YAML reading e debugging.

### ADR-010 — Plugin externo fora de escopo (v1)

**Contexto**: Permitir adapters de terceiros via pip plugins (entry points)?

**Decisão**: Não na primeira versão. Custom adapters vivem dentro de `src/stropha/adapters/`. Quando demanda real surgir, entry points são extensão trivial.

**Justificativa**: YAGNI. Plugin system tem complexidade própria (versioning, security, discovery). Não justificado hoje.

---

## 12. Decisões em aberto (histórico do RFC)

Registro preservado das opções avaliadas antes da implementação; não é uma lista
de decisões atuais. A decisão entregue é a fonte única
`~/.stropha/config.yaml` descrita em **Configuration**.

| # | Decisão | Opções | Default recomendado |
|---|---|---|---|
| 1 | Formato config | YAML / TOML / JSON / env-only | **YAML + env** |
| 2 | Localização YAML | per-projeto / per-usuário / ambos | **ambos com merge (projeto > usuário)** |
| 3 | Back-compat env vars antigas | aliases / hard break | **aliases por 2 versões** |
| 4 | Escopo refactor | tudo (5-6 dias) / Phase 1 só / Phase 1+2 | **Phase 1 só, valida, depois decide** |
| 5 | Sub-pipeline depth inicial | chunker only / chunker + retrieval | **chunker only (retrieval na Phase 4)** |
| 6 | CLI pipeline subcommands | show / validate / diff / doctor — quais agora | **show + validate na Phase 1; diff/doctor depois** |
| 7 | Health check timing | sync / async paralelo | **sync (latência aceitável)** |
| 8 | Adapter discovery | auto-import / lista explícita | **auto-import** |
| 9 | Schema version no YAML | sim desde dia 0 / só quando precisar | **sim (1 linha de custo)** |
| 10 | Custom adapters externos | plugin entry points / só internos | **só internos (YAGNI)** |

---

## 13. Compatibilidade backward

### 13.1 Garantias históricas da migração

1. **Zero código existente quebra** após Phase 1.
2. A camada de adapter foi entregue sem quebrar os comandos principais.
3. A configuração e o transporte MCP mudaram depois desta migração: o contrato
   atual é `~/.stropha/config.yaml` + daemon HTTP compartilhado.
4. Índices existentes avançam somente por migrations idempotentes do storage
   adapter ativo; SQLite e PostgreSQL mantêm caminhos próprios.
5. CLI commands (`stropha index`, `stropha search`, `stropha stats`) preservam exit codes, flags e output format.
6. Tests existentes (52 passando) continuam passando.

### 13.2 Não-garantias

1. Performance pode mudar levemente (overhead de pipeline assembly: ~50ms no startup).
2. Logs podem ter novos eventos (`pipeline.stage.start`, `pipeline.stage.done`).
3. `stats` output ganha seções (adapter atual por stage) — campos antigos preservados.

---

## 14. Estratégia de testes

### 14.1 Unit

- Stage protocol contract test (cada adapter satisfaz contract)
- Registry: register / lookup / unknown name → ConfigError
- Config loader: `~/.stropha/config.yaml`, CLI override e precedência
- Runtime-only env registry: validate the canonical `_ENV_REGISTRY`
- Validation: pydantic errors estruturados
- Cross-stage consistency: embedder.dim ↔ storage.embedding_dim
- Adapter health: ready/warning/error com mensagens
- Adapter ID derivation: ollama com 2 modelos diferentes → 2 ids

### 14.2 Integração

- Pipeline end-to-end com cada combinação relevante de adapters
- Adapter swap: trocar enricher, rodar index, verificar re-processamento correto
- Cache: hit/miss/invalidation
- Multi-repo: 2 repos com adapters diferentes (não vai acontecer porque single-pipeline, mas testa que adapter_id discrimina)

### 14.3 Smoke

- `stropha pipeline show` produz output válido
- `stropha pipeline validate` retorna exit 0 quando tudo ready
- `stropha adapters list` lista todos os registrados
- `stropha adapters info ollama` mostra schema

### 14.4 Regression

- Suíte atual (52 testes) roda sem modificação após Phase 1.
- Reindex do Mimoria pós-refactor produz mesma quantidade de chunks (4654) com mesmos chunk_ids (modulo enricher se mudou).

---

## 15. Observabilidade

### 15.1 Logging estruturado (structlog)

Novos eventos:

```
pipeline.assemble.start
pipeline.assemble.done                stages=6 duration_ms=…
pipeline.stage.start                  stage=enricher adapter=ollama
pipeline.stage.done                   stage=enricher adapter=ollama duration_ms=… items=…
pipeline.stage.health                 stage=enricher status=ready
pipeline.stage.error                  stage=enricher adapter=ollama error=…
config.load                           source=~/.stropha/config.yaml overrides=cli
adapter.registered                    stage=enricher name=hierarchical
adapter.config_invalid                stage=enricher key=timeout_s message=…
enricher.cache_hit                    content_hash=… adapter_id=…
enricher.cache_miss                   content_hash=… adapter_id=…
```

### 15.2 Métricas (futuro Phase 2 da spec mestre)

```
stropha_stage_duration_seconds{stage, adapter, quantile}
stropha_stage_errors_total{stage, adapter, reason}
stropha_enricher_cache_hit_ratio
stropha_adapter_health_status{stage, adapter, status}
```

---

## 16. Riscos

| Risco | Probabilidade | Impacto | Mitigação |
|---|---|---|---|
| Refactor introduz regressão silenciosa | Médio | Alto | Suíte de regression test executada em CI; Phase 1 não toca walker/chunker/storage/retrieval |
| Config YAML divergente entre máquinas | Médio | Médio | Inspecionar a fonte única com `get_config` / `stropha pipeline show` |
| Auto-import importa modules pesados antes de necessário | Baixo | Médio | Adapter `__init__` é cheap (só decorator); imports pesados (ONNX, requests) ficam lazy nos métodos |
| Adapter de terceiro com bug derruba pipeline inteira | N/A (v1) | N/A | Plugin externo fora de escopo |
| Schema migration v2 → v3 falha em DB legado | Baixo | Alto | Migration idempotente; ALTER TABLE com guard; backup automático sugerido antes de upgrade |
| Health check timeout bloqueia `stropha index` | Médio | Baixo | Health timeout default 2s; falha não bloqueia, apenas warning |
| Cache de enrichment cresce sem limite | Médio | Baixo | TTL futuro / VACUUM periódico; índice em `(content_hash, adapter_id)` mantém lookup O(1) |
| Naming colisão entre adapters (dois `ollama` em diferentes stages) | Baixo | Baixo | Registry chave é `(stage, name)`; sem ambiguidade |
| Documentação fica obsoleta vs código | Alto | Médio | `stropha adapters info <name>` gera doc a partir do schema pydantic (dinâmico) |

---

## 17. Referências

### Documentos internos

- `docs/architecture/stropha-system.md` — spec mestre
- `docs/architecture/stropha-graphify-integration.md` — integração graphify
- `CLAUDE.md` — guia de agentes

### Padrões e práticas

- Strategy pattern (Gamma et al., *Design Patterns*, 1994)
- Pydantic Settings — https://docs.pydantic.dev/latest/concepts/pydantic_settings/
- YAML 1.2 spec — https://yaml.org/spec/1.2.2/
- 12-factor app config — https://12factor.net/config

### Inspirações de arquitetura

- LangChain `Runnable` interface
- Haystack 2.x pipelines
- LlamaIndex `IngestionPipeline`
- dbt project configuration
- Kubernetes Custom Resource Definitions (CRDs)

---

## 18. Glossário

| Termo | Definição |
|---|---|
| **Stage** | Uma responsabilidade discreta da pipeline (walker, chunker, enricher, embedder, storage, retrieval) |
| **Adapter** | Implementação concreta de uma `Stage`. Selecionada por config. |
| **adapter_id** | Identificador canônico do adapter incluindo modelo/versão (ex: `ollama:qwen2.5-coder:14b`). Usado como cache key. |
| **adapter_name** | Identificador curto do adapter, único por stage (ex: `ollama`). Usado em YAML/env. |
| **Sub-pipeline** | Stage cujo adapter coordena sub-adapters (ex: `chunker.tree-sitter-dispatch` dispatcha pra adapters por linguagem). |
| **Registry** | Mapeamento `(stage, adapter_name)` → classe. Populado por auto-import + decorator. |
| **Builder** | Função que pega config + registry e produz Pipeline instanciado pronto pra rodar. |
| **Drift** | Discrepância entre adapter_id atual e o que está persistido em DB. Aciona re-processamento. |
| **Translation layer** | Componente que mapeia env vars legadas pra novas chaves de config. |

---

## 19. Apêndice — exemplos de YAML

> **Historical RFC examples, not runnable configuration.** Adapter names,
> nesting, and secret injection in these sketches predate the canonical config
> schema. Generate current configuration from `~/.stropha/config.yaml`, CLI
> help, and adapter introspection rather than copying these blocks.

### 19.1 Stack 100% local

```yaml
version: 1
pipeline:
  walker:    { adapter: git-ls-files }
  chunker:   { adapter: tree-sitter-dispatch }
  enricher:  { adapter: hierarchical }
  embedder:
    adapter: local
    config: { model: mixedbread-ai/mxbai-embed-large-v1 }
  storage:
    adapter: sqlite-vec
    config: { path: ~/.stropha/index.db }
  retrieval: { adapter: hybrid-rrf }
```

### 19.2 Stack com LLM enrichment local

```yaml
version: 1
pipeline:
  walker:    { adapter: git-ls-files }
  chunker:   { adapter: tree-sitter-dispatch }
  enricher:
    adapter: ollama
    config:
      model: qwen2.5-coder:14b
      host: http://localhost:11434
      timeout_s: 30
  embedder:
    adapter: local
    config: { model: mixedbread-ai/mxbai-embed-large-v1 }
  storage:
    adapter: sqlite-vec
    config: { path: ~/.stropha/index.db }
  retrieval: { adapter: hybrid-rrf }
```

### 19.3 Stack cloud (Voyage embedder + Anthropic enricher)

```yaml
version: 1
secrets:
  anthropic_api_key: <key>
  voyage_api_key: <key>
pipeline:
  walker:    { adapter: git-ls-files }
  chunker:   { adapter: tree-sitter-dispatch }
  enricher:
    adapter: anthropic
    config:
      model: claude-haiku-4-5
  embedder:
    adapter: voyage
    config:
      model: voyage-code-3
      dim: 512
  storage:
    adapter: sqlite-vec
    config: { path: ~/.stropha/index.db }
  retrieval:
    adapter: hybrid-rrf-rerank          # Phase 2 future
    config:
      reranker:
        adapter: voyage-rerank
        config: { model: rerank-2.5 }
```

### 19.4 Stack baseline (apenas BM25, sem embedder)

Util pra repos pequenos onde lexical é suficiente:

```yaml
version: 1
pipeline:
  walker:    { adapter: git-ls-files }
  chunker:   { adapter: tree-sitter-dispatch }
  enricher:  { adapter: noop }
  embedder:  { adapter: noop }            # special: no-op embedder (returns zeros)
  storage:
    adapter: sqlite-vec
    config:
      path: ~/.stropha/index.db
      skip_vec_table: true                # don't create vec_chunks
  retrieval: { adapter: bm25-only }
```

---

## §A — Appendix: Adapters concretos registrados (audited delivery snapshot)

> Audited 2026-08-11 at commit `b0d391d7d67606a04666d8e9a48cfbef24c1726d`.
> This is a historical snapshot, not a timeless inventory. Use
> `stropha adapters list`, `stropha pipeline show`, or `get_config` for the live
> registry and active composition.

### Walker

| Adapter | Arquivo | Uso típico |
|---|---|---|
| `git-ls-files` (default) | `adapters/walker/git_ls_files.py` | Repos git padrão. Honra `.gitignore` + `.strophaignore`, fallback filesystem quando `git` falha. |
| `filesystem` | `adapters/walker/filesystem.py` | Diretórios sem `.git/` (downloads, vendored deps, snapshots). Skipa caches padrão (`.venv`, `node_modules`, `__pycache__`, `dist`, `build`, …). |
| `nested-git` | `adapters/walker/nested_git.py` | Monorepos com submódulos / vendored repos. Descobre `.git/` aninhados até `max_depth=4`, rebase paths para a raiz. |
| `git-diff` | `adapters/walker/git_diff.py` | Indexação incremental baseada em `git diff`. Detecta arquivos modificados/adicionados/renomeados entre commits. |

### Chunker and language chunkers

| Adapter | Stage | Arquivo | Uso |
|---|---|---|---|
| `tree-sitter-dispatch` (default) | `chunker` | `adapters/chunker/tree_sitter_dispatch.py` | Roteia por linguagem via `config.languages.<lang>.adapter`. |
| `ast-generic` | `language-chunker` | `adapters/chunker/languages/ast_generic.py` | tree-sitter para Java/TS/JS/Python/Rust/Go/Kotlin (via `tree-sitter-language-pack`). |
| `heading-split` | `language-chunker` | `adapters/chunker/languages/heading_split.py` | Markdown — split por heading. |
| `sfc-split` | `language-chunker` | `adapters/chunker/languages/sfc_split.py` | Vue Single-File Components (`<script>` / `<template>` / `<style>`). |
| `regex-feature-scenario` | `language-chunker` | `adapters/chunker/languages/regex_feature_scenario.py` | Gherkin `.feature` — split Feature/Scenario via regex (tree-sitter-language-pack não inclui gherkin). |
| `file-level` | `language-chunker` | `adapters/chunker/languages/file_level.py` | Fallback para linguagens não suportadas — um chunk por arquivo. |
| `yaml-structure` | `language-chunker` | `adapters/chunker/languages/yaml_structure.py` | YAML estrutural por mappings/sequences, preservando paths de chave. |

### Enricher

| Adapter | Arquivo | Custo | Quando usar |
|---|---|---|---|
| `noop` (default) | `adapters/enricher/noop.py` | zero | Baseline; `embedding_text == content`. |
| `hierarchical` | `adapters/enricher/hierarchical.py` | zero | Prepend skeleton do chunk pai (recupera contexto class → method). |
| `graph-aware` | `adapters/enricher/graph_aware.py` | zero (precisa de Storage injetada) | Prepend community label + node label do graphify. Boost de recall BM25/FTS lexical sem custo de embedding (L2). |
| `ollama` | `adapters/enricher/ollama.py` | LLM local | One-line summary via Ollama HTTP (`qwen2.5-coder:1.5b`). Fail-graceful. |
| `mlx` | `adapters/enricher/mlx.py` | LLM local | Mesma proposta do `ollama` mas via `mlx-lm` nativo Apple Silicon. Optional dep `[mlx]`. Lazy-load. |
| `contextual` | `adapters/enricher/contextual.py` | backend de inference configurado | Contextual retrieval batched com identidade de prompt/config no `adapter_id`. |

### Embedder

| Adapter | Arquivo | Modelo padrão | Cost / network |
|---|---|---|---|
| `local` (default) | `adapters/embedder/local.py` | `mixedbread-ai/mxbai-embed-large-v1` @ 1024d | local ONNX, ~1.7 GB on disk |
| `voyage` | `adapters/embedder/voyage.py` | `voyage-code-3` @ 512d (Matryoshka) | cloud, requires `secrets.voyage_api_key` |
| `bge-m3` | `adapters/embedder/bge_m3.py` | `BAAI/bge-m3` @ 1024d | local Apple Silicon via `[mlx-embed]`, multilingual |
| `mlx` | `adapters/embedder/mlx.py` | registry MLX or Qwen3-Embedding | local Apple Silicon via `mlx-embedding-models` or `mlx-lm`; optional dep `[mlx-embed]` |
| `nvidia` | `adapters/embedder/hardware.py` | configurável (SentenceTransformers) | native CUDA / PyTorch engine |
| `amd` | `adapters/embedder/hardware.py` | configurável (SentenceTransformers) | native ROCm / PyTorch engine |
| `intel-npu` | `adapters/embedder/hardware.py` | configurável (SentenceTransformers) | native OpenVINO NPU engine |
| `model2vec` | `adapters/embedder/model2vec.py` | `minishlab/potion-code-16M` @ 256d | local static lookup; primary or optional secondary lane |

### Storage

| Adapter | Arquivo | Notas |
|---|---|---|
| `sqlite-vec` (default) | `adapters/storage/sqlite_vec.py` | Subclass do `stropha.storage.Storage` legacy — herda a superfície read/write completa. Schema v1→v14. |
| `postgres` | `adapters/storage/postgres.py` | pgvector com paridade de schema v14. Mantém uma conexão writer dedicada para transactions/advisory lock e um pool bounded independente para retrieval reads. HNSW usa `vector_l2_ops` até 2000d e `halfvec_l2_ops` entre 2001d e 4000d. |

### Retrieval coordinator

| Adapter | Arquivo | Notas |
|---|---|---|
| `hybrid-rrf` (default) | `adapters/retrieval/hybrid_rrf.py` | RRF (k=60) sobre N streams configuráveis em `config.streams.{name: {adapter, config}}`. Preserva ordem determinística ao executar streams/paráfrases com concorrência limitada quando storage e streams declaram reads concorrentes seguros. Skipa `embed_query` sem stream dense e propaga `repo_id` em todos os caminhos. |

### Retrieval stream sub-adapters

| Adapter | Arquivo | Lane semântica |
|---|---|---|
| `vec-cosine` (default composition) | `adapters/retrieval/streams/vec_cosine.py` | Dense retrieval delegated to the active storage adapter. |
| `fts5-bm25` (default composition) | `adapters/retrieval/streams/fts5_bm25.py` | Sparse BM25 delegated to SQLite FTS5 or PostgreSQL `chunks.fts`. |
| `like-tokens` (default) | `adapters/retrieval/streams/like_tokens.py` | Symbol-token `LIKE` match — identificadores. |
| `graph-vec` (opt-in) | `adapters/retrieval/streams/graph_vec.py` | Cosine sobre `graph_nodes.embedding`; retorna vazio quando o lane não foi materializado. |
| `model2vec-cosine` (opt-in) | `adapters/retrieval/streams/model2vec_cosine.py` | ANN sobre o espaço secundário 256-d; embebe a query literal reescrita com a mesma instância Model2Vec usada no backfill. |

The built-in default composition enables `vec-cosine`, `fts5-bm25`, and
`like-tokens`. Registered stream count is not the active-stream count; optional
`graph-vec` and `model2vec-cosine` lanes appear only when configured.

`vec-cosine`, `fts5-bm25`, and `like-tokens` are safe to fan out when backed by
PostgreSQL's read pool. SQLite remains serial because it owns one thread-bound
connection. `graph-vec` stays serial while it uses the mixed read/write `_conn`
compatibility shim, and `model2vec-cosine` stays serial because query embedding
thread safety is not part of the secondary embedder contract.

### Reranker

| Adapter | Arquivo |
|---|---|
| `noop` | `adapters/retrieval/reranker/noop.py` |
| `cross-encoder` | `adapters/retrieval/reranker/cross_encoder.py` |
| `code-aware` | `adapters/retrieval/reranker/code_aware.py` |
| `llm` | `adapters/retrieval/reranker/llm.py` |

### Router

| Adapter | Arquivo |
|---|---|
| `noop` | `adapters/router/noop.py` |
| `pattern` | `adapters/router/pattern.py` |
| `llm` | `adapters/router/llm.py` |
| `escalating` | `adapters/router/escalating.py` |

O contrato interno de retrieval usa `repo_id: int | None` keyword-only. ID
concreto restringe dense, BM25, symbol, Model2Vec e GraphVec antes do budget de
candidatos; `None` reserva-se à busca all-repo explicitamente solicitada pelas
fronteiras públicas. MCP, CLI e frontend resolvem o repositório corrente por
padrão e falham fechados se ele não estiver registrado.

Adapters de retrieval/storage usados por buscas scoped devem implementar o
keyword `repo_id` e filtrar antes de ranking/`LIMIT`. Para compatibilidade com
adapters antigos, chamadas internas broad (`repo_id=None`) omitem o keyword onde
prático; chamadas com ID concreto nunca fazem fallback sem escopo.

`BuiltStages.secondary_embedder` é `None` quando o stream não está configurado.
Quando habilitado, o builder cria uma única instância Model2Vec, valida
`dim=256` + `normalize=true`, injeta a mesma instância no `Pipeline` e no
stream e inclui seu `adapter_id` no `indexing_hash`. O loader secundário roda
depois do commit dos chunks primários e antes do lifecycle graphify. Schema
v14 mantém metadados `content_hash` + `embedder_id` junto ao espaço vetorial
secundário; drift ou interrupção causam backfill incremental sem tocar os
vetores primários. O stream permanece opt-in: no golden set Stropha (n=52),
Recall@10 ficou estável em 0.8462, enquanto MRR caiu de 0.6220 para 0.5586 e
latência p50 subiu de 78.7 ms para 90.8 ms.

### Quick-reference dos adapter_id shapes (drift detection)

| Stage | Shape | Exemplo |
|---|---|---|
| walker | `<name>:max=<bytes>[:depth=<n>]` | `git-ls-files:max=524288` |
| chunker | `tree-sitter-dispatch:<hash>` | `tree-sitter-dispatch:2a342cf6` |
| enricher | `<name>:<flags>` | `graph-aware:c:n:p` |
| embedder | `<name>:<model>:<dim>[:flags]` | `model2vec:minishlab/potion-code-16M:256:normalize=true` |
| storage | `sqlite-vec:dim=<n>` | `sqlite-vec:dim=1024` |
| retrieval | `hybrid-rrf:k=<n>:streams=<hash>` | `hybrid-rrf:k=60:streams=35c9b110` |
| retrieval-stream | `<name>:k=<n>[:min=<sim>]` | `graph-vec:k=20:min=0.3` |

A mudança de qualquer caractere de um `adapter_id` invalida a freshness check e força re-process dos rows afetados — esse é o contrato de drift detection da ADR-004.

---

**Fim do documento.**
