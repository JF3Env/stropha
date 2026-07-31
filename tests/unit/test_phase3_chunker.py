"""Tests for Phase 3 chunker dispatcher and language sub-adapters."""

from __future__ import annotations

from pathlib import Path

import pytest

from stropha import adapters  # noqa: F401
from stropha.adapters.chunker.languages.file_level import FileLevelLanguageChunker
from stropha.adapters.chunker.languages.heading_split import (
    HeadingSplitLanguageChunker,
)
from stropha.adapters.chunker.languages.yaml_structure import (
    YamlStructureConfig,
    YamlStructureLanguageChunker,
)
from stropha.adapters.chunker.tree_sitter_dispatch import (
    TreeSitterDispatchChunker,
    TreeSitterDispatchConfig,
)
from stropha.errors import ChunkerError
from stropha.ingest.chunker import Chunker as LegacyChunker
from stropha.models import SourceFile
from stropha.pipeline.registry import all_adapters

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def py_file(tmp_path: Path) -> SourceFile:
    """Long enough (>15 lines) to bypass AST chunker's SIMPLE_FILE shortcut."""
    p = tmp_path / "main.py"
    body = "\n".join(
        [
            '"""A small module to chunk."""',
            "",
            "class Greeter:",
            "    def __init__(self, name: str) -> None:",
            "        self.name = name",
            "",
            "    def greet(self) -> str:",
            "        return f'hello, {self.name}'",
            "",
            "    def shout(self) -> str:",
            "        return self.greet().upper()",
            "",
            "",
            "def make_greeter(name: str) -> Greeter:",
            "    return Greeter(name)",
            "",
            "",
            "def main() -> None:",
            "    print(make_greeter('world').greet())",
            "",
        ]
    )
    p.write_text(body, encoding="utf-8")
    return SourceFile(path=p, rel_path="main.py", language="python", size_bytes=p.stat().st_size)


@pytest.fixture
def md_file(tmp_path: Path) -> SourceFile:
    p = tmp_path / "README.md"
    p.write_text(
        "# Title\n\nIntro paragraph.\n\n## Section A\n\nA stuff.\n\n## Section B\n\nB stuff.\n",
        encoding="utf-8",
    )
    return SourceFile(path=p, rel_path="README.md", language="markdown", size_bytes=p.stat().st_size)


@pytest.fixture
def txt_file(tmp_path: Path) -> SourceFile:
    p = tmp_path / "notes.txt"
    p.write_text("plain text\nanother line\n", encoding="utf-8")
    return SourceFile(path=p, rel_path="notes.txt", language="text", size_bytes=p.stat().st_size)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


def test_chunker_and_language_chunker_stages_registered() -> None:
    reg = all_adapters()
    assert "tree-sitter-dispatch" in reg["chunker"]
    for name in (
        "ast-generic",
        "heading-split",
        "sfc-split",
        "regex-feature-scenario",
        "yaml-structure",
        "file-level",
    ):
        assert name in reg["language-chunker"], f"missing {name}"


# ---------------------------------------------------------------------------
# Sub-adapter contracts
# ---------------------------------------------------------------------------


def test_heading_split_sub_adapter(md_file: SourceFile) -> None:
    sub = HeadingSplitLanguageChunker()
    chunks = list(sub.chunk(md_file, md_file.path.read_text()))
    # Two H2 sections + intro/title material → at least 2 chunks.
    assert len(chunks) >= 2
    assert sub.adapter_name == "heading-split"
    assert sub.health().status == "ready"


def test_file_level_sub_adapter(txt_file: SourceFile) -> None:
    sub = FileLevelLanguageChunker()
    chunks = list(sub.chunk(txt_file, txt_file.path.read_text()))
    assert len(chunks) == 1
    assert chunks[0].kind == "file"


def test_yaml_sub_adapter_splits_documents_and_names_resources(tmp_path: Path) -> None:
    content = """apiVersion: v1
kind: ConfigMap
metadata:
  name: app-config
data:
  LOG_LEVEL: info
---
apiVersion: apps/v1
kind: Deployment
metadata:
  name: api
spec:
  replicas: 2
"""
    path = tmp_path / "resources.yml"
    path.write_text(content, encoding="utf-8")
    source = SourceFile(
        path=path, rel_path="resources.yml", language="yaml", size_bytes=len(content)
    )

    chunks = list(YamlStructureLanguageChunker().chunk(source, content))

    assert [chunk.symbol for chunk in chunks] == ["ConfigMap/app-config", "Deployment/api"]
    assert all(chunk.kind == "yaml_document" for chunk in chunks)
    assert chunks[0].end_line < chunks[1].start_line


def test_yaml_sub_adapter_splits_large_mapping_on_keys(tmp_path: Path) -> None:
    content = "metadata:\n  name: service\nspec:\n" + "".join(
        f"  setting_{index}: {'x' * 80}\n" for index in range(20)
    )
    path = tmp_path / "values.yaml"
    path.write_text(content, encoding="utf-8")
    source = SourceFile(
        path=path, rel_path="values.yaml", language="yaml", size_bytes=len(content)
    )

    chunker = YamlStructureLanguageChunker(YamlStructureConfig(max_chars_per_chunk=500))
    chunks = list(chunker.chunk(source, content))

    assert len(chunks) > 1
    assert all(chunk.kind == "yaml_section" for chunk in chunks)
    assert any((chunk.symbol or "").endswith(".spec") for chunk in chunks)
    assert all(len(chunk.content) <= 500 for chunk in chunks)


def test_yaml_sub_adapter_falls_back_for_invalid_yaml(tmp_path: Path) -> None:
    content = "metadata:\n  name: valid prefix\ninvalid: [unterminated\n"
    path = tmp_path / "broken.yaml"
    path.write_text(content, encoding="utf-8")
    source = SourceFile(
        path=path, rel_path="broken.yaml", language="yaml", size_bytes=len(content)
    )

    chunks = list(YamlStructureLanguageChunker().chunk(source, content))

    assert len(chunks) == 1
    assert chunks[0].kind == "file"


def test_yaml_sub_adapter_preserves_comment_only_files(tmp_path: Path) -> None:
    content = "# Deployment values and operational guidance\n"
    path = tmp_path / "comments.yaml"
    path.write_text(content, encoding="utf-8")
    source = SourceFile(
        path=path, rel_path="comments.yaml", language="yaml", size_bytes=len(content)
    )

    chunks = list(YamlStructureLanguageChunker().chunk(source, content))

    assert len(chunks) == 1
    assert chunks[0].content == content


# ---------------------------------------------------------------------------
# Dispatcher behavior
# ---------------------------------------------------------------------------


def test_dispatcher_routes_to_python_ast(py_file: SourceFile) -> None:
    chunker = TreeSitterDispatchChunker()
    chunks = list(chunker.chunk([py_file]))
    # File is >15 lines so AST decomposition kicks in; expect class + methods.
    kinds = {c.kind for c in chunks}
    assert "class" in kinds or "method" in kinds or "function" in kinds, (
        f"expected AST kinds, got {kinds}"
    )
    symbols = {c.symbol for c in chunks if c.symbol}
    # At least one of the known symbols must surface.
    assert symbols & {"Greeter", "Greeter.greet", "main", "make_greeter"}


def test_dispatcher_falls_back_to_file_level_for_unknown_language(
    txt_file: SourceFile,
) -> None:
    chunker = TreeSitterDispatchChunker()
    chunks = list(chunker.chunk([txt_file]))
    assert len(chunks) >= 1
    # Plain text -> file-level fallback only.
    assert all(c.kind == "file" for c in chunks)


def test_dispatcher_adapter_id_changes_with_language_override() -> None:
    default = TreeSitterDispatchChunker().adapter_id
    override = TreeSitterDispatchChunker(
        TreeSitterDispatchConfig(
            languages={"markdown": {"adapter": "file-level"}}
        )
    ).adapter_id
    assert default != override


def test_dispatcher_adapter_id_includes_sub_adapter_config() -> None:
    compact = TreeSitterDispatchChunker(
        TreeSitterDispatchConfig(
            languages={
                "yaml": {
                    "adapter": "yaml-structure",
                    "config": {"max_chars_per_chunk": 4_000},
                }
            }
        )
    ).adapter_id
    broad = TreeSitterDispatchChunker(
        TreeSitterDispatchConfig(
            languages={
                "yaml": {
                    "adapter": "yaml-structure",
                    "config": {"max_chars_per_chunk": 8_000},
                }
            }
        )
    ).adapter_id

    assert compact != broad


def test_dispatcher_unknown_sub_adapter_raises() -> None:
    from stropha.errors import ConfigError

    with pytest.raises(ConfigError, match="Unknown language-chunker adapter"):
        TreeSitterDispatchChunker(
            TreeSitterDispatchConfig(
                languages={"python": {"adapter": "no-such-sub"}}
            )
        )


def test_dispatcher_health_reports_sub_adapter_count() -> None:
    chunker = TreeSitterDispatchChunker()
    h = chunker.health()
    assert h.status == "ready"
    # 11 default languages + _fallback = 12 entries.
    assert "12" in h.message


# ---------------------------------------------------------------------------
# Output parity vs legacy Chunker
# ---------------------------------------------------------------------------


def test_dispatcher_output_matches_legacy_chunker_for_python(
    py_file: SourceFile,
) -> None:
    """Phase 3 must NOT change chunk_id of unchanged files (no spurious re-embed)."""
    legacy = list(LegacyChunker().chunk([py_file], repo_key="test-repo"))
    new = list(TreeSitterDispatchChunker().chunk([py_file], repo_key="test-repo"))
    assert len(legacy) == len(new)
    legacy_ids = sorted(c.chunk_id for c in legacy)
    new_ids = sorted(c.chunk_id for c in new)
    assert legacy_ids == new_ids


def test_dispatcher_output_matches_legacy_chunker_for_markdown(
    md_file: SourceFile,
) -> None:
    legacy = list(LegacyChunker().chunk([md_file], repo_key="test-repo"))
    new = list(TreeSitterDispatchChunker().chunk([md_file], repo_key="test-repo"))
    legacy_ids = sorted(c.chunk_id for c in legacy)
    new_ids = sorted(c.chunk_id for c in new)
    assert legacy_ids == new_ids


def test_dispatcher_unreadable_file_raises_chunker_error(tmp_path: Path) -> None:
    sf = SourceFile(
        path=tmp_path / "does-not-exist.py",
        rel_path="does-not-exist.py",
        language="python",
        size_bytes=0,
    )
    chunker = TreeSitterDispatchChunker()
    with pytest.raises(ChunkerError):
        list(chunker.chunk([sf]))


# ---------------------------------------------------------------------------
# Parent linkage under repo-key namespacing
#
# Regression: ``_with_repo_key`` rewrote ``chunk_id`` with the repo_key prefix
# but left ``parent_chunk_id`` pointing at the pre-namespacing id, so every
# child's parent was dangling. That silently turned the hierarchical enricher
# (``ctx.parent_chunk`` always None) and recursive-merge parent promotion into
# no-ops whenever a repo_key was set — i.e. in all real indexing.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("repo_key", [None, "github.com/acme/widget"])
def test_parent_chunk_id_resolves(py_file: SourceFile, repo_key: str | None) -> None:
    chunks = list(TreeSitterDispatchChunker().chunk([py_file], repo_key=repo_key))
    ids = {c.chunk_id for c in chunks}
    parented = [c for c in chunks if c.parent_chunk_id]
    # Greeter's methods carry a parent — otherwise the assertion is vacuous.
    assert parented, "expected at least one chunk with a parent_chunk_id"
    dangling = [c for c in parented if c.parent_chunk_id not in ids]
    assert not dangling, (
        f"{len(dangling)} chunk(s) have a dangling parent_chunk_id "
        f"(repo_key={repo_key!r}): "
        f"{[(c.symbol, c.parent_chunk_id) for c in dangling]}"
    )


def test_parent_chunk_id_points_at_enclosing_class_with_repo_key(
    py_file: SourceFile,
) -> None:
    chunks = list(
        TreeSitterDispatchChunker().chunk([py_file], repo_key="github.com/acme/widget")
    )
    by_id = {c.chunk_id: c for c in chunks}
    methods = [c for c in chunks if c.symbol and c.symbol.startswith("Greeter.")]
    assert methods, "expected Greeter.* method chunks"
    for m in methods:
        assert m.parent_chunk_id in by_id, f"{m.symbol} has a dangling parent"
        assert by_id[m.parent_chunk_id].symbol == "Greeter"
