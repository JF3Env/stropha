"""Structure-aware YAML chunker using PyYAML node source marks."""

from __future__ import annotations

from collections.abc import Iterator

import yaml  # type: ignore[import-untyped]
from pydantic import BaseModel, Field
from yaml.nodes import MappingNode, Node, ScalarNode, SequenceNode  # type: ignore[import-untyped]

from ....ingest.chunkers.base import (
    MAX_CHARS_PER_CHUNK,
    MIN_CHARS_PER_CHUNK,
    make_chunk_id,
    sha256_hex,
    slice_lines,
)
from ....ingest.chunkers.fallback import FallbackChunker
from ....models import Chunk, SourceFile
from ....pipeline.base import StageHealth
from ....pipeline.registry import register_adapter


class YamlStructureConfig(BaseModel):
    max_chars_per_chunk: int = Field(default=8_000, ge=500, le=MAX_CHARS_PER_CHUNK)


def _line_span(node: Node) -> tuple[int, int]:
    start = node.start_mark.line + 1
    end = node.end_mark.line + (1 if node.end_mark.column else 0)
    return start, max(start, end)


def _mapping_key(node: Node) -> str | None:
    return node.value if isinstance(node, ScalarNode) else None


def _mapping_value(node: MappingNode, key: str) -> Node | None:
    for key_node, value_node in node.value:
        if _mapping_key(key_node) == key:
            return value_node
    return None


def _scalar_value(node: Node | None) -> str | None:
    return node.value if isinstance(node, ScalarNode) else None


def _document_symbol(node: Node, document_number: int) -> str:
    if not isinstance(node, MappingNode):
        return f"document[{document_number}]"
    kind = _scalar_value(_mapping_value(node, "kind"))
    metadata = _mapping_value(node, "metadata")
    name = (
        _scalar_value(_mapping_value(metadata, "name"))
        if isinstance(metadata, MappingNode)
        else None
    )
    if kind and name:
        return f"{kind}/{name}"
    return kind or name or f"document[{document_number}]"


def _children(node: Node, path: str) -> list[tuple[Node, str, int | None]]:
    if isinstance(node, MappingNode):
        result: list[tuple[Node, str, int | None]] = []
        for index, (key_node, value_node) in enumerate(node.value):
            key = _mapping_key(key_node) or f"key[{index}]"
            child_path = f"{path}.{key}" if path else key
            # Include the key line so each mapping chunk remains meaningful YAML.
            result.append((value_node, child_path, key_node.start_mark.line + 1))
        return result
    if isinstance(node, SequenceNode):
        return [
            (child, f"{path}[{index}]", None) for index, child in enumerate(node.value)
        ]
    return []


def _partition(
    node: Node,
    content: str,
    path: str,
    limit: int,
    start_override: int | None = None,
) -> Iterator[tuple[int, int, str]]:
    start, end = _line_span(node)
    start = start_override or start
    if len(slice_lines(content, start, end)) <= limit:
        yield start, end, path
        return

    children = _children(node, path)
    if not children:
        yield start, end, path
        return

    group_start: int | None = None
    group_end = 0
    group_paths: list[str] = []
    for child, child_path, child_start_override in children:
        child_start, child_end = _line_span(child)
        child_start = child_start_override or child_start
        child_text = slice_lines(content, child_start, child_end)
        if len(child_text) > limit:
            if group_start is not None:
                yield group_start, group_end, path if len(group_paths) > 1 else group_paths[0]
                group_start = None
                group_paths = []
            yield from _partition(child, content, child_path, limit, child_start_override)
            continue

        candidate_start = group_start if group_start is not None else child_start
        if group_start is not None and len(slice_lines(content, candidate_start, child_end)) > limit:
            yield group_start, group_end, path if len(group_paths) > 1 else group_paths[0]
            group_start = child_start
            group_paths = [child_path]
        else:
            group_start = candidate_start
            group_paths.append(child_path)
        group_end = child_end

    if group_start is not None:
        yield group_start, group_end, path if len(group_paths) > 1 else group_paths[0]


@register_adapter(stage="language-chunker", name="yaml-structure")
class YamlStructureLanguageChunker:
    Config = YamlStructureConfig

    def __init__(self, config: YamlStructureConfig | None = None) -> None:
        self._config = config or YamlStructureConfig()

    @property
    def stage_name(self) -> str: return "language-chunker"

    @property
    def adapter_name(self) -> str: return "yaml-structure"

    @property
    def adapter_id(self) -> str:
        return f"yaml-structure:{self._config.max_chars_per_chunk}"

    @property
    def config_schema(self) -> type[BaseModel]: return YamlStructureConfig

    def health(self) -> StageHealth:
        return StageHealth(status="ready", message="YAML document and node boundary split")

    def chunk(self, file: SourceFile, content: str) -> Iterator[Chunk]:
        try:
            documents = [node for node in yaml.compose_all(content) if node is not None]
        except yaml.YAMLError:
            yield from FallbackChunker().chunk(file, content)
            return

        if not documents:
            yield from FallbackChunker().chunk(file, content)
            return

        for document_number, node in enumerate(documents, start=1):
            symbol = _document_symbol(node, document_number)
            for start, end, path in _partition(
                node, content, symbol, self._config.max_chars_per_chunk
            ):
                text = slice_lines(content, start, end)
                if len(text.strip()) < MIN_CHARS_PER_CHUNK:
                    continue
                content_hash = sha256_hex(text)
                yield Chunk(
                    chunk_id=make_chunk_id(file.rel_path, start, end, content_hash),
                    rel_path=file.rel_path,
                    language=file.language,
                    kind="yaml_document" if path == symbol else "yaml_section",
                    symbol=path,
                    start_line=start,
                    end_line=end,
                    content=text,
                    content_hash=content_hash,
                )
