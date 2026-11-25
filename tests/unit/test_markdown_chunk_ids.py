"""Markdown chunk ids stay unique when a split part matches a child section."""

from __future__ import annotations

from collections import Counter
from pathlib import Path

from stropha.ingest.chunkers.base import MAX_CHARS_PER_CHUNK
from stropha.ingest.chunkers.markdown import MarkdownChunker
from stropha.models import SourceFile


def _wide_section_whose_last_part_is_a_child() -> str:
    """An oversized `#` section that splits exactly at its `##` child.

    The parent is too long for one chunk, so it is split on line boundaries.
    The filler ends a few characters short of the limit, which puts the cut
    right before the child heading: the last split part then covers the same
    lines, with the same text, as the child section itself.
    """
    heading = "# Wide section\n"
    budget = MAX_CHARS_PER_CHUNK - 5 - len(heading)
    full_lines, rest = divmod(budget, 100)
    filler = ("x" * 99 + "\n") * full_lines + "y" * (rest - 1) + "\n"
    child = "## Child section\n" + "The child section has enough text to be indexed.\n"
    return heading + filler + child


def test_split_part_and_child_section_do_not_share_a_chunk_id() -> None:
    content = _wide_section_whose_last_part_is_a_child()
    file = SourceFile(
        path=Path("doc.md"), rel_path="doc.md", language="markdown", size_bytes=len(content)
    )

    chunks = list(MarkdownChunker().chunk(file, content))

    duplicates = [cid for cid, n in Counter(c.chunk_id for c in chunks).items() if n > 1]
    assert duplicates == []
    child_lines = content.count("\n") - 1
    assert any(c.start_line == child_lines for c in chunks)
