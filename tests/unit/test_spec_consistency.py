"""Unit tests for spec-consistency check + MCP wrapper
(WIN-250, EPIC 1.11 #88)."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from stropha.retrieval.spec_consistency import (
    SpecConflict,
    SpecConsistencyReport,
    SpecInvariant,
    check_spec_consistency,
    extract_invariants,
)


# ---------------------------------------------------------------------------
# Dataclass surface
# ---------------------------------------------------------------------------


class TestDataclasses:
    def test_invariant_required_fields(self) -> None:
        inv = SpecInvariant(
            text="x", spec_file="f", line=1, keyword="MUST", severity="HIGH",
        )
        assert inv.severity == "HIGH"

    def test_conflict_defaults(self) -> None:
        inv = SpecInvariant(
            text="x", spec_file="f", line=1, keyword="MUST", severity="HIGH",
        )
        c = SpecConflict(invariant=inv, matched="Foo", severity="HIGH")
        assert c.reason == ""

    def test_report_defaults(self) -> None:
        r = SpecConsistencyReport()
        assert r.invariants_found == 0
        assert r.conflicts == []
        assert r.notes == []
        assert r.spec_files_scanned == []

    def test_frozen(self) -> None:
        inv = SpecInvariant(
            text="x", spec_file="f", line=1, keyword="MUST", severity="HIGH",
        )
        with pytest.raises((AttributeError, Exception)):
            inv.severity = "LOW"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# extract_invariants
# ---------------------------------------------------------------------------


class TestExtractInvariants:
    def test_empty_text(self) -> None:
        assert extract_invariants("") == []

    def test_must_detected(self) -> None:
        out = extract_invariants("Frobnicators MUST validate input.")
        assert len(out) == 1
        assert out[0].keyword == "MUST"
        assert out[0].severity == "HIGH"

    def test_never_detected(self) -> None:
        out = extract_invariants("We NEVER use globals.")
        assert out[0].keyword == "NEVER"
        assert out[0].severity == "HIGH"

    def test_forward_only_detected(self) -> None:
        out = extract_invariants("Schema migrations are forward-only.")
        assert out[0].keyword == "forward-only"

    def test_should_classified_medium(self) -> None:
        out = extract_invariants("New code should follow PEP-8.")
        assert out[0].severity == "MEDIUM"

    def test_avoid_classified_medium(self) -> None:
        out = extract_invariants("Avoid global state.")
        assert out[0].keyword == "avoid"
        assert out[0].severity == "MEDIUM"

    def test_note_classified_low(self) -> None:
        # Note: keyword detection uses the word at start of sentence;
        # "Note: x" treated as keyword "note:". Capital N also works
        # because matching is case-insensitive.
        out = extract_invariants("Note: be careful.")
        assert out[0].severity == "LOW"

    def test_non_invariant_skipped(self) -> None:
        text = "Hello world. This is regular text. Done."
        assert extract_invariants(text) == []

    def test_priority_must_over_should(self) -> None:
        # Sentence has both MUST and SHOULD — MUST wins because it
        # comes first in the keyword list.
        out = extract_invariants("Frobnicators MUST validate; they should log.")
        # Note: the splitter doesn't break on `;` so it's one sentence.
        assert out[0].keyword == "MUST"

    def test_line_number_captured(self) -> None:
        text = "First line.\n\nSecond line MUST do X."
        out = extract_invariants(text)
        # Second sentence on line 3.
        assert out[0].line == 3

    def test_spec_file_attached(self) -> None:
        out = extract_invariants("X MUST Y.", spec_file="CLAUDE.md")
        assert out[0].spec_file == "CLAUDE.md"


# ---------------------------------------------------------------------------
# check_spec_consistency
# ---------------------------------------------------------------------------


class TestCheckSpecConsistency:
    def test_no_specs(self) -> None:
        report = check_spec_consistency(
            ["Foo"], [],
            spec_loader=lambda: {},
        )
        assert report.invariants_found == 0
        assert report.conflicts == []

    def test_loader_exception_captured(self) -> None:
        def boom():
            raise RuntimeError("loader down")

        report = check_spec_consistency(
            ["Foo"], [],
            spec_loader=boom,
        )
        assert report.conflicts == []
        assert any("loader down" in n for n in report.notes)

    def test_symbol_match_high(self) -> None:
        specs = {"CLAUDE.md": "FooClass MUST validate every call."}
        report = check_spec_consistency(
            ["FooClass"], [],
            spec_loader=lambda: specs,
        )
        assert len(report.conflicts) == 1
        assert report.conflicts[0].severity == "HIGH"
        assert report.conflicts[0].matched == "FooClass"

    def test_file_match_basename(self) -> None:
        specs = {"CLAUDE.md": "server.py MUST exit in 100ms."}
        report = check_spec_consistency(
            [], ["src/stropha/server.py"],
            spec_loader=lambda: specs,
        )
        assert len(report.conflicts) >= 1
        # Either the full path or the basename matched.
        labels = {c.matched for c in report.conflicts}
        assert "server.py" in labels or "src/stropha/server.py" in labels

    def test_no_token_yields_note(self) -> None:
        report = check_spec_consistency(
            [], [],
            spec_loader=lambda: {"x.md": "MUST do something"},
        )
        assert any("No changed_symbols" in n for n in report.notes)

    def test_dedup_same_conflict(self) -> None:
        """Same invariant matching the same token twice (e.g., across
        symbol + file forms) collapses to one conflict."""
        specs = {"CLAUDE.md": "FooClass MUST do X."}
        report = check_spec_consistency(
            ["FooClass"], [],
            spec_loader=lambda: specs,
        )
        assert len(report.conflicts) == 1

    def test_dedup_overlapping_symbol_and_basename(self) -> None:
        """When the symbol matches a file basename, the token list
        contains the same label twice — dedup keeps one conflict."""
        specs = {"CLAUDE.md": "server.py MUST exit fast."}
        report = check_spec_consistency(
            ["server.py"], ["src/stropha/server.py"],
            spec_loader=lambda: specs,
        )
        # ``server.py`` appears via both the symbol entry and the
        # file basename — the second visit hits the seen_keys guard.
        matches = [c for c in report.conflicts if c.matched == "server.py"]
        assert len(matches) == 1

    def test_sorted_by_severity(self) -> None:
        specs = {
            "CLAUDE.md": (
                "FooClass MUST validate. FooClass should retry. "
                "Note: FooClass is legacy."
            ),
        }
        report = check_spec_consistency(
            ["FooClass"], [],
            spec_loader=lambda: specs,
        )
        severities = [c.severity for c in report.conflicts]
        # HIGH first, then MEDIUM, then LOW.
        assert severities == sorted(
            severities,
            key=lambda s: {"HIGH": 0, "MEDIUM": 1, "LOW": 2}[s],
        )

    def test_invariants_counted_even_without_conflicts(self) -> None:
        specs = {"CLAUDE.md": "Unrelated MUST happen."}
        report = check_spec_consistency(
            ["Foo"], [],
            spec_loader=lambda: specs,
        )
        assert report.invariants_found == 1
        assert report.conflicts == []

    def test_empty_input_strings_skipped(self) -> None:
        specs = {"CLAUDE.md": "Foo MUST do X."}
        report = check_spec_consistency(
            ["Foo", ""], ["", "  "],
            spec_loader=lambda: specs,
        )
        # Empty strings filtered out; only "Foo" remains so the
        # conflict still fires.
        assert any(c.matched == "Foo" for c in report.conflicts)

    def test_spec_files_scanned_listed(self) -> None:
        specs = {"a.md": "X.", "b.md": "Y."}
        report = check_spec_consistency(
            [], [],
            spec_loader=lambda: specs,
        )
        assert set(report.spec_files_scanned) == {"a.md", "b.md"}


# ---------------------------------------------------------------------------
# __all__ contract
# ---------------------------------------------------------------------------


class TestModuleSurface:
    def test_all_exports(self) -> None:
        from stropha.retrieval import spec_consistency as mod
        assert set(mod.__all__) == {
            "SpecConflict",
            "SpecConsistencyReport",
            "SpecInvariant",
            "check_spec_consistency",
            "extract_invariants",
        }


# ---------------------------------------------------------------------------
# MCP wrapper
# ---------------------------------------------------------------------------


def _make_ctx(stats_repos: list[dict] | None = None, raise_stats: bool = False) -> MagicMock:
    app = MagicMock()
    if raise_stats:
        app.storage.stats.side_effect = RuntimeError("stats boom")
    else:
        app.storage.stats.return_value = {"repos": stats_repos or []}
    ctx = MagicMock()
    ctx.request_context.lifespan_context = app
    return ctx


class TestSpecConsistencyMCP:
    def test_registered(self) -> None:
        from stropha.server import mcp

        tool_names = [t.name for t in mcp._tool_manager.list_tools()]
        assert "spec_consistency_tool" in tool_names

    def test_no_repo_root(self) -> None:
        """When stats() returns no repos, the loader returns {}."""
        from stropha.server import spec_consistency_tool

        response = spec_consistency_tool(
            ["Foo"], None, ctx=_make_ctx(stats_repos=[]),
        )
        assert response.invariants_found == 0

    def test_stats_exception_treated_as_no_repos(self) -> None:
        from stropha.server import spec_consistency_tool

        response = spec_consistency_tool(
            ["Foo"], [], ctx=_make_ctx(raise_stats=True),
        )
        assert response.invariants_found == 0

    def test_real_spec_loader(self, tmp_path: Path) -> None:
        """Plug a real repo root with a CLAUDE.md file and verify
        the default loader reads it."""
        from stropha.server import spec_consistency_tool

        (tmp_path / "CLAUDE.md").write_text(
            "FooClass MUST do X.\n",
            encoding="utf-8",
        )
        (tmp_path / "docs" / "architecture").mkdir(parents=True)
        (tmp_path / "docs" / "architecture" / "spec.md").write_text(
            "BarClass NEVER does Y.\n",
            encoding="utf-8",
        )

        response = spec_consistency_tool(
            ["FooClass", "BarClass"], [],
            ctx=_make_ctx(stats_repos=[{"local_path": str(tmp_path)}]),
        )
        # Both invariants found; both should produce conflicts.
        assert response.invariants_found >= 2
        assert len(response.conflicts) >= 2

    def test_no_spec_files_at_root(self, tmp_path: Path) -> None:
        """Repo with no CLAUDE.md and no docs/ — loader returns {}."""
        from stropha.server import spec_consistency_tool

        response = spec_consistency_tool(
            ["Foo"], [],
            ctx=_make_ctx(stats_repos=[{"local_path": str(tmp_path)}]),
        )
        assert response.invariants_found == 0
        assert response.spec_files_scanned == []

    def test_unreadable_spec_file(self, tmp_path: Path, monkeypatch) -> None:
        """A file that fails to read is silently skipped."""
        from stropha.server import spec_consistency_tool

        (tmp_path / "CLAUDE.md").write_text("MUST X.", encoding="utf-8")
        (tmp_path / "AGENTS.md").write_text("AGENTS MUST Y.", encoding="utf-8")

        original_read_text = Path.read_text

        def flaky(self, *args, **kwargs):
            if self.name == "AGENTS.md":
                raise OSError("permission denied")
            return original_read_text(self, *args, **kwargs)

        monkeypatch.setattr(Path, "read_text", flaky)

        response = spec_consistency_tool(
            [], [],
            ctx=_make_ctx(stats_repos=[{"local_path": str(tmp_path)}]),
        )
        # CLAUDE.md should still be scanned; AGENTS.md skipped silently.
        assert "CLAUDE.md" in response.spec_files_scanned
        assert "AGENTS.md" not in response.spec_files_scanned

    def test_repos_field_named_path(self, tmp_path: Path) -> None:
        """``stats()`` may return ``path`` instead of ``local_path``."""
        from stropha.server import spec_consistency_tool

        (tmp_path / "CLAUDE.md").write_text("Foo MUST do X.", encoding="utf-8")
        response = spec_consistency_tool(
            ["Foo"], [],
            ctx=_make_ctx(stats_repos=[{"path": str(tmp_path)}]),
        )
        assert response.invariants_found >= 1
