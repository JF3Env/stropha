"""Unit tests for MCP recipe prompts (WIN-254, EPIC 1.12 #91)."""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Prompt registration
# ---------------------------------------------------------------------------


class TestPromptRegistration:
    def test_all_recipe_prompts_registered(self) -> None:
        from stropha.server import mcp

        prompt_names = [p.name for p in mcp._prompt_manager.list_prompts()]
        for name in (
            "bug-fix",
            "add-feature",
            "review",
            "rename-symbol",
            "understand-file",
        ):
            assert name in prompt_names


# ---------------------------------------------------------------------------
# Render helper
# ---------------------------------------------------------------------------


class TestRenderRecipePrompt:
    def test_known_recipe(self) -> None:
        from stropha.server import _render_recipe_prompt

        out = _render_recipe_prompt(
            "bug-fix",
            {"symptom": "panic", "file": "src/x.py", "symbol": "Foo"},
        )
        assert "bug-fix" in out
        # Steps include the four key tools.
        for tool in ("search_code", "find_callers", "find_tests_for",
                     "find_rationale"):
            assert tool in out
        # Args substituted.
        assert "panic" in out
        assert "src/x.py" in out
        assert "Foo" in out

    def test_unknown_recipe(self) -> None:
        from stropha.server import _render_recipe_prompt

        out = _render_recipe_prompt("nonexistent", {})
        assert "Unknown recipe" in out

    def test_missing_placeholder_passes_through(self) -> None:
        from stropha.server import _render_recipe_prompt

        out = _render_recipe_prompt("bug-fix", {})  # no args
        # Placeholders left as literal {name} tokens (strict=False).
        assert "{symptom}" in out
        assert "{file}" in out


# ---------------------------------------------------------------------------
# Per-prompt invocation
# ---------------------------------------------------------------------------


class TestBugFixPrompt:
    def test_invocation(self) -> None:
        from stropha.server import bug_fix_prompt

        out = bug_fix_prompt(
            symptom="null pointer", file="src/x.py", symbol="Foo",
        )
        assert "bug-fix" in out
        assert "null pointer" in out


class TestAddFeaturePrompt:
    def test_invocation(self) -> None:
        from stropha.server import add_feature_prompt

        out = add_feature_prompt(
            feature_query="add login", anchor_symbol="AuthService",
        )
        assert "add-feature" in out
        assert "AuthService" in out


class TestReviewPrompt:
    def test_invocation(self) -> None:
        from stropha.server import review_prompt

        out = review_prompt(changed_file="x.py", changed_symbol="Foo")
        assert "review" in out
        assert "x.py" in out


class TestRenameSymbolPrompt:
    def test_invocation(self) -> None:
        from stropha.server import rename_symbol_prompt

        out = rename_symbol_prompt(symbol="OldName")
        assert "rename-symbol" in out
        assert "OldName" in out


class TestUnderstandFilePrompt:
    def test_invocation(self) -> None:
        from stropha.server import understand_file_prompt

        out = understand_file_prompt(file="x.py", primary_symbol="Foo")
        assert "understand-file" in out
        assert "x.py" in out
