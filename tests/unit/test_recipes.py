"""Unit tests for named task recipes + MCP wrapper (WIN-240, EPIC 1.10 #31)."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from stropha.retrieval.recipes import (
    Recipe,
    RecipeInvocation,
    RecipeStep,
    _resolve_args,
    _substitute_placeholders,
    apply_recipe,
    get_recipe,
    list_recipes,
    required_placeholders,
)


# ---------------------------------------------------------------------------
# Dataclass surface
# ---------------------------------------------------------------------------


class TestDataclasses:
    def test_step_defaults(self) -> None:
        s = RecipeStep(tool="x")
        assert s.args_template == {}
        assert s.reason == ""
        assert s.budget_share == 0.0

    def test_recipe_defaults(self) -> None:
        r = Recipe(name="x", description="d")
        assert r.steps == ()
        assert r.default_budget == 4000
        assert r.placeholders == ()

    def test_invocation_defaults(self) -> None:
        inv = RecipeInvocation(tool="t")
        assert inv.args == {}
        assert inv.reason == ""
        assert inv.budget_tokens == 0

    def test_frozen(self) -> None:
        r = Recipe(name="x", description="d")
        with pytest.raises((AttributeError, Exception)):
            r.name = "y"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Registry helpers
# ---------------------------------------------------------------------------


class TestRegistry:
    def test_list_recipes_non_empty(self) -> None:
        recipes = list_recipes()
        assert len(recipes) >= 5
        names = {r.name for r in recipes}
        assert {"bug-fix", "add-feature", "review",
                "rename-symbol", "understand-file"} <= names

    def test_get_recipe_exact(self) -> None:
        r = get_recipe("bug-fix")
        assert r is not None
        assert r.name == "bug-fix"

    def test_get_recipe_case_insensitive(self) -> None:
        assert get_recipe("BUG-FIX") is not None
        assert get_recipe("Bug-Fix") is not None

    def test_get_recipe_underscore_variant(self) -> None:
        assert get_recipe("bug_fix") is not None
        assert get_recipe("rename_symbol") is not None

    def test_get_recipe_unknown_returns_none(self) -> None:
        assert get_recipe("nonexistent") is None


# ---------------------------------------------------------------------------
# Placeholder helpers
# ---------------------------------------------------------------------------


class TestSubstitution:
    def test_replaces_known_placeholders(self) -> None:
        assert _substitute_placeholders(
            "tests for {symbol}", {"symbol": "Foo"},
        ) == "tests for Foo"

    def test_leaves_unknown_placeholders_verbatim(self) -> None:
        assert _substitute_placeholders(
            "tests for {symbol}", {},
        ) == "tests for {symbol}"

    def test_stringifies_non_string_values(self) -> None:
        assert _substitute_placeholders("{n}", {"n": 42}) == "42"

    def test_no_placeholders_passes_through(self) -> None:
        assert _substitute_placeholders("plain text", {"x": 1}) == "plain text"

    def test_multiple_placeholders(self) -> None:
        out = _substitute_placeholders(
            "{a}/{b}", {"a": "x", "b": "y"},
        )
        assert out == "x/y"


class TestResolveArgs:
    def test_string_values_substituted(self) -> None:
        out = _resolve_args(
            {"query": "{symbol}"},
            {"symbol": "Foo"},
        )
        assert out == {"query": "Foo"}

    def test_non_string_values_pass_through(self) -> None:
        out = _resolve_args(
            {"top_k": 10, "use_hyde": True, "tags": ["a", "b"]},
            {"symbol": "Foo"},
        )
        assert out == {"top_k": 10, "use_hyde": True, "tags": ["a", "b"]}


class TestRequiredPlaceholders:
    def test_collects_from_all_steps(self) -> None:
        r = Recipe(
            name="r", description="",
            steps=(
                RecipeStep(tool="t1", args_template={"q": "{a}"}),
                RecipeStep(tool="t2", args_template={"p": "{b}"}),
            ),
        )
        assert required_placeholders(r) == {"a", "b"}

    def test_ignores_non_string_values(self) -> None:
        r = Recipe(
            name="r", description="",
            steps=(
                RecipeStep(tool="t", args_template={"n": 42, "q": "{a}"}),
            ),
        )
        assert required_placeholders(r) == {"a"}

    def test_empty_recipe(self) -> None:
        r = Recipe(name="r", description="")
        assert required_placeholders(r) == set()


# ---------------------------------------------------------------------------
# apply_recipe
# ---------------------------------------------------------------------------


class TestApplyRecipe:
    def test_substitutes_placeholders(self) -> None:
        r = Recipe(
            name="r", description="",
            steps=(
                RecipeStep(
                    tool="search_code",
                    args_template={"query": "{symbol}"},
                    reason="r",
                    budget_share=0.5,
                ),
            ),
            default_budget=1000,
        )
        out = apply_recipe(r, {"symbol": "Foo"})
        assert len(out) == 1
        assert out[0].tool == "search_code"
        assert out[0].args == {"query": "Foo"}
        assert out[0].reason == "r"
        assert out[0].budget_tokens == 500

    def test_budget_override(self) -> None:
        r = Recipe(
            name="r", description="",
            steps=(
                RecipeStep(tool="t", budget_share=1.0),
            ),
            default_budget=1000,
        )
        out = apply_recipe(r, {}, budget_tokens=2000)
        assert out[0].budget_tokens == 2000

    def test_budget_zero(self) -> None:
        r = Recipe(
            name="r", description="",
            steps=(RecipeStep(tool="t", budget_share=1.0),),
            default_budget=1000,
        )
        out = apply_recipe(r, {}, budget_tokens=0)
        assert out[0].budget_tokens == 0

    def test_budget_negative_clamped_to_zero(self) -> None:
        r = Recipe(
            name="r", description="",
            steps=(RecipeStep(tool="t", budget_share=1.0),),
            default_budget=1000,
        )
        out = apply_recipe(r, {}, budget_tokens=-100)
        assert out[0].budget_tokens == 0

    def test_budget_share_negative_clamped(self) -> None:
        r = Recipe(
            name="r", description="",
            steps=(RecipeStep(tool="t", budget_share=-0.5),),
            default_budget=1000,
        )
        out = apply_recipe(r, {})
        assert out[0].budget_tokens == 0

    def test_strict_missing_placeholder_raises(self) -> None:
        r = Recipe(
            name="r", description="",
            steps=(
                RecipeStep(tool="t", args_template={"q": "{symbol}"}),
            ),
        )
        with pytest.raises(KeyError, match="symbol"):
            apply_recipe(r, {}, strict=True)

    def test_strict_with_all_placeholders_passes(self) -> None:
        r = Recipe(
            name="r", description="",
            steps=(
                RecipeStep(tool="t", args_template={"q": "{symbol}"}),
            ),
        )
        out = apply_recipe(r, {"symbol": "Foo"}, strict=True)
        assert out[0].args == {"q": "Foo"}

    def test_lenient_missing_passes_through(self) -> None:
        r = Recipe(
            name="r", description="",
            steps=(
                RecipeStep(tool="t", args_template={"q": "{symbol}"}),
            ),
        )
        out = apply_recipe(r, {}, strict=False)
        assert out[0].args == {"q": "{symbol}"}

    def test_bug_fix_recipe_end_to_end(self) -> None:
        r = get_recipe("bug-fix")
        assert r is not None
        out = apply_recipe(
            r,
            {"symptom": "null pointer", "file": "src/x.py", "symbol": "doX"},
        )
        tools = [inv.tool for inv in out]
        assert "search_code" in tools
        assert "find_callers" in tools
        # Sum of budgets equals the default budget (modulo rounding).
        total = sum(inv.budget_tokens for inv in out)
        assert abs(total - r.default_budget) <= len(out)


# ---------------------------------------------------------------------------
# __all__ contract
# ---------------------------------------------------------------------------


class TestModuleSurface:
    def test_all_exports(self) -> None:
        from stropha.retrieval import recipes as mod
        assert set(mod.__all__) == {
            "Recipe",
            "RecipeInvocation",
            "RecipeStep",
            "apply_recipe",
            "get_recipe",
            "list_recipes",
            "required_placeholders",
        }


# ---------------------------------------------------------------------------
# MCP wrapper
# ---------------------------------------------------------------------------


def _make_ctx() -> MagicMock:
    ctx = MagicMock()
    ctx.request_context.lifespan_context = MagicMock()
    return ctx


class TestMCPListRecipes:
    def test_registered(self) -> None:
        from stropha.server import mcp

        tool_names = [t.name for t in mcp._tool_manager.list_tools()]
        assert "list_task_recipes" in tool_names

    def test_returns_summaries(self) -> None:
        from stropha.server import list_task_recipes

        response = list_task_recipes(ctx=_make_ctx())
        assert len(response.recipes) >= 5
        bf = next(r for r in response.recipes if r.name == "bug-fix")
        assert bf.step_count >= 3
        assert set(bf.placeholders) == {"symptom", "file", "symbol"}


class TestMCPApplyRecipe:
    def test_registered(self) -> None:
        from stropha.server import mcp

        tool_names = [t.name for t in mcp._tool_manager.list_tools()]
        assert "apply_task_recipe" in tool_names

    def test_known_recipe_returns_steps(self) -> None:
        from stropha.server import apply_task_recipe

        response = apply_task_recipe(
            "bug-fix",
            {"symptom": "panic", "file": "src/x.py", "symbol": "Foo"},
            ctx=_make_ctx(),
        )
        assert response.found is True
        assert response.name == "bug-fix"
        assert len(response.steps) >= 3
        assert response.missing_placeholders == []

    def test_unknown_recipe_found_false(self) -> None:
        from stropha.server import apply_task_recipe

        response = apply_task_recipe(
            "nonexistent", {}, ctx=_make_ctx(),
        )
        assert response.found is False
        assert response.steps == []
        assert response.budget_tokens == 0

    def test_missing_placeholders_surfaced(self) -> None:
        from stropha.server import apply_task_recipe

        response = apply_task_recipe(
            "bug-fix", {"symbol": "Foo"}, ctx=_make_ctx(),
        )
        assert response.found is True
        assert set(response.missing_placeholders) == {"symptom", "file"}

    def test_budget_override_honoured(self) -> None:
        from stropha.server import apply_task_recipe

        response = apply_task_recipe(
            "review",
            {"changed_file": "x.py", "changed_symbol": "Foo"},
            budget_tokens=10000,
            ctx=_make_ctx(),
        )
        assert response.budget_tokens == 10000

    def test_negative_budget_override_clamped(self) -> None:
        from stropha.server import apply_task_recipe

        response = apply_task_recipe(
            "bug-fix",
            {"symptom": "x", "file": "y", "symbol": "z"},
            budget_tokens=-1,
            ctx=_make_ctx(),
        )
        assert response.budget_tokens == 0

    def test_context_none_treated_as_empty(self) -> None:
        from stropha.server import apply_task_recipe

        response = apply_task_recipe(
            "bug-fix", None, ctx=_make_ctx(),
        )
        assert response.found is True
        # All placeholders missing.
        assert set(response.missing_placeholders) == {"symptom", "file", "symbol"}
