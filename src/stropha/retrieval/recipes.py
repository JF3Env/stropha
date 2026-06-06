"""Named task recipes (WIN-240, EPIC 1.10 #31).

Most agent tasks fall into a small number of patterns: bug-fix,
add-feature, code-review, rename-symbol, understand-file. Each
pattern has a predictable sequence of stropha tool invocations the
agent benefits from running — but today every agent has to
reinvent the workflow on every task.

WIN-240 ships a recipe registry that captures these patterns as
declarative templates. A recipe is a name + description + ordered
``RecipeStep`` list + suggested token budget. Each step names a
stropha tool, an argument template (with ``{placeholder}`` slots),
and a human-readable reason. ``apply_recipe(name, context)`` substitutes
the placeholders from the caller's context dict and returns a list of
concrete ``RecipeInvocation`` records the agent can execute one by
one (or hand to ``execute_plan`` for server-side DAG execution).

The module is a standalone primitive matching the WIN-230 / WIN-236
/ WIN-237 / WIN-238 / WIN-239 pattern — the MCP wrapper is the
thin adapter in ``server.py``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

# Placeholder pattern: matches ``{name}`` but NOT ``{{name}}`` (escape
# for literal braces). Reuses Python's str.format conventions.
_PLACEHOLDER_RE = re.compile(r"\{([a-zA-Z_][a-zA-Z0-9_]*)\}")


@dataclass(frozen=True)
class RecipeStep:
    """One step of a named recipe.

    ``args_template`` is a dict whose string values may contain
    ``{placeholder}`` slots. Non-string values pass through unchanged
    (e.g., booleans, integers, lists of constants).
    """

    tool: str
    args_template: dict[str, Any] = field(default_factory=dict)
    reason: str = ""
    # Soft budget allocation. ``budget_share`` in ``[0, 1]`` says
    # what fraction of the recipe's ``default_budget`` this step should
    # consume. Sums across a recipe don't have to equal 1 — the
    # caller decides whether to interpret them as weights or hard caps.
    budget_share: float = 0.0


@dataclass(frozen=True)
class Recipe:
    """A named workflow template.

    The recipe is metadata only — it doesn't execute anything. Use
    :func:`apply_recipe` to materialise the steps with concrete args,
    then hand them to the orchestrator of your choice
    (``execute_plan`` is the obvious candidate).
    """

    name: str
    description: str
    steps: tuple[RecipeStep, ...] = field(default_factory=tuple)
    default_budget: int = 4000
    placeholders: tuple[str, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class RecipeInvocation:
    """A recipe step with placeholders resolved.

    Mirrors the shape ``execute_plan`` consumes: ``tool`` + ``args``
    are ready to forward to the named MCP tool.
    """

    tool: str
    args: dict[str, Any] = field(default_factory=dict)
    reason: str = ""
    budget_tokens: int = 0


# ---------------------------------------------------------------------------
# Built-in recipes
# ---------------------------------------------------------------------------


_BUG_FIX = Recipe(
    name="bug-fix",
    description=(
        "Locate the failure site, surface tests + callers, then "
        "fetch rationale so the fix doesn't violate an undocumented "
        "invariant."
    ),
    steps=(
        RecipeStep(
            tool="search_code",
            args_template={"query": "{symptom}"},
            reason="Find the code that matches the failure symptom.",
            budget_share=0.30,
        ),
        RecipeStep(
            tool="get_file_outline",
            args_template={"path": "{file}"},
            reason="Map the file containing the suspect symbol.",
            budget_share=0.10,
        ),
        RecipeStep(
            tool="find_tests_for",
            args_template={"symbol": "{symbol}"},
            reason="Find the tests exercising the symbol — start the regression net.",
            budget_share=0.20,
        ),
        RecipeStep(
            tool="find_callers",
            args_template={"symbol": "{symbol}"},
            reason="Find every call site so the fix doesn't break a caller.",
            budget_share=0.20,
        ),
        RecipeStep(
            tool="find_rationale",
            args_template={"symbol": "{symbol}"},
            reason="Pull design notes — explains constraints the fix must respect.",
            budget_share=0.20,
        ),
    ),
    default_budget=4000,
    placeholders=("symptom", "file", "symbol"),
)


_ADD_FEATURE = Recipe(
    name="add-feature",
    description=(
        "Map the surface, find the integration point, check sibling "
        "patterns, and verify which tests cover the area before "
        "writing new code."
    ),
    steps=(
        RecipeStep(
            tool="bootstrap_environment",
            args_template={},
            reason="Confirm the stack + entry points before adding new surface.",
            budget_share=0.15,
        ),
        RecipeStep(
            tool="search_code",
            args_template={"query": "{feature_query}"},
            reason="Find the area the new feature plugs into.",
            budget_share=0.25,
        ),
        RecipeStep(
            tool="find_related",
            args_template={"symbol": "{anchor_symbol}"},
            reason="Surface sibling patterns the new feature should mimic.",
            budget_share=0.20,
        ),
        RecipeStep(
            tool="find_tests_for",
            args_template={"symbol": "{anchor_symbol}"},
            reason="Find the test suite the new feature must extend.",
            budget_share=0.20,
        ),
        RecipeStep(
            tool="find_rationale",
            args_template={"symbol": "{anchor_symbol}"},
            reason="Check ADRs / design notes — avoid violating an invariant.",
            budget_share=0.20,
        ),
    ),
    default_budget=4500,
    placeholders=("feature_query", "anchor_symbol"),
)


_CODE_REVIEW = Recipe(
    name="review",
    description=(
        "Read the changed file's outline, gather callers + tests, "
        "and check rationale for every changed symbol — the minimum "
        "context for a competent review."
    ),
    steps=(
        RecipeStep(
            tool="get_file_outline",
            args_template={"path": "{changed_file}"},
            reason="Outline the file being reviewed.",
            budget_share=0.15,
        ),
        RecipeStep(
            tool="find_callers",
            args_template={"symbol": "{changed_symbol}"},
            reason="See which call sites a change to this symbol affects.",
            budget_share=0.30,
        ),
        RecipeStep(
            tool="find_tests_for",
            args_template={"symbol": "{changed_symbol}"},
            reason="Confirm the test coverage of the changed symbol.",
            budget_share=0.30,
        ),
        RecipeStep(
            tool="find_rationale",
            args_template={"symbol": "{changed_symbol}"},
            reason="Surface design notes — flag departures from established intent.",
            budget_share=0.25,
        ),
    ),
    default_budget=3500,
    placeholders=("changed_file", "changed_symbol"),
)


_RENAME_SYMBOL = Recipe(
    name="rename-symbol",
    description=(
        "Audit every reference and test before renaming so no call "
        "site is missed."
    ),
    steps=(
        RecipeStep(
            tool="get_symbol",
            args_template={"symbol": "{symbol}"},
            reason="Locate the definition of the symbol about to be renamed.",
            budget_share=0.20,
        ),
        RecipeStep(
            tool="find_callers",
            args_template={"symbol": "{symbol}"},
            reason="Enumerate every caller — these will need the new name.",
            budget_share=0.40,
        ),
        RecipeStep(
            tool="find_tests_for",
            args_template={"symbol": "{symbol}"},
            reason="Find tests that reference the symbol by name.",
            budget_share=0.40,
        ),
    ),
    default_budget=3000,
    placeholders=("symbol",),
)


_UNDERSTAND_FILE = Recipe(
    name="understand-file",
    description=(
        "Outline the file, find its callers across the codebase, and "
        "fetch related rationale — the cheap orientation for a "
        "previously unseen file."
    ),
    steps=(
        RecipeStep(
            tool="get_file_outline",
            args_template={"path": "{file}"},
            reason="Map the symbolic structure of the file.",
            budget_share=0.25,
        ),
        RecipeStep(
            tool="search_code",
            args_template={"query": "{file}"},
            reason="See how this file is referenced elsewhere.",
            budget_share=0.25,
        ),
        RecipeStep(
            tool="find_related",
            args_template={"symbol": "{primary_symbol}"},
            reason="Surface sibling symbols in the same area.",
            budget_share=0.25,
        ),
        RecipeStep(
            tool="find_rationale",
            args_template={"symbol": "{primary_symbol}"},
            reason="Pull design notes that frame the file's purpose.",
            budget_share=0.25,
        ),
    ),
    default_budget=3500,
    placeholders=("file", "primary_symbol"),
)


_BUILTIN_REGISTRY: dict[str, Recipe] = {
    r.name: r
    for r in (_BUG_FIX, _ADD_FEATURE, _CODE_REVIEW, _RENAME_SYMBOL,
              _UNDERSTAND_FILE)
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def list_recipes() -> list[Recipe]:
    """Return all built-in recipes in registration order."""
    return list(_BUILTIN_REGISTRY.values())


def get_recipe(name: str) -> Recipe | None:
    """Return the recipe named ``name`` or ``None`` when unknown.

    Lookup is case-insensitive and tolerates hyphen / underscore
    variation (``bug_fix`` == ``bug-fix``).
    """
    normalized = name.strip().lower().replace("_", "-")
    return _BUILTIN_REGISTRY.get(normalized)


def required_placeholders(recipe: Recipe) -> set[str]:
    """Return the union of placeholders any step of ``recipe`` uses.

    Distinct from ``recipe.placeholders`` which is the declared
    expectation; this walks the actual templates so a typo in the
    template surfaces immediately.
    """
    seen: set[str] = set()
    for step in recipe.steps:
        for value in step.args_template.values():
            if isinstance(value, str):
                seen.update(_PLACEHOLDER_RE.findall(value))
    return seen


def apply_recipe(
    recipe: Recipe,
    context: dict[str, Any],
    *,
    budget_tokens: int | None = None,
    strict: bool = False,
) -> list[RecipeInvocation]:
    """Materialise ``recipe`` against the caller's ``context``.

    Args:
        recipe: the template to apply.
        context: dict whose keys map to placeholders. Missing keys
            either trigger ``KeyError`` (when ``strict=True``) or
            leave the placeholder as the literal ``{name}`` token
            (when ``strict=False``, default).
        budget_tokens: total budget to distribute across the steps
            via each step's ``budget_share``. ``None`` falls back to
            ``recipe.default_budget``. ``0`` distributes zeros.
        strict: when True, every placeholder referenced by the
            template MUST appear in ``context`` — otherwise
            ``KeyError`` is raised. When False, missing placeholders
            pass through verbatim so the caller can post-process.

    Returns:
        A list of :class:`RecipeInvocation` ready to forward to the
        appropriate MCP tool.
    """
    if strict:
        missing = required_placeholders(recipe) - set(context.keys())
        if missing:
            raise KeyError(
                f"Recipe {recipe.name!r} requires placeholders "
                f"{sorted(missing)} not present in context."
            )

    budget = recipe.default_budget if budget_tokens is None else budget_tokens
    budget = max(0, int(budget))

    out: list[RecipeInvocation] = []
    for step in recipe.steps:
        resolved_args = _resolve_args(step.args_template, context)
        share = max(0.0, float(step.budget_share))
        step_budget = round(share * budget)
        out.append(RecipeInvocation(
            tool=step.tool,
            args=resolved_args,
            reason=step.reason,
            budget_tokens=step_budget,
        ))
    return out


def _resolve_args(
    template: dict[str, Any], context: dict[str, Any],
) -> dict[str, Any]:
    """Substitute ``{placeholder}`` slots in string values of
    ``template`` with matching keys from ``context``."""
    resolved: dict[str, Any] = {}
    for key, value in template.items():
        if isinstance(value, str):
            resolved[key] = _substitute_placeholders(value, context)
        else:
            resolved[key] = value
    return resolved


def _substitute_placeholders(text: str, context: dict[str, Any]) -> str:
    """Replace ``{key}`` with ``context[key]`` (stringified). Missing
    keys are left verbatim so the caller can spot them downstream."""
    def _repl(match: re.Match[str]) -> str:
        key = match.group(1)
        if key in context:
            return str(context[key])
        return match.group(0)
    return _PLACEHOLDER_RE.sub(_repl, text)


__all__ = [
    "Recipe",
    "RecipeInvocation",
    "RecipeStep",
    "apply_recipe",
    "get_recipe",
    "list_recipes",
    "required_placeholders",
]
