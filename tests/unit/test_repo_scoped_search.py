"""Repository-scope behavior at public semantic search boundaries."""

from __future__ import annotations

from collections import deque
from inspect import signature
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from typer.testing import CliRunner

from stropha.cli import app as cli_app
from stropha.ingest.git_meta import RepoIdentity
from stropha.models import Chunk, RepoRef, SearchHit
from stropha.retrieval.query_router import QueryIntent, RoutedQuery
from stropha.retrieval.smart_search import SmartSearchResult
from stropha.storage import Storage
from stropha.tools.git_url import GitUrlSearchResult


def _hit(chunk_id: str, repo: str) -> SearchHit:
    return SearchHit(
        chunk_id=chunk_id,
        rel_path="src/shared.py",
        start_line=1,
        end_line=5,
        language="python",
        kind="function",
        symbol="shared",
        snippet=f"def {chunk_id}(): pass",
        score=1.0,
        rank=1,
        repo=RepoRef(normalized_key=repo),
    )


def _mcp_app(hits: list[SearchHit], repo_ids: dict[str, int]) -> MagicMock:
    app = MagicMock()
    app.session_id = "scope-test"
    app.client_id = None
    app.seen_chunk_ids = set()
    app.pinned_chunk_ids = set()
    app.recent_queries = deque(maxlen=5)
    app.built = SimpleNamespace(router=None)
    app.search_engine.search.return_value = hits
    ids_by_key = {
        hit.repo.normalized_key: repo_ids[hit.chunk_id]
        for hit in hits
        if hit.repo is not None and hit.chunk_id in repo_ids
    }
    app.storage.lookup_repo_id.side_effect = ids_by_key.get
    app.storage.get_chunk_by_id.side_effect = lambda chunk_id: (
        {"chunk_id": chunk_id, "repo_id": repo_ids[chunk_id]}
        if chunk_id in repo_ids
        else None
    )
    app.storage.lookup_by_chunk_id.side_effect = lambda chunk_id: next(
        (hit for hit in hits if hit.chunk_id == chunk_id), None
    )
    app.storage.compute_facets.return_value = {"language": {}, "kind": {}, "repo": {}}
    return app


def _ctx(app: MagicMock) -> MagicMock:
    ctx = MagicMock()
    ctx.request_context.lifespan_context = app
    return ctx


def _seed_direct_route_storage(tmp_path: Path) -> tuple[Storage, int, int]:
    storage = Storage(tmp_path / "scope.db", embedding_dim=4)
    repo_ids: list[int] = []
    for name in ("foreign", "local"):
        root = tmp_path / name
        root.mkdir()
        repo_id = storage.register_repo(
            RepoIdentity(
                normalized_key=f"local:{name}",
                remote_url=None,
                root_path=root,
                default_branch="main",
                head_commit="abc",
            )
        )
        repo_ids.append(repo_id)
        storage.upsert_chunk(
            Chunk(
                chunk_id=f"{name}-chunk",
                rel_path="src/shared.py",
                language="python",
                kind="function",
                symbol="Shared",
                parent_chunk_id=None,
                start_line=1,
                end_line=2,
                content=f"def {name}_shared(): pass",
                content_hash=f"hash-{name}",
            ),
            [0.0] * 4,
            "stub",
            4,
            repo_id=repo_id,
        )
        target_id = f"{repo_id}:target"
        caller_id = f"{repo_id}:caller"
        storage._conn.executemany(
            """INSERT INTO graph_nodes(
                   node_id, label, file_type, source_file, source_location,
                   community_id, community_label, repo_id
               ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            [
                (
                    target_id,
                    "Shared",
                    "function",
                    "src/shared.py",
                    "L1",
                    7,
                    f"{name} community",
                    repo_id,
                ),
                (
                    caller_id,
                    f"{name}_caller",
                    "function",
                    "src/shared.py",
                    "L1",
                    7,
                    f"{name} community",
                    repo_id,
                ),
            ],
        )
        storage._conn.execute(
            """INSERT INTO graph_edges(
                   source, target, relation, confidence, confidence_score,
                   context, source_file, source_location, weight, repo_id
               ) VALUES (?, ?, 'calls', 'EXTRACTED', 1.0, '', ?, 'L1', 1.0, ?)""",
            (caller_id, target_id, "src/shared.py", repo_id),
        )
        storage._conn.execute(
            "INSERT INTO graph_meta(key, value) VALUES (?, 'now')",
            (f"last_loaded_at:{repo_id}",),
        )
    storage.commit()
    foreign_repo, local_repo = repo_ids
    return storage, local_repo, foreign_repo


def test_mcp_search_defaults_to_current_repo_and_opt_in_broadens() -> None:
    from stropha.server import search_code

    local = _hit("local", "github.com/example/local")
    foreign = _hit("foreign", "github.com/example/foreign")
    app = _mcp_app([local, foreign], {"local": 7, "foreign": 8})

    with patch("stropha.server._resolve_target_repo_id", return_value=7):
        scoped = search_code("shared", ctx=_ctx(app))
        broad = search_code("shared", include_other_repos=True, ctx=_ctx(app))

    assert [result.chunk_id for result in scoped.results] == ["local"]
    assert {result.chunk_id for result in broad.results} == {"local", "foreign"}
    assert app.search_engine.search.call_args_list[0].kwargs["repo_id"] == 7
    assert "repo_id" not in app.search_engine.search.call_args_list[1].kwargs


def test_mcp_search_accepts_explicit_indexed_repository_selector() -> None:
    from stropha.server import search_code

    local = _hit("local", "github.com/example/local")
    app = _mcp_app([local], {"local": 7})
    with patch("stropha.server._resolve_target_repo_id", return_value=7) as resolve:
        response = search_code(
            "shared",
            repo_url="github.com/example/local",
            ctx=_ctx(app),
        )

    assert [result.chunk_id for result in response.results] == ["local"]
    resolve.assert_called_once_with(app, repo="github.com/example/local")


def test_mcp_search_rejects_relative_repository_selector() -> None:
    from stropha.server import search_code

    local = _hit("local", "github.com/example/local")
    app = _mcp_app([local], {"local": 7})
    with patch("stropha.server._resolve_target_repo_id", return_value=None) as resolve:
        response = search_code("shared", repo_url=".", ctx=_ctx(app))

    assert response.results == []
    resolve.assert_called_once_with(app, repo=".")


def test_direct_mcp_routes_scope_duplicate_symbols_and_community_ids(
    tmp_path: Path,
) -> None:
    from stropha.server import find_callers, get_community, get_file_outline, get_symbol

    storage, local_repo, foreign_repo = _seed_direct_route_storage(tmp_path)
    app = _mcp_app([], {})
    app.storage = storage
    try:
        with patch("stropha.server._resolve_target_repo_id", return_value=local_repo):
            scoped_symbol = get_symbol("Shared", limit=1, ctx=_ctx(app))
            broad_symbols = get_symbol(
                "Shared", limit=10, include_other_repos=True, ctx=_ctx(app)
            )
            scoped_outline = get_file_outline("src/shared.py", ctx=_ctx(app))
            broad_outline = get_file_outline(
                "src/shared.py", include_other_repos=True, ctx=_ctx(app)
            )
            scoped_callers = find_callers("Shared", ctx=_ctx(app))
            scoped_community = get_community("7", ctx=_ctx(app))
            broad_community = get_community(
                "7", include_other_repos=True, ctx=_ctx(app)
            )
    finally:
        storage.close()

    assert [hit.chunk_id for hit in scoped_symbol] == ["local-chunk"]
    assert {hit.chunk_id for hit in broad_symbols} == {
        "local-chunk",
        "foreign-chunk",
    }
    assert [entry.chunk_id for entry in scoped_outline] == ["local-chunk"]
    assert {entry.chunk_id for entry in broad_outline} == {
        "local-chunk",
        "foreign-chunk",
    }
    assert [node["label"] for node in scoped_callers["callers"]] == [
        "local_caller"
    ]
    assert {member["repo_id"] for member in scoped_community["members"]} == {
        local_repo
    }
    assert {member["repo_id"] for member in broad_community["members"]} == {
        local_repo,
        foreign_repo,
    }


def test_every_direct_symbol_and_graph_mcp_route_exposes_scope_opt_in() -> None:
    from stropha import server

    routes = (
        server.get_symbol,
        server.get_file_outline,
        server.expand_chunk,
        server.expand_chunks,
        server.pin_chunks,
        server.list_pins,
        server.audit_citations_tool,
        server.suggest_next_tool_mcp,
        server.assert_snippet_freshness_tool,
        server.working_set_tool,
        server.find_callers,
        server.find_related,
        server.get_community,
        server.find_rationale,
        server.find_tests_for,
        server.trace_feature,
        server.verify_claim_tool,
        server.check_symbols_tool,
        server.blast_radius_tool,
        server.test_impact_tool,
        server.contract_tests_tool,
    )

    for route in routes:
        parameter = signature(route).parameters["include_other_repos"]
        assert parameter.annotation in (bool, "bool")
        assert parameter.default is False


def test_direct_graph_routes_fail_closed_when_current_repo_is_unresolved() -> None:
    from stropha import server

    app = _mcp_app([], {})
    calls = (
        lambda: server.find_callers("Shared", ctx=_ctx(app)),
        lambda: server.find_related("Shared", ctx=_ctx(app)),
        lambda: server.get_community("7", ctx=_ctx(app)),
        lambda: server.find_rationale("Shared", ctx=_ctx(app)),
        lambda: server.find_tests_for("Shared", ctx=_ctx(app)),
        lambda: server.trace_feature("shared flow", ctx=_ctx(app)),
    )

    with patch("stropha.server._resolve_target_repo_id", return_value=None):
        results = [call() for call in calls]
        symbol = server.get_symbol("Shared", ctx=_ctx(app))
        blast = server.blast_radius_tool(["Shared"], ctx=_ctx(app))
        impact = server.test_impact_tool(["Shared"], ctx=_ctx(app))
        contracts = server.contract_tests_tool("Shared", ctx=_ctx(app))

    assert symbol == []
    assert all(result["graph_loaded"] is False for result in results)
    assert all(result["message"] for result in results)
    assert blast.graph_loaded is False
    assert impact.graph_loaded is False
    assert contracts.graph_loaded is False
    app.storage.lookup_by_symbol.assert_not_called()


def test_mcp_search_fails_closed_when_current_repo_cannot_be_resolved() -> None:
    from stropha.server import search_code

    app = _mcp_app([_hit("foreign", "github.com/example/foreign")], {"foreign": 8})

    with patch("stropha.server._resolve_target_repo_id", return_value=None):
        result = search_code("shared", ctx=_ctx(app))

    assert result.results == []
    app.search_engine.search.assert_not_called()


def test_mcp_search_does_not_reinsert_foreign_pins() -> None:
    from stropha.server import search_code

    local = _hit("local-pin", "github.com/example/local")
    foreign = _hit("foreign-pin", "github.com/example/foreign")
    app = _mcp_app([local, foreign], {"local-pin": 7, "foreign-pin": 8})
    app.search_engine.search.return_value = []
    app.pinned_chunk_ids = {"local-pin", "foreign-pin"}

    with patch("stropha.server._resolve_target_repo_id", return_value=7):
        result = search_code("shared", ctx=_ctx(app))

    assert [item.chunk_id for item in result.results] == ["local-pin"]


def test_mcp_remote_search_uses_remote_identity_for_zero_result_pins() -> None:
    from stropha.server import search_code

    remote = _hit("remote-pin", "github.com/example/remote")
    foreign = _hit("foreign-pin", "github.com/example/foreign")
    app = _mcp_app([remote, foreign], {"remote-pin": 23, "foreign-pin": 8})
    app.pinned_chunk_ids = {"remote-pin", "foreign-pin"}

    with patch(
        "stropha.tools.git_url.search_git_url_scoped",
        return_value=GitUrlSearchResult(hits=[], repo_id=23),
    ):
        result = search_code(
            "missing", repo_url="https://github.com/example/remote", ctx=_ctx(app)
        )

    assert [item.chunk_id for item in result.results] == ["remote-pin"]


def test_mcp_rejects_invalid_nonempty_repo_url() -> None:
    from stropha.server import search_code

    app = _mcp_app([_hit("local", "github.com/example/local")], {"local": 7})
    result = search_code("shared", repo_url="not-a-git-url", ctx=_ctx(app))

    assert result.results == []
    app.search_engine.search.assert_not_called()


def test_search_code_include_other_repos_is_last_optional_parameter() -> None:
    from stropha.server import search_code

    names = list(signature(search_code).parameters)
    assert names[-2:] == ["include_other_repos", "ctx"]


def _routed_search_result(query: str) -> SmartSearchResult:
    return SmartSearchResult(
        hits=[],
        routed=RoutedQuery(
            intent=QueryIntent.SEARCH_CODE,
            symbol=None,
            original_query=query,
            confidence=1.0,
            reason="test",
        ),
        tool_used="search_code",
    )


def test_smart_search_defaults_scoped_and_explicitly_broadens() -> None:
    from stropha.server import smart_search_tool

    app = _mcp_app([], {})
    with (
        patch("stropha.server._resolve_target_repo_id", return_value=7),
        patch(
            "stropha.server.smart_search",
            side_effect=lambda query, *args, **kwargs: _routed_search_result(query),
        ) as search,
        patch("stropha.server.graph_loaded", return_value=False),
    ):
        smart_search_tool("shared", ctx=_ctx(app))
        smart_search_tool("shared", include_other_repos=True, ctx=_ctx(app))

    assert search.call_args_list[0].kwargs["repo_id"] == 7
    assert search.call_args_list[0].kwargs["search_engine"] is app.search_engine
    assert search.call_args_list[1].kwargs["repo_id"] is None


def test_composed_tools_default_scoped_and_explicitly_broaden() -> None:
    from stropha.server import assemble_context, execute_investigation, plan_investigation

    app = _mcp_app([], {})
    with (
        patch("stropha.server._resolve_target_repo_id", return_value=7),
        patch("stropha.retrieval.graph.graph_loaded", return_value=False),
    ):
        plan_investigation("task", ctx=_ctx(app))
        plan_investigation("task", include_other_repos=True, ctx=_ctx(app))
        assemble_context("task", ctx=_ctx(app))
        assemble_context("task", include_other_repos=True, ctx=_ctx(app))
        execute_investigation("task", ctx=_ctx(app))
        execute_investigation("task", include_other_repos=True, ctx=_ctx(app))

    repo_ids = [call.kwargs.get("repo_id") for call in app.search_engine.search.call_args_list]
    assert repo_ids == [7, None, 7, None, 7, 7, None, None]
    assert all(
        "repo_id" not in call.kwargs
        for call in app.search_engine.search.call_args_list
        if call.kwargs.get("repo_id") is None
    )


def test_scoped_assembly_defensively_drops_foreign_primary_hits() -> None:
    from stropha.server import assemble_context

    local = _hit("local", "github.com/example/local")
    foreign = _hit("foreign", "github.com/example/foreign")
    app = _mcp_app([local, foreign], {"local": 7, "foreign": 8})

    with patch(
        "stropha.server_memory_tools._resolve_repo",
        return_value=(7, Path("/tmp/local")),
    ) as resolve:
        response = assemble_context(
            "shared",
            repo="github.com/example/local",
            ctx=_ctx(app),
        )

    primary = next(section for section in response.sections if section.key == "primary")
    assert [hit.chunk_id for hit in primary.hits] == ["local"]
    assert resolve.call_args.args == (app, "github.com/example/local")


def test_explicit_repo_rejects_cross_repository_broadening() -> None:
    from stropha.server import assemble_context, search_code

    app = _mcp_app([_hit("foreign", "github.com/example/foreign")], {"foreign": 8})

    searched = search_code(
        "shared",
        repo_url="github.com/example/local",
        include_other_repos=True,
        ctx=_ctx(app),
    )
    assembled = assemble_context(
        "shared",
        repo="github.com/example/local",
        include_other_repos=True,
        ctx=_ctx(app),
    )

    assert searched.results == []
    assert assembled.sections == []
    app.search_engine.search.assert_not_called()


def test_assembly_rejects_relative_repository_selector() -> None:
    from stropha.server import assemble_context

    app = _mcp_app([_hit("local", "github.com/example/local")], {"local": 7})

    assembled = assemble_context("shared", repo=".", ctx=_ctx(app))

    assert assembled.sections == []
    app.search_engine.search.assert_not_called()


def test_explicit_repo_drives_diff_scope_repository() -> None:
    from stropha.server import assemble_context

    app = _mcp_app([], {})
    selected = Path("/tmp/selected-repo")
    assembled = SimpleNamespace(
        task="task",
        bundle="",
        sections=[],
        total_tokens=0,
        budget_tokens=4000,
        spilled_count=0,
        truncated_oversize=False,
    )
    with (
        patch("stropha.server_memory_tools._resolve_repo", return_value=(7, selected)),
        patch("stropha.retrieval.assemble.assemble_context", return_value=assembled) as run,
    ):
        assemble_context(
            "task",
            repo="github.com/example/local",
            diff_scope=True,
            ctx=_ctx(app),
        )

    assert run.call_args.kwargs["diff_repo"] == selected


def test_broad_investigation_plan_preserves_opt_in_on_every_step() -> None:
    from stropha.server import plan_investigation

    app = _mcp_app([_hit("local", "github.com/example/local")], {"local": 7})
    with patch("stropha.retrieval.graph.graph_loaded", return_value=True):
        plan = plan_investigation(
            "task", include_other_repos=True, ctx=_ctx(app)
        )

    assert {step.tool for step in plan.steps} == {
        "search_code",
        "find_callers",
        "find_tests_for",
        "find_rationale",
    }
    assert all(step.args["include_other_repos"] is True for step in plan.steps)


def test_all_composed_tools_fail_closed_when_repo_is_unresolved() -> None:
    from stropha.server import (
        assemble_context,
        execute_investigation,
        plan_investigation,
        smart_search_tool,
    )

    app = _mcp_app([_hit("foreign", "github.com/example/foreign")], {"foreign": 8})
    with patch("stropha.server._resolve_target_repo_id", return_value=None):
        smart = smart_search_tool("task", ctx=_ctx(app))
        plan = plan_investigation("task", ctx=_ctx(app))
        assembled = assemble_context("task", ctx=_ctx(app))
        executed = execute_investigation("task", ctx=_ctx(app))

    assert smart.results == []
    assert plan.steps == []
    assert assembled.sections == []
    assert executed.sections == []
    app.search_engine.search.assert_not_called()


def test_cli_search_defaults_to_current_repo_and_has_explicit_opt_in() -> None:
    storage = MagicMock()
    storage.lookup_repo_id.return_value = 7
    retrieval = MagicMock()
    retrieval.search.return_value = []
    built = SimpleNamespace(storage=storage, retrieval=retrieval)
    identity = SimpleNamespace(normalized_key="github.com/example/local")

    with (
        patch("stropha.cli.load_pipeline_config", return_value=MagicMock()),
        patch("stropha.cli.build_stages", return_value=built),
        patch(
            "stropha.cli._load_config",
            return_value=SimpleNamespace(target_repo=Path.cwd(), log_level="INFO"),
        ),
        patch("stropha.ingest.git_meta.detect", return_value=identity),
    ):
        scoped = CliRunner().invoke(cli_app, ["search", "shared"])
        broad = CliRunner().invoke(
            cli_app, ["search", "shared", "--include-other-repos"]
        )

    assert scoped.exit_code == 0
    assert broad.exit_code == 0
    assert retrieval.search.call_args_list[0].kwargs["repo_id"] == 7
    assert "repo_id" not in retrieval.search.call_args_list[1].kwargs


def test_cli_recursive_search_uses_pipeline_override() -> None:
    storage = MagicMock()
    storage.lookup_repo_id.return_value = 7
    retrieval = MagicMock()
    retrieval.search.return_value = []
    built = SimpleNamespace(storage=storage, retrieval=retrieval)
    identity = SimpleNamespace(normalized_key="github.com/example/local")

    with (
        patch("stropha.cli.load_pipeline_config", return_value={}) as load_config,
        patch("stropha.cli.build_stages", return_value=built),
        patch(
            "stropha.cli._load_config",
            return_value=SimpleNamespace(target_repo=Path.cwd(), log_level="INFO"),
        ),
        patch("stropha.ingest.git_meta.detect", return_value=identity),
    ):
        result = CliRunner().invoke(cli_app, ["search", "shared", "--recursive"])

    assert result.exit_code == 0
    load_config.assert_called_once_with(
        overrides={"retrieval": {"config": {"recursive_enabled": True}}}
    )


def test_cli_rejects_invalid_nonempty_repo_url() -> None:
    with patch("stropha.cli.build_stages") as build:
        result = CliRunner().invoke(cli_app, ["search", "shared", "not-a-git-url"])

    assert result.exit_code == 1
    assert "Invalid git repository URL" in result.output
    build.assert_not_called()
