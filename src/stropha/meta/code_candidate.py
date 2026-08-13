"""Validation primitives for constrained v1 code candidates."""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..errors import StrophaError

MAX_CODE_ARTIFACTS = 1
MAX_CODE_ARTIFACT_BYTES = 64 * 1024

_ARTIFACT_PATH_RE = re.compile(
    r"^adapters/reranker/(?P<name>[A-Za-z0-9][A-Za-z0-9_-]*)\.py$"
)
_FORBIDDEN_IMPORT_ROOTS = frozenset(
    {
        "asyncio",
        "builtins",
        "ctypes",
        "http",
        "importlib",
        "marshal",
        "multiprocessing",
        "os",
        "pathlib",
        "pickle",
        "requests",
        "shutil",
        "signal",
        "socket",
        "subprocess",
        "sys",
        "tempfile",
        "threading",
        "urllib",
    }
)
_FORBIDDEN_CALLS = frozenset(
    {
        "__import__",
        "breakpoint",
        "compile",
        "eval",
        "exec",
        "input",
        "open",
    }
)


class CodeCandidateError(StrophaError):
    """Raised when a proposed code candidate violates the v1 contract."""


@dataclass(frozen=True)
class CodeCandidateSpec:
    """Validated identity and constructor config for one reranker artifact."""

    rel_path: str
    name: str
    source: str
    adapter_config: dict[str, Any]


def validate_code_candidate(
    config: dict[str, Any], artifacts: Any
) -> CodeCandidateSpec:
    """Validate an in-memory code candidate and return its normalized spec."""
    if not isinstance(artifacts, dict):
        raise CodeCandidateError(
            f"code artifacts must be a dict[str, str]; got {type(artifacts).__name__}"
        )
    if len(artifacts) != MAX_CODE_ARTIFACTS:
        raise CodeCandidateError("code candidate must contain exactly one artifact")

    rel_path, source = next(iter(artifacts.items()))
    if not isinstance(rel_path, str) or not isinstance(source, str):
        raise CodeCandidateError("artifact path and source must both be strings")
    match = _ARTIFACT_PATH_RE.fullmatch(rel_path)
    if match is None:
        raise CodeCandidateError(
            "artifact path must match adapters/reranker/<name>.py"
        )
    source_bytes = source.encode("utf-8")
    if not source.strip():
        raise CodeCandidateError("artifact source must not be empty")
    if len(source_bytes) > MAX_CODE_ARTIFACT_BYTES:
        raise CodeCandidateError(
            f"artifact exceeds {MAX_CODE_ARTIFACT_BYTES} UTF-8 bytes"
        )

    name = match.group("name")
    registered_name = validate_code_candidate_source(source)
    if registered_name != name:
        raise CodeCandidateError(
            f"artifact path name {name!r} does not match registration {registered_name!r}"
        )
    configured_name, adapter_config = _reranker_config(config)
    if configured_name != name:
        raise CodeCandidateError(
            f"candidate reranker {configured_name!r} does not match registration {name!r}"
        )
    return CodeCandidateSpec(rel_path, name, source, adapter_config)


def validate_code_candidate_directory(candidate_dir: Path) -> CodeCandidateSpec:
    """Validate the persisted config and sole Python artifact in a candidate."""
    import yaml  # type: ignore[import-untyped]

    adapters_dir = candidate_dir / "adapters"
    artifacts: dict[str, str] = {}
    if adapters_dir.is_dir():
        for path in adapters_dir.rglob("*"):
            if path.is_symlink():
                raise CodeCandidateError(f"candidate artifact must not be a symlink: {path}")
            if path.is_file():
                rel_path = path.relative_to(candidate_dir).as_posix()
                try:
                    artifacts[rel_path] = path.read_text(encoding="utf-8")
                except (OSError, UnicodeError) as exc:
                    raise CodeCandidateError(
                        f"failed to read candidate artifact {rel_path!r}: {exc}"
                    ) from exc

    config_path = candidate_dir / "config.yaml"
    if config_path.is_symlink():
        raise CodeCandidateError("candidate config must not be a symlink")
    try:
        raw_config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise CodeCandidateError(f"failed to read candidate config: {exc}") from exc
    if not isinstance(raw_config, dict):
        raise CodeCandidateError("candidate config must be a mapping")
    return validate_code_candidate(raw_config, artifacts)


def _reranker_config(config: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    value = config.get("reranker")
    if isinstance(value, str) and value:
        return value, {}
    if isinstance(value, dict):
        adapter = value.get("adapter")
        adapter_config = value.get("config", {})
        if isinstance(adapter, str) and adapter and isinstance(adapter_config, dict):
            return adapter, dict(adapter_config)
    raise CodeCandidateError(
        "candidate config must select the generated reranker by string or adapter mapping"
    )


def validate_code_candidate_source(source: str) -> str:
    """Apply syntax, import, call, and registration checks to one module."""
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        raise CodeCandidateError(f"artifact has invalid Python syntax: {exc}") from exc

    registrations: list[ast.Call] = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            modules = (
                [alias.name for alias in node.names]
                if isinstance(node, ast.Import)
                else [node.module or ""]
            )
            for module in modules:
                if module.split(".", 1)[0] in _FORBIDDEN_IMPORT_ROOTS:
                    raise CodeCandidateError(f"prohibited import in artifact: {module}")
        if isinstance(node, ast.Call):
            called_name = (
                node.func.id
                if isinstance(node.func, ast.Name)
                else node.func.attr if isinstance(node.func, ast.Attribute) else None
            )
            if called_name in _FORBIDDEN_CALLS:
                raise CodeCandidateError(f"prohibited call in artifact: {called_name}")
            if isinstance(node.func, ast.Name) and node.func.id == "register_adapter":
                registrations.append(node)

    if len(registrations) != 1:
        raise CodeCandidateError(
            "artifact must contain exactly one literal @register_adapter call"
        )
    registration = registrations[0]
    if registration.args or {kw.arg for kw in registration.keywords} != {"stage", "name"}:
        raise CodeCandidateError(
            "register_adapter must use only literal stage= and name= keywords"
        )
    values = {kw.arg: kw.value for kw in registration.keywords}
    stage = values["stage"]
    name = values["name"]
    if not isinstance(stage, ast.Constant) or stage.value != "reranker":
        raise CodeCandidateError("register_adapter stage must be the literal 'reranker'")
    if not isinstance(name, ast.Constant) or not isinstance(name.value, str):
        raise CodeCandidateError("register_adapter name must be a string literal")

    decorated = any(
        decorator is registration
        for node in tree.body
        if isinstance(node, ast.ClassDef)
        for decorator in node.decorator_list
    )
    if not decorated:
        raise CodeCandidateError("register_adapter must directly decorate a top-level class")
    return name.value


__all__ = [
    "MAX_CODE_ARTIFACTS",
    "MAX_CODE_ARTIFACT_BYTES",
    "CodeCandidateError",
    "CodeCandidateSpec",
    "validate_code_candidate",
    "validate_code_candidate_directory",
    "validate_code_candidate_source",
]
