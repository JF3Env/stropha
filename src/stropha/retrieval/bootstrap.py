"""Environment bootstrap — repo snapshot for warm session start (WIN-227).

The paper (TerminalBench-2) shows agents burn 2-4 exploratory turns
re-discovering basic facts about a repo at the start of every
session: "what language is this?", "how do I build?", "how do I run
the tests?", "where's the entry point?". WIN-227 short-circuits that
by detecting the answers from top-level project files in O(filesystem
stat) and returning them in one call.

Detection strategy
------------------

Pattern-match against well-known marker files at the repo root. No
heuristics that need to parse files (parsing is reserved for the
chunker / search paths). Each detector is a small dataclass
declaring:

- ``marker``: file or glob the detector activates on.
- ``language``: the language tag to surface (e.g. ``"python"``).
- ``build_command``: idiomatic build / install command.
- ``test_command``: idiomatic test command.

Multiple detectors can match (a monorepo with Python + TS). The
caller gets every detected stack in priority order so the agent
sees the most-relevant one first.

Entry-point detection
---------------------

Looks for ``README.md`` / ``README.rst`` / ``README.txt`` plus a
small set of language-specific main files (``main.py`` /
``index.js`` / ``src/main.rs`` / ``cmd/main.go``). These are HINTS,
not guarantees — the agent should treat them as starting points
for deeper exploration.

What we don't do
----------------

- No filesystem walk beyond the top level (would shadow the
  walker's job).
- No content parsing.
- No network calls (no ``pip show`` / ``npm ls`` / etc.).

Cost is O(detector count × stat()) per call; the whole thing
finishes in single-digit milliseconds on a cold cache.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class StackDetector:
    """One stack to detect.

    A repo can have several detectors active (monorepo, polyglot).
    ``priority`` orders the matches when multiple fire so the
    agent's first-glance read is the most-likely primary stack.
    """

    name: str
    markers: tuple[str, ...]
    language: str
    build_command: str | None = None
    test_command: str | None = None
    run_command: str | None = None
    priority: int = 50


_DETECTORS: tuple[StackDetector, ...] = (
    # Python — `uv` is the preferred runner; fall back to plain
    # python -m pytest. ``pyproject.toml`` covers uv / poetry /
    # hatch / pdm. ``setup.py`` is the legacy fallback.
    StackDetector(
        name="python-uv",
        markers=("pyproject.toml", "uv.lock"),
        language="python",
        build_command="uv sync",
        test_command="uv run pytest",
        run_command="uv run python -m <module>",
        priority=80,
    ),
    StackDetector(
        name="python-pip",
        markers=("requirements.txt", "setup.py", "setup.cfg"),
        language="python",
        build_command="pip install -e .",
        test_command="pytest",
        run_command="python -m <module>",
        priority=60,
    ),
    # JavaScript / TypeScript
    StackDetector(
        name="node",
        markers=("package.json",),
        language="javascript",
        build_command="npm install",
        test_command="npm test",
        run_command="npm start",
        priority=70,
    ),
    StackDetector(
        name="pnpm",
        markers=("pnpm-lock.yaml",),
        language="javascript",
        build_command="pnpm install",
        test_command="pnpm test",
        run_command="pnpm start",
        priority=75,
    ),
    StackDetector(
        name="yarn",
        markers=("yarn.lock",),
        language="javascript",
        build_command="yarn install",
        test_command="yarn test",
        run_command="yarn start",
        priority=72,
    ),
    # Rust
    StackDetector(
        name="rust",
        markers=("Cargo.toml",),
        language="rust",
        build_command="cargo build",
        test_command="cargo test",
        run_command="cargo run",
        priority=80,
    ),
    # Go
    StackDetector(
        name="go",
        markers=("go.mod",),
        language="go",
        build_command="go build ./...",
        test_command="go test ./...",
        run_command="go run .",
        priority=80,
    ),
    # Java
    StackDetector(
        name="java-gradle",
        markers=(
            "build.gradle", "build.gradle.kts",
            "gradlew", "settings.gradle",
        ),
        language="java",
        build_command="./gradlew build",
        test_command="./gradlew test",
        run_command="./gradlew run",
        priority=70,
    ),
    StackDetector(
        name="java-maven",
        markers=("pom.xml",),
        language="java",
        build_command="mvn package",
        test_command="mvn test",
        run_command="mvn exec:java",
        priority=65,
    ),
    # Kotlin (when not Java-shaped)
    StackDetector(
        name="kotlin",
        markers=("build.gradle.kts",),
        language="kotlin",
        build_command="./gradlew build",
        test_command="./gradlew test",
        priority=60,
    ),
    # Ruby
    StackDetector(
        name="ruby",
        markers=("Gemfile",),
        language="ruby",
        build_command="bundle install",
        test_command="bundle exec rspec",
        priority=60,
    ),
    # .NET
    StackDetector(
        name="dotnet",
        markers=("*.csproj", "*.sln"),
        language="csharp",
        build_command="dotnet build",
        test_command="dotnet test",
        run_command="dotnet run",
        priority=60,
    ),
)


# Entry-point hints. Per-language list of paths (relative to repo
# root) the agent might want to read first.
_ENTRY_POINT_HINTS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("readme", (
        "README.md", "README.rst", "README.txt", "README",
    )),
    ("python", ("main.py", "src/main.py", "__main__.py", "app.py")),
    ("javascript", ("index.js", "index.ts", "src/index.ts", "src/index.js")),
    ("rust", ("src/main.rs", "src/lib.rs")),
    ("go", ("main.go", "cmd/main.go")),
    ("docs", (
        "CLAUDE.md", "AGENTS.md", "CONTRIBUTING.md",
        "docs/README.md", "docs/index.md",
    )),
)


@dataclass(frozen=True)
class DetectedStack:
    """One stack that fired during detection."""

    name: str
    language: str
    build_command: str | None
    test_command: str | None
    run_command: str | None
    matched_markers: tuple[str, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class BootstrapSnapshot:
    """The full per-repo bootstrap payload returned to the agent."""

    repo_path: str
    stacks: list[DetectedStack] = field(default_factory=list)
    """Detected language/build stacks, sorted by priority DESC
    (highest priority = most-likely primary stack)."""
    entry_points: dict[str, list[str]] = field(default_factory=dict)
    """``{category: [paths]}`` — found readmes, mains, docs."""
    has_git: bool = False
    has_tests_dir: bool = False
    has_docs_dir: bool = False
    has_graphify_out: bool = False


def _matches_marker(root: Path, marker: str) -> str | None:
    """Return the first matching path (relative to ``root``) for the
    marker, or ``None`` when nothing matches.

    Marker may contain a single ``*`` glob (e.g. ``*.csproj``). For
    plain names we do a single stat — the cheap path. For globs we
    list the directory (also cheap on the typical repo root).
    """
    if "*" in marker:
        for child in root.iterdir():
            if child.is_file() and child.match(marker):
                return child.name
        return None
    candidate = root / marker
    if candidate.exists():
        return marker
    return None


def detect_stacks(root: Path) -> list[DetectedStack]:
    """Return every stack whose markers match at ``root``.

    Sorted by ``priority`` descending so the agent sees the
    most-likely primary stack first.
    """
    out: list[DetectedStack] = []
    for det in _DETECTORS:
        matched: list[str] = []
        for marker in det.markers:
            hit = _matches_marker(root, marker)
            if hit is not None:
                matched.append(hit)
        if matched:
            out.append(
                DetectedStack(
                    name=det.name,
                    language=det.language,
                    build_command=det.build_command,
                    test_command=det.test_command,
                    run_command=det.run_command,
                    matched_markers=tuple(matched),
                )
            )
    # Sort by detector priority. We re-look up the priority from the
    # original detector list to keep DetectedStack lean.
    priority_map = {d.name: d.priority for d in _DETECTORS}
    out.sort(key=lambda s: priority_map.get(s.name, 0), reverse=True)
    return out


def detect_entry_points(root: Path) -> dict[str, list[str]]:
    """Return a category→paths map of known entry-point hints."""
    found: dict[str, list[str]] = {}
    for category, candidates in _ENTRY_POINT_HINTS:
        hits = [c for c in candidates if (root / c).exists()]
        if hits:
            found[category] = hits
    return found


def detect_bootstrap(root: Path) -> BootstrapSnapshot:
    """Build a :class:`BootstrapSnapshot` for ``root``.

    All checks are pure ``Path.exists`` / ``Path.iterdir`` calls;
    no content reads, no subprocesses. Fast enough to call on every
    MCP session open.
    """
    root = root.expanduser()
    if not root.is_dir():
        return BootstrapSnapshot(
            repo_path=str(root),
            stacks=[],
            entry_points={},
        )
    root = root.resolve()
    return BootstrapSnapshot(
        repo_path=str(root),
        stacks=detect_stacks(root),
        entry_points=detect_entry_points(root),
        has_git=(root / ".git").exists(),
        has_tests_dir=(
            (root / "tests").is_dir() or (root / "test").is_dir()
        ),
        has_docs_dir=(root / "docs").is_dir(),
        has_graphify_out=(root / "graphify-out").is_dir(),
    )


__all__ = [
    "BootstrapSnapshot",
    "DetectedStack",
    "StackDetector",
    "detect_bootstrap",
    "detect_entry_points",
    "detect_stacks",
]
