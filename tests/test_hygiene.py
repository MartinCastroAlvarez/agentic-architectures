"""Package-wide guards: no dynamic execution, no processes, no silenced findings, stdlib only."""

from __future__ import annotations

import ast
import sys
from pathlib import Path

import agentic_architectures

PACKAGE = Path(agentic_architectures.__file__).parent
SOURCES = sorted(PACKAGE.rglob("*.py"))
BANNED_CALLS = {"eval", "exec", "compile", "__import__"}
BANNED_MODULES = {
    "pickle",
    "marshal",
    "shelve",
    "yaml",
    "subprocess",
    "ctypes",
    "socket",
    "requests",
}


def _imports(tree: ast.AST) -> set[str]:
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names |= {a.name.split(".")[0] for a in node.names}
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            names.add(node.module.split(".")[0])
    return names


def test_no_dynamic_execution_processes_or_unsafe_serialisation() -> None:
    assert len(SOURCES) > 20
    for path in SOURCES:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                assert node.func.id not in BANNED_CALLS, f"{path.name}: {node.func.id}()"
            if isinstance(node, ast.keyword):
                assert node.arg != "shell", f"{path.name}: shell="
        imported = _imports(tree)
        assert not imported & BANNED_MODULES, f"{path.name}: {imported & BANNED_MODULES}"
        assert "urllib.request" not in path.read_text("utf-8"), path.name


def test_the_package_depends_on_the_standard_library_only() -> None:
    allowed = set(sys.stdlib_module_names) | {"agentic_architectures"}
    for path in SOURCES:
        extra = _imports(ast.parse(path.read_text("utf-8"))) - allowed
        assert not extra, f"{path.name}: {extra}"


def test_only_the_providers_module_opens_network_connections() -> None:
    for path in SOURCES:
        if "http" in _imports(ast.parse(path.read_text("utf-8"))):
            assert path.name == "providers.py", path.name


def test_no_silenced_findings() -> None:
    for path in SOURCES:
        text = path.read_text(encoding="utf-8")
        for marker in ("# noqa", "# type: ignore", "# nosec"):
            assert marker not in text, f"{path.name}: {marker}"


def test_bundled_documents_ship_with_the_package() -> None:
    names = sorted(p.name for p in (PACKAGE / "data" / "docs").iterdir())
    assert names == [
        "agent-budgets.md",
        "idempotency.md",
        "ollama.md",
        "prompt-injection.md",
        "rate-limits.md",
        "units.md",
    ]
    assert (PACKAGE / "py.typed").exists()
