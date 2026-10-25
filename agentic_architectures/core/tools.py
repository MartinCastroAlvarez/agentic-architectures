"""The ``Tool`` protocol and the sandboxed tools the patterns share.

Tool *input* comes from the model, so it is untrusted: the registry validates argument
names and sizes before a tool runs, a tool never raises into the pattern (errors come
back as text the model can read), and output is capped. Tool *output* is untrusted too:
the runtime wraps it with ``parsing.untrusted`` before it reaches a prompt.

* ``Calculator`` evaluates arithmetic by walking the AST against an allowlist (no
  ``eval``), with bounded size, exponents and results.
* ``SearchDocs`` queries a :class:`~agentic_architectures.core.retrieval.Retriever`.
* ``ReadFile`` / ``ListFiles`` / ``WriteNote`` are confined to one data directory:
  paths are resolved and must stay inside it, symlinks and hidden files are refused,
  sizes are capped. ``WriteNote`` has a side effect, so it ``requires_approval``.
"""

from __future__ import annotations

import ast
import math
import operator
import re
from collections.abc import Callable
from collections.abc import Iterable
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from typing import Protocol
from typing import runtime_checkable

from agentic_architectures.core.retrieval import Retriever

MAX_ARG_CHARS = 2000
MAX_OUTPUT_CHARS = 4000


class ToolError(Exception):
    """A tool refused its input. The message is shown to the model."""


@runtime_checkable
class Tool(Protocol):
    name: str
    description: str
    requires_approval: bool

    @property
    def parameters(self) -> Mapping[str, str]: ...

    def run(self, args: Mapping[str, str]) -> str: ...


# --------------------------------------------------------------------------- #
# Calculator
# --------------------------------------------------------------------------- #
_BIN: dict[type[ast.operator], Callable[[Any, Any], Any]] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_UNARY: dict[type[ast.unaryop], Callable[[Any], Any]] = {ast.UAdd: operator.pos, ast.USub: operator.neg}
_FUNCS: dict[str, Callable[..., float]] = {
    "sqrt": math.sqrt,
    "log": math.log,
    "log10": math.log10,
    "exp": math.exp,
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
    "abs": abs,
    "round": round,
    "min": min,
    "max": max,
}
_CONSTS = {"pi": math.pi, "e": math.e}


def calculate(expression: str, *, max_chars: int = 200, max_nodes: int = 64) -> float | int:
    if len(expression) > max_chars:
        raise ToolError(f"expression longer than {max_chars} characters")
    try:
        tree = ast.parse(expression.strip(), mode="eval")
    except SyntaxError as exc:
        raise ToolError("not an arithmetic expression") from exc
    if sum(1 for _ in ast.walk(tree)) > max_nodes:
        raise ToolError("expression too complex")

    def walk(node: ast.AST) -> Any:
        if isinstance(node, ast.Expression):
            return walk(node.body)
        if isinstance(node, ast.Constant) and type(node.value) in (int, float):
            return node.value
        if isinstance(node, ast.Name) and node.id in _CONSTS:
            return _CONSTS[node.id]
        if isinstance(node, ast.BinOp) and type(node.op) in _BIN:
            left, right = walk(node.left), walk(node.right)
            if isinstance(node.op, ast.Pow) and (abs(right) > 64 or abs(left) > 1e6):
                raise ToolError("exponent too large")
            return _BIN[type(node.op)](left, right)
        if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY:
            return _UNARY[type(node.op)](walk(node.operand))
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id in _FUNCS
            and not node.keywords
            and len(node.args) <= 4
        ):
            return _FUNCS[node.func.id](*(walk(a) for a in node.args))
        raise ToolError(f"not allowed: {type(node).__name__}")

    try:
        result = walk(tree)
    except (ArithmeticError, ValueError, TypeError) as exc:
        raise ToolError(f"math error: {type(exc).__name__}") from exc
    if type(result) not in (int, float):
        raise ToolError("result is not a real number")
    if isinstance(result, float) and (math.isnan(result) or math.isinf(result)):
        raise ToolError("result is not finite")
    if abs(result) > 1e15:
        raise ToolError("result too large")
    if isinstance(result, float) and result.is_integer():
        return int(result)
    return round(result, 10) if isinstance(result, float) else result


@dataclass
class Calculator:
    name: str = "calculator"
    description: str = "Evaluate an arithmetic expression, e.g. (120 + 80) / 2. Returns the number."
    requires_approval: bool = False

    @property
    def parameters(self) -> Mapping[str, str]:
        return {"expression": "arithmetic expression"}

    def run(self, args: Mapping[str, str]) -> str:
        return str(calculate(args.get("expression", "")))


# --------------------------------------------------------------------------- #
# Retrieval
# --------------------------------------------------------------------------- #
@dataclass
class SearchDocs:
    retriever: Retriever
    k: int = 3
    name: str = "search"
    description: str = "Keyword search over the local documents. Returns [doc-id] passages."
    requires_approval: bool = False

    @property
    def parameters(self) -> Mapping[str, str]:
        return {"query": "keywords to search for"}

    def run(self, args: Mapping[str, str]) -> str:
        hits = self.retriever.search(args.get("query", ""), self.k)
        if not hits:
            return "no results"
        return "\n".join(f"[{h.doc.id}] {h.doc.text.strip()[:600]}" for h in hits)


# --------------------------------------------------------------------------- #
# Files, confined to one directory
# --------------------------------------------------------------------------- #
SAFE_NAME = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9_-]{0,63}$")


def confined(root: Path, relative: str) -> Path:
    """Resolve ``relative`` inside ``root`` or raise ``ToolError``."""
    if not relative or "\x00" in relative:
        raise ToolError("empty path")
    candidate = Path(relative)
    if candidate.is_absolute() or any(p in {"..", ""} or p.startswith(".") for p in candidate.parts):
        raise ToolError("path must be relative, without '..' or hidden parts")
    base = root.resolve(strict=True)
    path = base / candidate
    # Refuse a symlink anywhere along the path, then check where it really lands.
    probe = base
    for part in candidate.parts:
        probe = probe / part
        if probe.is_symlink():
            raise ToolError("symlinks are not followed")
    if not path.resolve().is_relative_to(base):
        raise ToolError("path escapes the data directory")
    return path


@dataclass
class ReadFile:
    root: Path
    max_bytes: int = 20_000
    name: str = "read_file"
    description: str = "Read a text file from the data directory by relative path."
    requires_approval: bool = False

    @property
    def parameters(self) -> Mapping[str, str]:
        return {"path": "relative path inside the data directory"}

    def run(self, args: Mapping[str, str]) -> str:
        path = confined(self.root, args.get("path", ""))
        if not path.is_file():
            raise ToolError("no such file")
        if path.stat().st_size > self.max_bytes:
            raise ToolError("file too large")
        return path.read_text("utf-8", errors="replace")


@dataclass
class ListFiles:
    root: Path
    name: str = "list_files"
    description: str = "List the files in the data directory."
    requires_approval: bool = False

    @property
    def parameters(self) -> Mapping[str, str]:
        return {}

    def run(self, args: Mapping[str, str]) -> str:
        base = self.root.resolve(strict=True)
        names = [
            p.relative_to(base).as_posix()
            for p in sorted(base.rglob("*"))
            if p.is_file()
            and not p.is_symlink()
            and not any(part.startswith(".") for part in p.relative_to(base).parts)
        ]
        return "\n".join(names[:200]) or "(empty)"


@dataclass
class WriteNote:
    """Write ``notes/<name>.md`` in the data directory. A side effect: needs approval."""

    root: Path
    max_chars: int = 4000
    name: str = "write_note"
    description: str = "Save a short Markdown note as notes/<name>.md (needs human approval)."
    requires_approval: bool = True

    @property
    def parameters(self) -> Mapping[str, str]:
        return {"name": "file name, letters/digits/-/_", "text": "note body"}

    def run(self, args: Mapping[str, str]) -> str:
        name = args.get("name", "")
        if not SAFE_NAME.match(name):
            raise ToolError("invalid note name")
        text = args.get("text", "")
        if len(text) > self.max_chars:
            raise ToolError("note too long")
        notes = self.root.resolve(strict=True) / "notes"
        if notes.is_symlink():
            raise ToolError("symlinks are not followed")
        notes.mkdir(exist_ok=True)
        path = confined(self.root, f"notes/{name}.md")
        path.write_text(text, "utf-8")
        return f"saved notes/{name}.md ({len(text)} chars)"


# --------------------------------------------------------------------------- #
# Registry
# --------------------------------------------------------------------------- #
class ToolRegistry:
    def __init__(self, tools: Iterable[Tool] = ()) -> None:
        self._tools: dict[str, Tool] = {}
        for tool in tools:
            self.add(tool)

    def add(self, tool: Tool) -> None:
        if not re.fullmatch(r"[a-z][a-z0-9_]{0,31}", tool.name):
            raise ValueError(f"bad tool name {tool.name!r}")
        self._tools[tool.name] = tool

    def __contains__(self, name: object) -> bool:
        return name in self._tools

    def __len__(self) -> int:
        return len(self._tools)

    def names(self) -> list[str]:
        return sorted(self._tools)

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def subset(self, names: Iterable[str]) -> ToolRegistry:
        """A registry with only ``names``: give each role the least tools it needs."""
        wanted = set(names)
        return ToolRegistry(t for n, t in self._tools.items() if n in wanted)

    def catalog(self) -> str:
        if not self._tools:
            return "(no tools)"
        lines = []
        for name in self.names():
            tool = self._tools[name]
            params = ", ".join(tool.parameters)
            lines.append(f"- {name}({params}): {tool.description}")
        return "\n".join(lines)

    def validate(self, name: str, args: Any) -> tuple[Tool, dict[str, str]]:
        tool = self._tools.get(name)
        if tool is None:
            raise ToolError(f"unknown tool {name!r}; available: {', '.join(self.names()) or 'none'}")
        if not isinstance(args, Mapping):
            raise ToolError("args must be a JSON object")
        unknown = set(args) - set(tool.parameters)
        if unknown:
            raise ToolError(f"unknown argument(s): {', '.join(sorted(map(str, unknown)))}")
        clean: dict[str, str] = {}
        for key, value in args.items():
            text = value if isinstance(value, str) else str(value)
            if len(text) > MAX_ARG_CHARS:
                raise ToolError(f"argument {key!r} too long")
            clean[str(key)] = text
        return tool, clean

    def run(self, name: str, args: Any) -> tuple[bool, str]:
        """Run a tool. Returns ``(ok, output)``; never raises for bad model input."""
        try:
            tool, clean = self.validate(name, args)
            output = tool.run(clean)
        except ToolError as exc:
            return False, f"error: {exc}"
        except OSError as exc:
            return False, f"error: {type(exc).__name__}"
        if len(output) > MAX_OUTPUT_CHARS:
            output = output[:MAX_OUTPUT_CHARS] + " [truncated]"
        return True, output
