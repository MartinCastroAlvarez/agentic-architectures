"""Every major agentic architecture, side by side, on one tiny framework-free runtime."""

from __future__ import annotations

import tempfile
from pathlib import Path

from agentic_architectures.core.budget import Budget
from agentic_architectures.core.model import ChatModel
from agentic_architectures.core.retrieval import BUNDLED_DOCS
from agentic_architectures.core.retrieval import Retriever
from agentic_architectures.core.retrieval import load_documents
from agentic_architectures.core.runtime import Result
from agentic_architectures.core.runtime import Runtime
from agentic_architectures.core.tools import Calculator
from agentic_architectures.core.tools import ListFiles
from agentic_architectures.core.tools import ReadFile
from agentic_architectures.core.tools import SearchDocs
from agentic_architectures.core.tools import ToolRegistry
from agentic_architectures.core.tools import WriteNote
from agentic_architectures.core.trace import Tracer

__version__ = "0.1.0"


def build_runtime(
    model: ChatModel,
    *,
    docs_dir: Path | str = BUNDLED_DOCS,
    workspace: Path | str | None = None,
    budget: Budget | None = None,
    echo: bool = False,
    max_workers: int = 4,
) -> Runtime:
    """A runtime with the standard tools.

    ``docs_dir`` is searched and readable (read-only); ``workspace`` is the only place a
    tool may write (``write_note``, approval-gated). It defaults to a fresh temp folder.
    """
    docs = Path(docs_dir)
    retriever = Retriever(load_documents(docs))
    space = Path(workspace) if workspace else Path(tempfile.mkdtemp(prefix="agentic-"))
    space.mkdir(parents=True, exist_ok=True)
    tools = ToolRegistry(
        [Calculator(), SearchDocs(retriever), ReadFile(docs), ListFiles(docs), WriteNote(space)]
    )
    return Runtime(
        model,
        tools,
        retriever,
        budget or Budget(),
        Tracer(echo=echo),
        max_workers,
    )


__all__ = ["Budget", "Result", "Runtime", "__version__", "build_runtime"]
