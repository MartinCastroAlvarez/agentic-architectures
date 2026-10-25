"""Agentic RAG: decide to retrieve, grade the passages, retry the query, cite only what was found."""

from agentic_architectures.patterns import agentic_rag
from examples._common import IDEMPOTENCY
from examples._common import PORT
from examples._common import runtime
from examples._common import show


def main() -> None:
    cited = show(agentic_rag.run(runtime(), PORT))
    print("queries:", cited.data["queries"], "citations:", cited.data["citations"])
    direct = show(agentic_rag.run(runtime(), IDEMPOTENCY))
    print("queries:", direct.data["queries"])


if __name__ == "__main__":
    main()
