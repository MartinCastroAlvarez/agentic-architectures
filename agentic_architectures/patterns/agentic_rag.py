"""Agentic RAG: the model decides whether and what to retrieve, grades the results, retries, cites.

Plain RAG always retrieves once with the raw question. Here a decision call chooses
whether retrieval is needed and writes the search query; after each search a grading
call keeps only the passages that are relevant, and, if none are, writes a better
query (up to ``max_queries``). The answer call sees only graded passages and must cite
them; code drops citations of documents that were not retrieved and reports
``no_evidence`` rather than letting the model answer from nothing.

When to use
    Knowledge-heavy questions over a corpus where queries need rewriting, some
    questions need no retrieval, and answers must be grounded and cited. The same loop
    works over tables (Table-Augmented Generation: retrieve rows with SQL instead of
    passages; see langchain-tag-agent).

Trade-offs
    Better recall and precision than one-shot RAG and honest "I don't know" answers, at
    2-3 extra calls. Grading by the model is itself fallible; a cross-encoder reranker
    is a cheaper grader when you have one.

Cost profile
    1 decide + (1 grade per query, 1-3 queries) + 1 answer: 3-5 calls.

Failure modes
    Query drift (rewrites wander off topic), a grader that rejects everything (bounded,
    then ``no_evidence``), citations to passages that were never retrieved (filtered),
    and injected instructions inside documents (delimited as untrusted).

Sources
    Asai et al., "Self-RAG" (ICLR 2024). Yan et al., "Corrective RAG" (2024). Singh et
    al., "Agentic Retrieval-Augmented Generation: A Survey" (2025). Biswal et al.,
    "Text2SQL is Not Enough: Unifying AI and Databases with TAG" (2024). See
    adk-vertex-vector-search and local-rag-ollama.
"""

from __future__ import annotations

import re
from typing import Any

from agentic_architectures.core.parsing import as_list
from agentic_architectures.core.parsing import as_str
from agentic_architectures.core.parsing import untrusted
from agentic_architectures.core.retrieval import Hit
from agentic_architectures.core.runtime import Partial
from agentic_architectures.core.runtime import Result
from agentic_architectures.core.runtime import Runtime
from agentic_architectures.core.runtime import run_pattern

NAME = "agentic_rag"
DECIDE = (
    "Decide whether answering needs the local documents. Reply with JSON: "
    '{"retrieve": true|false, "query": "<keyword search query>"}'
)
GRADE = (
    "Which passages help answer the question? If none do, propose a better keyword query. "
    'Reply with JSON: {"relevant": ["<doc-id>", ...], "rewrite": "<new query or empty>"}'
)
ANSWER = (
    "Answer using only the passages, citing them as [doc-id]. If they do not contain the "
    'answer, say so. Reply with JSON: {"answer": "...", "citations": ["<doc-id>", ...]}'
)
CITE = re.compile(r"\[([a-zA-Z0-9_./-]{1,80})\]")


def _run(
    rt: Runtime,
    task: str,
    *,
    partial: Partial,
    max_queries: int = 3,
    k: int = 3,
) -> tuple[str, str, dict[str, Any]]:
    decision = rt.llm_json("rag:decide", DECIDE, untrusted("question", task))
    queries: list[str] = []
    kept: dict[str, Hit] = {}
    if decision.get("retrieve") is not False:
        query = as_str(decision.get("query"), task, limit=300).strip() or task
        for _ in range(max_queries):
            rt.budget.step()
            queries.append(query)
            hits = rt.retrieve(query, k)
            if hits:
                passages = "\n".join(untrusted(f"doc:{h.doc.id}", h.doc.text[:1500]) for h in hits)
                grade = rt.llm_json("rag:grade", GRADE, f"Question: {task}\n\nPassages:\n{passages}")
                ids = {h.doc.id: h for h in hits}
                for doc_id in as_list(grade.get("relevant")):
                    if str(doc_id) in ids:
                        kept[str(doc_id)] = ids[str(doc_id)]
                rewrite = as_str(grade.get("rewrite"), limit=300).strip()
            else:
                rewrite = task  # nothing matched: fall back to the question itself
            rt.note("grade", "rag", f"{query!r}: kept {sorted(kept)}")
            if kept:
                break
            if not rewrite or rewrite in queries:
                break
            query = rewrite
        if not kept:
            return "", "no_evidence", {"queries": queries, "citations": []}
    context = (
        "\n".join(untrusted(f"doc:{d}", h.doc.text[:1500]) for d, h in kept.items())
        or "(no retrieval needed)"
    )
    data = rt.llm_json("rag:answer", ANSWER, f"Question: {task}\n\nPassages:\n{context}")
    answer = as_str(data.get("answer")).strip()
    cited = {str(c) for c in as_list(data.get("citations"))} | set(CITE.findall(answer))
    citations = sorted(c for c in cited if c in kept)
    dropped = sorted(cited - set(citations))
    if dropped:
        rt.note("guard", "citations", f"dropped unretrieved {dropped}")
    partial.offer(answer)
    return answer, "answered", {"queries": queries, "citations": citations, "dropped": dropped}


def run(rt: Runtime, task: str, **options: Any) -> Result:
    return run_pattern(NAME, _run, rt, task, **options)
