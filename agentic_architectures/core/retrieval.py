"""A tiny BM25 retriever over a folder of Markdown/text documents.

Good enough to demonstrate retrieval-shaped patterns without an embedding model or a
vector database. The loader is confined: it resolves the root first, reads only regular
``.md``/``.txt`` files inside it, skips symlinks and hidden entries, and caps size and
count.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

TOKEN = re.compile(r"[a-z0-9]+")
_STOP_WORDS = (
    "a an and are as at be by for from how in is it of on or that the this to what when "
    "which who why with does do i my me you your"
)
STOP = frozenset(_STOP_WORDS.split())
MAX_FILE_BYTES = 200_000
MAX_FILES = 500
BUNDLED_DOCS = Path(__file__).resolve().parent.parent / "data" / "docs"


def tokenize(text: str) -> list[str]:
    return [t for t in TOKEN.findall(text.lower()) if t not in STOP]


@dataclass(frozen=True, slots=True)
class Document:
    id: str
    text: str


@dataclass(frozen=True, slots=True)
class Hit:
    doc: Document
    score: float


class Retriever:
    def __init__(self, docs: list[Document], *, k1: float = 1.5, b: float = 0.75) -> None:
        self.docs = docs
        self.k1 = k1
        self.b = b
        self._tfs = [Counter(tokenize(d.text)) for d in docs]
        self._lens = [sum(tf.values()) for tf in self._tfs]
        self._avg = (sum(self._lens) / len(self._lens)) if docs else 0.0
        df: Counter[str] = Counter()
        for tf in self._tfs:
            df.update(tf.keys())
        n = len(docs)
        self._idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}

    def search(self, query: str, k: int = 3) -> list[Hit]:
        terms = tokenize(query)
        scored: list[Hit] = []
        for doc, tf, length in zip(self.docs, self._tfs, self._lens, strict=True):
            score = 0.0
            for t in terms:
                if t not in tf:
                    continue
                f = tf[t]
                norm = self.k1 * (1 - self.b + self.b * length / (self._avg or 1))
                score += self._idf[t] * f * (self.k1 + 1) / (f + norm)
            if score > 0:
                scored.append(Hit(doc, round(score, 4)))
        scored.sort(key=lambda h: (-h.score, h.doc.id))
        return scored[: max(0, k)]

    def get(self, doc_id: str) -> Document | None:
        return next((d for d in self.docs if d.id == doc_id), None)


def load_documents(root: Path | str = BUNDLED_DOCS) -> list[Document]:
    base = Path(root).resolve(strict=True)
    if not base.is_dir():
        raise NotADirectoryError(str(base))
    docs: list[Document] = []
    for path in sorted(base.rglob("*")):
        rel = path.relative_to(base)
        if any(part.startswith(".") for part in rel.parts) or path.is_symlink():
            continue
        if path.suffix not in {".md", ".txt"} or not path.is_file():
            continue
        if not path.resolve().is_relative_to(base) or path.stat().st_size > MAX_FILE_BYTES:
            continue
        docs.append(Document(rel.with_suffix("").as_posix(), path.read_text("utf-8")))
        if len(docs) >= MAX_FILES:
            break
    return docs
