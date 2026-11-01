"""Retrieve -> prompt -> LLM answer with [filename, p. N] citations."""

import re
from dataclasses import dataclass

from . import config
from .llm import complete
from .store import Hit, search

SYSTEM_PROMPT = f"""You answer questions using ONLY the numbered context excerpts from PDF documents.

Rules:
- Every factual sentence must end with a citation copied exactly from the excerpt header, e.g. [paper.pdf, p. 3]. \
Cite several excerpts if needed: [a.pdf, p. 2] [b.pdf, p. 7].
- Use only citations that appear in the context. Never invent a filename or page.
- Do not use outside knowledge. If the excerpts do not contain the answer, reply with exactly:
{config.NOT_FOUND}
- If the excerpts only partly answer the question, answer that part and say what is missing.
- Be concise: a few sentences or a short list."""

CITATION_RE = re.compile(r"\[([^\[\],]+?\.pdf), p\. (\d+)\]")


@dataclass
class Answer:
    text: str
    hits: list[Hit]  # what was retrieved (and shown to the LLM, unless nothing was relevant)
    used_llm: bool

    @property
    def citations(self) -> list[str]:
        """Distinct citations in the answer, in order of appearance."""
        return list(dict.fromkeys(f"[{f}, p. {p}]" for f, p in CITATION_RE.findall(self.text)))

    @property
    def unsupported_citations(self) -> list[str]:
        """Citations the LLM made up: not among the retrieved excerpts."""
        retrieved = {h.citation for h in self.hits}
        return [c for c in self.citations if c not in retrieved]


def build_prompt(question: str, hits: list[Hit]) -> str:
    context = "\n\n".join(f"### Excerpt {i} {h.citation}\n{h.text}" for i, h in enumerate(hits, start=1))
    return f"Context:\n\n{context}\n\nQuestion: {question}"


def ask(question: str, k: int = config.TOP_K) -> Answer:
    hits = search(question, k)
    # Nothing even remotely similar: don't let the LLM guess.
    if not hits or hits[0].similarity < config.MIN_SIMILARITY:
        return Answer(config.NOT_FOUND, hits, used_llm=False)

    text = complete(SYSTEM_PROMPT, build_prompt(question, hits))
    if text.lower().startswith(config.NOT_FOUND.rstrip(".").lower()):
        text = config.NOT_FOUND  # drop any rambling after the refusal
    return Answer(text, hits, used_llm=True)
