"""Unit tests for prompt building and citation checking. No LLM calls."""

from rag.answer import Answer, build_prompt
from rag.store import Hit


def hit(source: str, page: int, text: str = "some text", similarity: float = 0.8) -> Hit:
    return Hit(text=text, source=source, page=page, similarity=similarity)


def test_hit_citation_format():
    assert hit("paper.pdf", 3).citation == "[paper.pdf, p. 3]"


def test_prompt_labels_every_excerpt_with_its_citation():
    prompt = build_prompt("What is X?", [hit("a.pdf", 1, "Alpha"), hit("b.pdf", 7, "Beta")])
    assert "### Excerpt 1 [a.pdf, p. 1]\nAlpha" in prompt
    assert "### Excerpt 2 [b.pdf, p. 7]\nBeta" in prompt
    assert prompt.endswith("Question: What is X?")


def test_citations_are_extracted_in_order_without_duplicates():
    answer = Answer(
        "X is a thing [b.pdf, p. 7]. It also does Y [a.pdf, p. 1] [b.pdf, p. 7].",
        hits=[hit("a.pdf", 1), hit("b.pdf", 7)],
        used_llm=True,
    )
    assert answer.citations == ["[b.pdf, p. 7]", "[a.pdf, p. 1]"]
    assert answer.unsupported_citations == []


def test_citation_that_was_not_retrieved_is_flagged():
    answer = Answer(
        "X is a thing [a.pdf, p. 1]. Also [a.pdf, p. 99].",
        hits=[hit("a.pdf", 1)],
        used_llm=True,
    )
    assert answer.unsupported_citations == ["[a.pdf, p. 99]"]


def test_filenames_with_spaces_and_dashes_are_recognised():
    answer = Answer("See [my report-2024 v2.pdf, p. 12].", hits=[hit("my report-2024 v2.pdf", 12)], used_llm=True)
    assert answer.citations == ["[my report-2024 v2.pdf, p. 12]"]
    assert answer.unsupported_citations == []


def test_text_without_citations():
    assert Answer("I don't know based on the provided documents.", hits=[], used_llm=False).citations == []
