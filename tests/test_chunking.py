"""Unit tests for text cleaning and chunking. No model or vector store needed."""

from rag.ingest import clean, split_text


def test_clean_fixes_ligatures_hyphenation_and_whitespace():
    raw = "The ﬁrst  step of the al-\ngorithm\n\nis   simple."
    assert clean(raw) == "The first step of the algorithm is simple."


def test_clean_keeps_real_hyphens():
    assert clean("state-of-the-art results") == "state-of-the-art results"


def test_short_text_is_a_single_chunk():
    assert split_text("One sentence. Another one.", size=800, overlap=150) == ["One sentence. Another one."]


def test_chunks_respect_the_size_limit():
    text = " ".join(f"This is sentence number {i} of the test text." for i in range(200))
    chunks = split_text(text, size=300, overlap=60)
    assert len(chunks) > 1
    assert all(len(c) <= 300 for c in chunks)


def test_chunks_overlap_by_whole_sentences():
    sentences = [f"Sentence {i} talks about topic {i}." for i in range(40)]
    chunks = split_text(" ".join(sentences), size=200, overlap=70)
    for prev, nxt in zip(chunks, chunks[1:]):
        first_sentence_of_next = nxt.split(". ")[0] + "."
        # The next chunk starts with a sentence the previous chunk ended with.
        assert first_sentence_of_next in prev


def test_no_text_is_lost():
    sentences = [f"Fact {i} is important." for i in range(60)]
    chunks = split_text(" ".join(sentences), size=150, overlap=40)
    joined = " ".join(chunks)
    assert all(s in joined for s in sentences)


def test_overlong_sentence_is_cut_on_word_boundaries():
    words = [f"word{i}" for i in range(300)]
    chunks = split_text(" ".join(words), size=100, overlap=20)
    assert all(len(c) <= 100 for c in chunks)
    # Every word survives intact (no word split in half).
    found = set(" ".join(chunks).split())
    assert set(words) <= found
