"""Minimal web UI.

    streamlit run app.py
"""

import streamlit as st

from rag import config
from rag.answer import ask
from rag.llm import pick_provider

st.set_page_config(page_title="Ask Your PDFs", page_icon="📄")
st.title("📄 Ask Your PDFs")

pdfs = sorted(p.name for p in config.PDF_DIR.glob("*.pdf"))
provider, model = pick_provider()
st.caption(f"{len(pdfs)} PDFs indexed · LLM: {provider} / {model} · embeddings: {config.EMBED_MODEL}")

with st.sidebar:
    k = st.slider("Chunks to retrieve (top-k)", 1, 15, config.TOP_K)
    st.markdown("**Documents**\n" + "\n".join(f"- {name}" for name in pdfs))

with st.form("ask"):
    question = st.text_input("Question", placeholder="What are the four functions of the AI RMF Core?")
    submitted = st.form_submit_button("Ask")

if submitted and question.strip():
    with st.spinner("Searching and answering..."):
        result = ask(question.strip(), k)

    st.markdown(result.text)
    if result.unsupported_citations:
        st.warning("Citations not among the retrieved excerpts: " + ", ".join(result.unsupported_citations))

    with st.expander(f"Retrieved excerpts ({len(result.hits)})"):
        for h in result.hits:
            st.markdown(f"**{h.citation}** · similarity {h.similarity:.2f}")
            st.text(h.text)
