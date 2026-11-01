"""Ask questions about the indexed PDFs.

    python ask.py "What is Self-RAG?"     # one question
    python ask.py                          # interactive mode
    python ask.py -k 8 --show-context "..."  # more excerpts, print what was retrieved
"""

import argparse
import sys

from rag.answer import ask


def answer_one(question: str, k: int, show_context: bool) -> None:
    result = ask(question, k)
    print(f"\n{result.text}\n")
    if result.unsupported_citations:
        print("Warning: citations not among retrieved excerpts:", ", ".join(result.unsupported_citations))
    if show_context:
        print("Retrieved excerpts:")
        for h in result.hits:
            print(f"  {h.similarity:.2f}  {h.citation}  {h.text[:100]}...")
        print()


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="Ask questions about your PDFs (RAG).")
    parser.add_argument("question", nargs="*", help="question; omit for interactive mode")
    parser.add_argument("-k", type=int, default=None, help="number of chunks to retrieve (default: TOP_K)")
    parser.add_argument("--show-context", action="store_true", help="print the retrieved chunks and scores")
    args = parser.parse_args()

    from rag import config

    k = args.k or config.TOP_K
    if args.question:
        answer_one(" ".join(args.question), k, args.show_context)
        return

    print("Ask a question about the PDFs (empty line or Ctrl+C to quit).")
    try:
        while question := input("\n> ").strip():
            answer_one(question, k, args.show_context)
    except (KeyboardInterrupt, EOFError):
        pass


if __name__ == "__main__":
    main()
