"""Measure how well the pipeline works on a fixed set of questions (eval/questions.json).

    python evaluate.py          # retrieval only: fast, free, no API key needed
    python evaluate.py --llm    # also generate answers and check them (calls the LLM)
    python evaluate.py --mode vector    # compare search modes: vector, hybrid, rerank

Retrieval metrics, for questions the documents can answer:
  hit@k   did the passage with the answer show up among the top-k excerpts?
          (a chunk from an expected page, containing the `evidence` text if one is given)
  MRR     mean reciprocal rank of that passage (1.0 = always ranked first)

With --llm, each answer is also checked for:
  cites expected page   the answer cites at least one of the expected pages
  key facts             the answer mentions every `must_mention` term
  no made-up citations  every citation in the answer was actually retrieved
and every question the documents can't answer must get the exact "I don't know" reply.

Exits with status 1 if any check fails, so it can guard changes in CI.
"""

import argparse
import json
import sys
from pathlib import Path

from rag import config

QUESTIONS = Path(__file__).parent / "eval" / "questions.json"


def mark(ok: bool) -> str:
    return "PASS" if ok else "FAIL"


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--llm", action="store_true", help="also generate and check answers (uses the API key)")
    parser.add_argument("-k", type=int, default=config.TOP_K, help=f"excerpts to retrieve (default {config.TOP_K})")
    parser.add_argument("--mode", choices=["vector", "hybrid", "rerank"], default=None, help=f"search mode (default {config.SEARCH_MODE})")
    args = parser.parse_args()
    if args.mode:
        config.SEARCH_MODE = args.mode

    from rag.answer import ask
    from rag.store import search

    data = json.loads(QUESTIONS.read_text(encoding="utf-8"))
    failures = 0

    print(f"Answerable questions ({config.SEARCH_MODE} search, top-{args.k}{', + LLM answer' if args.llm else ''})\n")
    hits_at_k, reciprocal_ranks = 0, []
    for item in data["answerable"]:
        expected = {(src, page) for src, page in item["expected"]}
        results = search(item["question"], args.k)
        evidence = item.get("evidence", "").lower()
        rank = next(
            (i for i, h in enumerate(results, 1) if (h.source, h.page) in expected and evidence in h.text.lower()),
            None,
        )
        hits_at_k += rank is not None
        reciprocal_ranks.append(1 / rank if rank else 0.0)

        checks = {"retrieved": rank is not None}
        if args.llm:
            answer = ask(item["question"], args.k)
            cited = {(c.split(",")[0][1:], int(c.split("p. ")[1][:-1])) for c in answer.citations}
            text = answer.text.lower()
            checks["cites expected page"] = bool(cited & expected)
            checks["key facts"] = all(term.lower() in text for term in item["must_mention"])
            checks["no made-up citations"] = not answer.unsupported_citations

        ok = all(checks.values())
        failures += not ok
        where = f"rank {rank}" if rank else "not in top-k"
        print(f"  {mark(ok)}  {item['question']}")
        print(f"        {where}; " + ", ".join(f"{name}: {'yes' if v else 'NO'}" for name, v in checks.items()))

    n = len(data["answerable"])
    print(f"\n  hit@{args.k}: {hits_at_k}/{n} ({hits_at_k / n:.0%})   MRR: {sum(reciprocal_ranks) / n:.2f}\n")

    print("Questions the documents can't answer\n")
    for question in data["unanswerable"]:
        best = max(h.similarity for h in search(question, args.k))
        below = best < config.MIN_SIMILARITY
        if args.llm:
            refused = ask(question, args.k).text == config.NOT_FOUND
            how = "similarity threshold" if below else "LLM"
            detail = f"best similarity {best:.2f}; refused by {how}" if refused else f"best similarity {best:.2f}; ANSWERED"
        else:
            # Without the LLM we can only see whether the threshold alone catches it;
            # the rest is left to the LLM, so that's not a failure here.
            refused = True
            detail = f"best similarity {best:.2f}; " + (
                "refused by threshold" if below else "passes threshold, the LLM decides (run with --llm)"
            )
        failures += not refused
        print(f"  {mark(refused)}  {question}\n        {detail}")

    print(f"\n{'All checks passed.' if not failures else f'{failures} check(s) failed.'}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
