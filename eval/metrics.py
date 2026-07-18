"""Deterministic evaluation metrics — pure functions, no API calls.

Everything here operates on plain data (dicts, strings, Documents and
citation-like objects with `.marker` / `.quote`), so it is unit-testable
without the app singletons or an Anthropic key.
"""

from __future__ import annotations

import pandas as pd


def normalize(text: str) -> str:
    """Lowercase and collapse whitespace, for tolerant exact matching."""
    return " ".join(text.lower().split())


def quote_in_chunk(quote: str, chunk_text: str) -> bool:
    """Is the (normalized) quote literally present in the chunk text?"""
    return normalize(quote) in normalize(chunk_text)


def retrieval_metrics(
    rows: list[dict], ks: tuple[int, ...] = (1, 2, 4, 10, 20)
) -> tuple[pd.DataFrame, dict[str, float]]:
    """Hit rate @k and MRR over ranked search results.

    `rows` items: {"id", "expected_source", "sources": [ranked source names]}.
    Returns a per-question DataFrame and an aggregate summary dict.
    """
    records = []
    for row in rows:
        sources = row["sources"]
        expected = row["expected_source"]
        rank = sources.index(expected) + 1 if expected in sources else None
        record = {"id": row["id"], "rank": rank, "rr": 1.0 / rank if rank else 0.0}
        for k in ks:
            record[f"hit@{k}"] = bool(rank and rank <= k)
        records.append(record)

    df = pd.DataFrame(records)
    summary = {f"hit_rate@{k}": float(df[f"hit@{k}"].mean()) for k in ks}
    summary["mrr"] = float(df["rr"].mean())
    return df, summary


def generation_row(gold: dict, answer: str, citations: list, docs: list) -> dict:
    """Deterministic checks for one generated answer.

    `citations` items need `.marker` (1-based index into `docs`) and `.quote`;
    `docs` are the retrieved LangChain Documents (marker [1] == docs[0]).
    """
    should_refuse = gold["should_refuse"]
    refused = len(citations) == 0

    row = {
        "id": gold["id"],
        "question": gold["question"],
        "should_refuse": should_refuse,
        "refused": refused,
        # Traps must come back with no citations; regular questions with some.
        "refused_correctly": refused == should_refuse,
        # Vacuously true when there are no citations to check.
        "citations_valid": all(
            quote_in_chunk(c.quote, docs[c.marker - 1].page_content) for c in citations
        ),
        "answer": answer,
    }

    if should_refuse:
        row["answer_contains"] = None
        row["cited_expected_source"] = None
    else:
        answer_lower = answer.lower()
        row["answer_contains"] = all(
            term.lower() in answer_lower for term in gold["expected_answer_contains"]
        )
        row["cited_expected_source"] = any(
            docs[c.marker - 1].metadata.get("source") == gold["expected_source"]
            for c in citations
        )
    return row
