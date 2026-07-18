"""Unit tests for the deterministic eval metrics — no API calls."""

from types import SimpleNamespace

from langchain_core.documents import Document

from eval.metrics import generation_row, normalize, quote_in_chunk, retrieval_metrics


def _doc(source: str, text: str) -> Document:
    return Document(page_content=text, metadata={"source": source, "page": 1})


def _cit(marker: int, quote: str) -> SimpleNamespace:
    return SimpleNamespace(marker=marker, quote=quote)


def test_normalize_collapses_whitespace_and_case():
    assert normalize("  Olympus\n  MONS \t is") == "olympus mons is"


def test_quote_in_chunk_tolerates_formatting_differences():
    chunk = "Mars hosts Olympus Mons —\nthe largest volcano in the Solar System."
    assert quote_in_chunk("olympus  mons — the largest volcano", chunk)


def test_quote_in_chunk_rejects_absent_text():
    assert not quote_in_chunk("Mount Everest", "Mars hosts Olympus Mons.")


def test_retrieval_metrics_hit_rate_and_mrr():
    rows = [
        {"id": "a", "expected_source": "x.md", "sources": ["y.md", "x.md", "z.md"]},
        {"id": "b", "expected_source": "z.md", "sources": ["y.md", "x.md"]},
    ]
    df, summary = retrieval_metrics(rows, ks=(1, 2))
    assert summary["hit_rate@1"] == 0.0
    assert summary["hit_rate@2"] == 0.5
    assert summary["mrr"] == 0.25  # (1/2 + 0) / 2
    assert df.loc[df["id"] == "a", "rank"].item() == 2


GOLD_Q = {
    "id": "q1",
    "question": "Where is the largest volcano?",
    "expected_source": "solar_system.md",
    "expected_answer_contains": ["Olympus Mons"],
    "should_refuse": False,
}
GOLD_TRAP = {
    "id": "t1",
    "question": "What is the capital of Australia?",
    "expected_source": None,
    "expected_answer_contains": [],
    "should_refuse": True,
}
DOCS = [
    _doc("solar_system.md", "Mars hosts Olympus Mons — the largest volcano known."),
    _doc("coffee_brewing.md", "A fine grind is used for espresso."),
]


def test_generation_row_answerable_all_checks_pass():
    row = generation_row(
        GOLD_Q,
        answer="The largest volcano is Olympus Mons on Mars [1].",
        citations=[_cit(1, "Olympus Mons — the largest volcano")],
        docs=DOCS,
    )
    assert row["answer_contains"] is True
    assert row["citations_valid"] is True
    assert row["cited_expected_source"] is True
    assert row["refused_correctly"] is True


def test_generation_row_detects_hallucinated_quote_and_wrong_source():
    row = generation_row(
        GOLD_Q,
        answer="It is Mauna Loa [2].",
        citations=[_cit(2, "Mauna Loa is the largest volcano")],
        docs=DOCS,
    )
    assert row["answer_contains"] is False
    assert row["citations_valid"] is False  # quote not present in chunk [2]
    assert row["cited_expected_source"] is False


def test_generation_row_trap_refusal_is_correct():
    row = generation_row(GOLD_TRAP, answer="The context does not say.", citations=[], docs=DOCS)
    assert row["refused_correctly"] is True
    assert row["answer_contains"] is None


def test_generation_row_trap_with_citations_fails():
    row = generation_row(
        GOLD_TRAP,
        answer="Canberra [1].",
        citations=[_cit(1, "Mars hosts Olympus Mons")],
        docs=DOCS,
    )
    assert row["refused_correctly"] is False
