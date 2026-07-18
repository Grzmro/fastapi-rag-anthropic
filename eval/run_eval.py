"""Evaluation loop over the gold set (eval/gold.jsonl).

Stages:
  retrieval   hit rate / MRR of the vector search — no API key needed (default)
  generation  deterministic checks of grounded answers (needs ANTHROPIC_API_KEY)
  all         both

Add --judge to also score faithfulness with an LLM judge.

Usage:
    python -m eval.run_eval
    python -m eval.run_eval --stage all --judge

The index is built fresh from data/samples/*.md in a temporary directory —
the production ./chroma_data index is never touched.
"""

import argparse
import json
import os
import sys
import tempfile
from datetime import datetime
from pathlib import Path

import pandas as pd

EVAL_DIR = Path(__file__).parent
SAMPLES_DIR = EVAL_DIR.parent / "data" / "samples"
RESULTS_DIR = EVAL_DIR / "results"
RETRIEVAL_K = 20
KS = (1, 2, 4, 10, 20)


def load_gold() -> list[dict]:
    lines = (EVAL_DIR / "gold.jsonl").read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines if line.strip()]


def setup_index(tmp_dir: str) -> None:
    """Point Chroma at a temp dir (fresh_store pattern from tests/conftest.py)
    and index all sample documents in-process."""
    os.environ["CHROMA_PERSIST_DIR"] = tmp_dir

    from app import config, vectorstore
    from app.ingestion import ingest_file

    config.get_settings.cache_clear()
    vectorstore.get_vectorstore.cache_clear()

    total = 0
    for path in sorted(SAMPLES_DIR.glob("*.md")):
        total += vectorstore.add_documents(ingest_file(path))
    print(f"Indexed {total} chunks from {len(list(SAMPLES_DIR.glob('*.md')))} sample documents\n")


def run_retrieval(gold: list[dict]) -> pd.DataFrame:
    from app import vectorstore
    from eval.metrics import retrieval_metrics

    rows = []
    for item in gold:
        if not item["expected_source"]:  # traps have no retrieval ground truth
            continue
        docs = vectorstore.search(item["question"], k=RETRIEVAL_K)
        rows.append(
            {
                "id": item["id"],
                "expected_source": item["expected_source"],
                "sources": [doc.metadata.get("source") for doc in docs],
            }
        )

    df, summary = retrieval_metrics(rows, ks=KS)
    hits = "  ".join(f"hit@{k} {summary[f'hit_rate@{k}']:.2f}" for k in KS)
    print(f"RETRIEVAL  {hits}  MRR {summary['mrr']:.2f}")
    return df


def run_generation(gold: list[dict], top_k: int | None, judge_model: str | None) -> pd.DataFrame:
    from app.config import get_settings
    from app.rag_chain import answer_question, format_context
    from eval.metrics import generation_row

    if not get_settings().anthropic_api_key:
        sys.exit("The generation stage needs ANTHROPIC_API_KEY — set it in .env")

    rows = []
    for item in gold:
        result, docs = answer_question(item["question"], top_k=top_k)
        row = generation_row(item, result.answer, result.citations, docs)

        if judge_model:
            from eval.judge import faithfulness, judge_answer

            verdict = judge_answer(
                item["question"], format_context(docs), result.answer, model=judge_model
            )
            row["faithfulness"] = faithfulness(verdict)
            row["judge_saw_refusal"] = verdict.is_refusal

        rows.append(row)
        ok = (
            row["refused_correctly"]
            and row["citations_valid"]
            and row["answer_contains"] in (True, None)
        )
        print(f"  {item['id']:>3}  {'ok' if ok else '<-- CHECK'}  {row['answer'][:80]}")

    df = pd.DataFrame(rows)

    answerable = df[~df["should_refuse"]]
    traps = df[df["should_refuse"]]
    print(
        f"\nGENERATION answer_contains {int(answerable['answer_contains'].sum())}/{len(answerable)}"
        f"  cited_expected_source {int(answerable['cited_expected_source'].sum())}/{len(answerable)}"
        f"  citations_valid {int(df['citations_valid'].sum())}/{len(df)}"
        f"  refusals {int(traps['refused_correctly'].sum())}/{len(traps)}"
    )
    if judge_model:
        print(
            f"JUDGE      faithfulness {df['faithfulness'].mean():.2f}"
            f"  refusal_phrasing {int(traps['judge_saw_refusal'].sum())}/{len(traps)}"
            f"  (judge: {judge_model})"
        )
    return df


def main() -> None:
    from eval.judge import DEFAULT_JUDGE_MODEL

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=["retrieval", "generation", "all"], default="retrieval")
    parser.add_argument("--judge", action="store_true", help="score faithfulness with an LLM judge")
    parser.add_argument("--judge-model", default=DEFAULT_JUDGE_MODEL)
    parser.add_argument("--k", type=int, default=None, help="top_k override for generation")
    args = parser.parse_args()

    gold = load_gold()
    frames: dict[str, pd.DataFrame] = {}

    # ignore_cleanup_errors: chromadb may keep its sqlite file locked on Windows.
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp_dir:
        setup_index(tmp_dir)
        if args.stage in ("retrieval", "all"):
            frames["retrieval"] = run_retrieval(gold)
        if args.stage in ("generation", "all"):
            frames["generation"] = run_generation(
                gold, top_k=args.k, judge_model=args.judge_model if args.judge else None
            )

    RESULTS_DIR.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    for name, df in frames.items():
        out = RESULTS_DIR / f"{timestamp}_{name}.csv"
        df.to_csv(out, index=False)
        print(f"wrote {out}")


if __name__ == "__main__":
    main()
