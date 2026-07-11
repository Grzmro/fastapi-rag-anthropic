"""Optional: pull a HuggingFace dataset into text files for RAG testing.

Curated docs in data/samples/ are better for *verifying citations* (you know
the facts). This script is for *volume* — stress-testing retrieval on many
documents. It uses SQuAD, whose paragraph "contexts" make decent documents
and which ships real questions you can try against them.

Requires the `datasets` package (not in requirements.txt — install on demand):

    pip install datasets

Usage:

    python data/load_hf_dataset.py --n 40 --out data/hf
    # then ingest:
    #   for f in data/hf/*.txt; do curl -s -F "file=@$f" http://localhost:8000/ingest; echo; done
"""

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="squad", help="HuggingFace dataset id")
    parser.add_argument("--split", default="validation", help="dataset split")
    parser.add_argument("--n", type=int, default=40, help="number of unique documents to write")
    parser.add_argument("--out", default="data/hf", help="output directory")
    args = parser.parse_args()

    try:
        from datasets import load_dataset
    except ImportError:
        raise SystemExit("This script needs the 'datasets' package. Run: pip install datasets")

    ds = load_dataset(args.dataset, split=args.split)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    seen_contexts: set[str] = set()
    sample_questions: list[str] = []
    written = 0

    for row in ds:
        context = row.get("context", "").strip()
        if not context or context in seen_contexts:
            continue
        seen_contexts.add(context)

        title = row.get("title", "doc").replace("/", "_")
        doc_path = out_dir / f"{written:03d}_{title}.txt"
        doc_path.write_text(context, encoding="utf-8")

        question = row.get("question", "").strip()
        if question:
            sample_questions.append(f"  [{doc_path.name}] {question}")

        written += 1
        if written >= args.n:
            break

    print(f"Wrote {written} documents to {out_dir}/")
    print("\nSome real questions to try against them:")
    print("\n".join(sample_questions[:10]))


if __name__ == "__main__":
    main()
