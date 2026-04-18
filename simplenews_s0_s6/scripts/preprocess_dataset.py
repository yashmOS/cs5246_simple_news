from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from scripts.common import ensure_project_dirs, load_split_df
from src.preprocess import article_features, token_frequency_rank


def main() -> None:
    parser = argparse.ArgumentParser(description="Preprocess and standardize one dataset split.")
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--split", default="test", choices=["train", "validation", "test"])
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    project_root = args.project_root.resolve()
    ensure_project_dirs(project_root)
    df = load_split_df(project_root, args.split, limit=args.limit)

    freq_rank = token_frequency_rank(df["article"].fillna("").astype(str).tolist())
    rows = []
    for row in df.itertuples(index=False):
        feats = article_features(str(getattr(row, "article", "") or ""), title=str(getattr(row, "title", "") or ""))
        rows.append({
            "doc_id": str(getattr(row, "doc_id", "")),
            "n_sentences": len(feats["sentences"]),
            "n_entities": len(feats["entities"]),
            "n_numbers": len(feats["numbers"]),
            "n_dates": len(feats["dates"]),
        })

    out_dir = project_root / "outputs" / "cache"
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir / f"{args.split}_preprocess_summary.csv"
    pd.DataFrame(rows).to_csv(summary_path, index=False)

    vocab_path = out_dir / f"{args.split}_token_frequency_rank.csv"
    pd.DataFrame({"token": list(freq_rank.keys()), "rank": list(freq_rank.values())}).to_csv(vocab_path, index=False)

    print(f"Saved preprocess summary to: {summary_path}")
    print(f"Saved token ranks to: {vocab_path}")


if __name__ == "__main__":
    main()
