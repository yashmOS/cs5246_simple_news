from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize one system output file.")
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--system", required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--input", type=Path, default=None)
    args = parser.parse_args()

    project_root = args.project_root.resolve()
    in_path = args.input.resolve() if args.input else project_root / "outputs" / "system_runs" / f"{args.split}_{args.system}.csv"
    df = pd.read_csv(in_path)

    metric_cols = [
        c for c in [
            "rouge1", "rouge2", "rougel",
            "flesch_reading_ease", "fkgl",
            "entity_f1", "number_f1", "date_f1",
            "entity_unsupported_ratio", "number_unsupported_ratio", "date_unsupported_ratio",
            "word_count", "sentence_count", "avg_sentence_len",
        ]
        if c in df.columns
    ]
    summary = df[metric_cols].mean(numeric_only=True).to_frame(name=args.system).T
    if {"entity_f1", "number_f1", "date_f1"}.issubset(summary.columns):
        summary["fact_f1_mean"] = summary[["entity_f1", "number_f1", "date_f1"]].mean(axis=1)
    if {"entity_unsupported_ratio", "number_unsupported_ratio", "date_unsupported_ratio"}.issubset(summary.columns):
        summary["unsupported_mean"] = summary[["entity_unsupported_ratio", "number_unsupported_ratio", "date_unsupported_ratio"]].mean(axis=1)

    out_dir = project_root / "outputs" / "tables"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{args.split}_{args.system}_metric_summary.csv"
    summary.to_csv(out_path, index=True)
    print(summary)
    print(f"Saved summary to: {out_path}")


if __name__ == "__main__":
    main()
