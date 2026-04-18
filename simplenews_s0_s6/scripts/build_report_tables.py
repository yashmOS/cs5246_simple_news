from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd


METRIC_COLS = [
    "rouge1", "rouge2", "rougel",
    "flesch_reading_ease", "fkgl",
    "entity_coverage", "entity_precision", "entity_f1",
    "number_coverage", "number_precision", "number_f1",
    "date_coverage", "date_precision", "date_f1",
    "entity_unsupported_ratio", "number_unsupported_ratio", "date_unsupported_ratio",
    "copy_token_ratio", "novel_token_ratio", "novel_content_token_ratio",
    "word_count", "sentence_count", "avg_sentence_len",
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Build report tables from combined outputs.")
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--combined", type=Path, default=None)
    parser.add_argument("--split", default="test")
    args = parser.parse_args()

    project_root = args.project_root.resolve()
    combined_path = args.combined.resolve() if args.combined else project_root / "outputs" / f"{args.split}_all_systems_combined.csv"
    out_dir = project_root / "outputs" / "tables"
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(combined_path, low_memory=False)
    available_metrics = [c for c in METRIC_COLS if c in df.columns]

    metric_summary = df.groupby("system", dropna=False)[available_metrics].mean(numeric_only=True).reset_index()
    if {"entity_f1", "number_f1", "date_f1"}.issubset(metric_summary.columns):
        metric_summary["fact_f1_mean"] = metric_summary[["entity_f1", "number_f1", "date_f1"]].mean(axis=1)
    if {"entity_unsupported_ratio", "number_unsupported_ratio", "date_unsupported_ratio"}.issubset(metric_summary.columns):
        metric_summary["unsupported_mean"] = metric_summary[["entity_unsupported_ratio", "number_unsupported_ratio", "date_unsupported_ratio"]].mean(axis=1)
    metric_summary.to_csv(out_dir / f"{args.split}_metric_summary_refactored.csv", index=False)

    doc_counts = df.groupby("system")["doc_id"].nunique().reset_index(name="n_docs")
    doc_counts.to_csv(out_dir / f"{args.split}_doc_counts_refactored.csv", index=False)

    if {"system", "replacements"}.issubset(df.columns):
        s3 = df[df["system"].astype(str).str.upper() == "S3"].copy()
        if not s3.empty:
            repl_count = s3["replacements"].fillna("[]").astype(str).apply(lambda s: 0 if s == "[]" else s.count("'original'") + s.count('"original"'))
            dist = pd.DataFrame({
                "replacement_count": sorted(repl_count.value_counts().index.tolist()),
                "n_docs": [int((repl_count == k).sum()) for k in sorted(repl_count.value_counts().index.tolist())],
            })
            dist.to_csv(out_dir / f"{args.split}_s3_replacement_distribution_refactored.csv", index=False)

    # Ranking-oriented compact table
    ranking_cols = [c for c in ["system", "rougel", "flesch_reading_ease", "fkgl", "fact_f1_mean", "unsupported_mean", "word_count"] if c in metric_summary.columns]
    metric_summary[ranking_cols].to_csv(out_dir / f"{args.split}_main_results_compact.csv", index=False)

    print(f"Wrote tables to: {out_dir}")


if __name__ == "__main__":
    main()
