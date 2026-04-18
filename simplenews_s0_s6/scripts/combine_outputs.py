from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from scripts.common import LEGACY_OUTPUT_COLUMNS, finalize_system_output, validate_legacy_output_schema

SYSTEMS = ["S0", "S1", "S2", "S3", "S4", "S5", "S6"]
MIN_EVALUATED_COLUMNS = {"article", "reference", "output", "rouge1", "rougel", "word_count"}


def main() -> None:
    parser = argparse.ArgumentParser(description="Combine per-system outputs into one shared file.")
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--split", default="test")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    project_root = args.project_root.resolve()
    system_dir = project_root / "outputs" / "system_runs"
    frames = []

    for system in SYSTEMS:
        path = system_dir / f"{args.split}_{system}.csv"
        if not path.exists():
            print(f"Skipping missing {system}: {path}")
            continue

        df = pd.read_csv(path, low_memory=False)
        if not MIN_EVALUATED_COLUMNS.issubset(df.columns):
            raise ValueError(
                f"{path} does not look like an evaluated system output. "
                f"Expected at least columns: {sorted(MIN_EVALUATED_COLUMNS)}"
            )

        normalized = finalize_system_output(df, system=system)
        validate_legacy_output_schema(normalized)
        frames.append(normalized)

    if not frames:
        raise FileNotFoundError("No per-system outputs found to combine.")

    combined = pd.concat(frames, ignore_index=True, sort=False)
    combined = combined[LEGACY_OUTPUT_COLUMNS]
    out_path = args.output.resolve() if args.output else project_root / "outputs" / f"{args.split}_all_systems_combined.csv"
    combined.to_csv(out_path, index=False)
    print(f"Saved combined file to: {out_path} ({len(combined)} rows)")


if __name__ == "__main__":
    main()
