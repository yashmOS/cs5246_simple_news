from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from scripts.common import ensure_project_dirs, finalize_system_output, standardize_external_outputs
from src.evaluation import evaluate_output
from src.preprocess import normalize_whitespace


FROZEN_S0_S5_PATTERNS = [
    "test_s0_s5_v46_stable_top_to_bottom_all_system_outputs*.csv",
    "*all_system_outputs*.csv",
]
FROZEN_S6_PATTERNS = [
    "s6_dpo_inference_results.csv",
    "*s6*dpo*inference*.csv",
]


def _find_first(project_root: Path, patterns: list[str]) -> Path | None:
    for pattern in patterns:
        matches = sorted(project_root.glob(pattern))
        if matches:
            return matches[0]
    return None


def _write_system_csv(df: pd.DataFrame, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)


def _evaluate_and_normalize_s6(df: pd.DataFrame) -> pd.DataFrame:
    out = standardize_external_outputs(df, system="S6")

    if not {"article", "reference", "output"}.issubset(out.columns):
        raise ValueError("S6 CSV must contain article, reference, and output columns.")

    metric_rows = []
    for row in out.itertuples(index=False):
        article = str(getattr(row, "article", "") or "")
        reference = str(getattr(row, "reference", "") or "")
        output = normalize_whitespace(str(getattr(row, "output", "") or ""))
        metric_rows.append(evaluate_output(source=article, reference=reference, pred=output))

    metric_df = pd.DataFrame(metric_rows)
    for col in metric_df.columns:
        out[col] = metric_df[col]

    return finalize_system_output(out, system="S6")


def main() -> None:
    parser = argparse.ArgumentParser(description="Split frozen S0-S5 outputs and add evaluated S6 into outputs/system_runs.")
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()

    project_root = args.project_root.resolve()
    ensure_project_dirs(project_root)
    system_dir = project_root / "outputs" / "system_runs"

    frozen = _find_first(project_root, FROZEN_S0_S5_PATTERNS)
    if frozen is not None:
        print(f"Using frozen S0-S5 file: {frozen}")
        full = pd.read_csv(frozen, low_memory=False)
        if "system" not in full.columns:
            raise ValueError("Frozen S0-S5 file does not contain a system column.")
        for system, part in full.groupby(full["system"].astype(str).str.upper(), dropna=False):
            normalized = finalize_system_output(part.copy(), system=system)
            out_path = system_dir / f"test_{system}.csv"
            _write_system_csv(normalized, out_path)
            print(f"Wrote {system}: {out_path}")
    else:
        print("No frozen S0-S5 combined CSV found. Skipping S0-S5 bootstrap.")

    s6_path = _find_first(project_root, FROZEN_S6_PATTERNS)
    if s6_path is not None:
        print(f"Using frozen S6 file: {s6_path}")
        s6 = pd.read_csv(s6_path, low_memory=False)
        s6 = _evaluate_and_normalize_s6(s6)
        out_path = system_dir / "test_S6.csv"
        _write_system_csv(s6, out_path)
        print(f"Wrote S6: {out_path}")
    else:
        print("No frozen S6 CSV found. Skipping S6 bootstrap.")

    print("Bootstrap complete.")


if __name__ == "__main__":
    main()
