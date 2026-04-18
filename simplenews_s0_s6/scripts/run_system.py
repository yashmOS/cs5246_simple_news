from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from scripts.common import (
    detect_device,
    ensure_project_dirs,
    finalize_system_output,
    has_legacy_output_schema,
    load_split_df,
    standardize_external_outputs,
    tuned_experiment_config,
)
from scripts.seq2seq_helpers import run_s4_dataframe, run_s5_dataframe, run_s6_dataframe
from src.pipeline import generate_single_system_output


SHORT_TO_LONG = {
    "S0": "s0_lead3",
    "S1": "s1_textrank",
    "S2": "s2_coverage",
    "S3": "s3_simplified",
}
ALL_SYSTEMS = {"S0", "S1", "S2", "S3", "S4", "S5", "S6"}
MIN_EVALUATED_COLUMNS = {"article", "reference", "output", "rouge1", "rougel", "word_count"}


def _default_output(project_root: Path, split: str, system: str) -> Path:
    return project_root / "outputs" / "system_runs" / f"{split}_{system}.csv"


def _cached_output_path(project_root: Path, split: str, system: str) -> Path:
    return project_root / "outputs" / "system_runs" / f"{split}_{system}.csv"


def _frozen_s6_source(project_root: Path) -> Path | None:
    candidates = sorted(project_root.glob("s6_dpo_inference_results.csv")) + sorted(project_root.glob("*s6*dpo*inference*.csv"))
    return candidates[0] if candidates else None


def _resolve_s4_model(project_root: Path, checkpoint_path: str | None) -> str | None:
    if checkpoint_path:
        return checkpoint_path
    default_ckpt = project_root / "outputs" / "checkpoints" / "s4_finetuned"
    if default_ckpt.exists():
        return str(default_ckpt)
    return None


def _resolve_s5_model(project_root: Path, checkpoint_path: str | None) -> tuple[str | None, str]:
    family = "facebook/bart-large-cnn"
    if checkpoint_path:
        return checkpoint_path, family
    default_ckpt = project_root / "outputs" / "checkpoints" / "s5_finetuned"
    if default_ckpt.exists():
        return str(default_ckpt), family
    return None, family


def _resolve_s6_checkpoint(project_root: Path, checkpoint_path: str | None) -> str | None:
    if checkpoint_path:
        return checkpoint_path

    preferred_dpo = project_root / "dpo_checkpoints" / "dpo_checkpoint_epoch3.pt"
    if preferred_dpo.exists():
        return str(preferred_dpo)

    dpo_dir = project_root / "dpo_checkpoints"
    if dpo_dir.exists():
        epoch_ckpts = list(dpo_dir.glob("dpo_checkpoint_epoch*.pt"))
        if epoch_ckpts:
            def _epoch_num(p: Path) -> int:
                try:
                    return int(p.stem.split("epoch")[-1])
                except Exception:
                    return -1

            epoch_ckpts = sorted(epoch_ckpts, key=_epoch_num, reverse=True)
            return str(epoch_ckpts[0])

    candidates = [
        project_root / "outputs" / "checkpoints" / "s6_dpo" / "model.safetensors",
        project_root / "checkpoints" / "model.safetensors",
    ]
    for cand in candidates:
        if cand.exists():
            return str(cand)

    return None


def _write_finalized_output(df: pd.DataFrame, out_path: Path, system: str) -> int:
    finalized = finalize_system_output(df, system=system)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    finalized.to_csv(out_path, index=False)
    return len(finalized)


def _evaluate_and_finalize_s6_df(df: pd.DataFrame) -> pd.DataFrame:
    from src.evaluation import evaluate_output
    from src.preprocess import normalize_whitespace

    out = standardize_external_outputs(df, system="S6")

    if not {"article", "reference", "output"}.issubset(out.columns):
        raise ValueError("S6 raw CSV must contain article, reference, and output columns.")

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


def _evaluate_raw_s6(project_root: Path, out_path: Path) -> bool:
    raw = _frozen_s6_source(project_root)
    if raw is None:
        return False

    df = pd.read_csv(raw, low_memory=False)
    finalized = _evaluate_and_finalize_s6_df(df)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    finalized.to_csv(out_path, index=False)
    return True


def _use_existing_cached_output(cached_out: Path, out_path: Path, system: str) -> bool:
    if not cached_out.exists():
        return False

    df = pd.read_csv(cached_out, low_memory=False)

    if system == "S6":
        if has_legacy_output_schema(df):
            finalized = finalize_system_output(df, system=system)
        elif {"article", "reference", "output"}.issubset(df.columns):
            finalized = _evaluate_and_finalize_s6_df(df)
        else:
            raise ValueError(
                f"Cached {system} output exists but does not look usable: {cached_out}"
            )
    else:
        if not MIN_EVALUATED_COLUMNS.issubset(df.columns):
            raise ValueError(
                f"Cached {system} output appears incomplete or raw. Expected at least {sorted(MIN_EVALUATED_COLUMNS)} in {cached_out}."
            )
        finalized = finalize_system_output(df, system=system)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    finalized.to_csv(out_path, index=False)
    print(f"Using existing cached output (normalized): {cached_out}")
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one SimpleNews system independently.")
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--system", required=True, choices=sorted(ALL_SYSTEMS))
    parser.add_argument("--split", default="test", choices=["train", "validation", "test"])
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--mode",
        choices=["auto", "cached", "inference"],
        default="auto",
        help="For S4/S5/S6: auto uses cached if available, otherwise inference if checkpoint exists.",
    )
    parser.add_argument("--checkpoint-path", default=None)
    parser.add_argument(
        "--model-name",
        default=None,
        help="Optional model family name for S5/S6 inference; S4 usually uses the checkpoint path.",
    )
    parser.add_argument("--force-recompute", action="store_true")
    args = parser.parse_args()

    project_root = args.project_root.resolve()
    ensure_project_dirs(project_root)
    out_path = args.output.resolve() if args.output else _default_output(project_root, args.split, args.system)

    print(f"Device: {detect_device()}")
    print(f"System: {args.system}")
    print(f"Split: {args.split}")
    print(f"Mode: {args.mode}")
    print(f"Output: {out_path}")

    cached_out = _cached_output_path(project_root, args.split, args.system)

    if args.system in {"S4", "S5", "S6"} and args.mode in {"auto", "cached"} and cached_out.exists() and not args.force_recompute:
        if _use_existing_cached_output(cached_out, out_path, args.system):
            return

    if args.system in SHORT_TO_LONG:
        cfg = tuned_experiment_config()
        df = load_split_df(project_root, args.split, limit=args.limit)
        out = generate_single_system_output(df=df, config=cfg, system=SHORT_TO_LONG[args.system]).copy()
        out["system"] = args.system
        n_rows = _write_finalized_output(out, out_path, args.system)
        print(f"Saved {n_rows} rows to: {out_path}")
        return

    df = load_split_df(project_root, args.split, limit=args.limit)

    if args.system == "S4":
        model_path = _resolve_s4_model(project_root, args.checkpoint_path)
        if args.mode == "cached" or (args.mode == "auto" and model_path is None):
            if _use_existing_cached_output(cached_out, out_path, "S4") and not args.force_recompute:
                return
            raise FileNotFoundError("No cached S4 output found and no S4 checkpoint directory available.")
        if model_path is None:
            raise FileNotFoundError("S4 inference requested but no checkpoint path was provided/found.")
        out = run_s4_dataframe(df=df, model_name_or_path=model_path)
        n_rows = _write_finalized_output(out, out_path, "S4")
        print(f"Saved {n_rows} rows to: {out_path}")
        return

    if args.system == "S5":
        model_path, family_name = _resolve_s5_model(project_root, args.checkpoint_path)
        family_name = args.model_name or family_name
        if args.mode == "cached" or (args.mode == "auto" and model_path is None):
            if _use_existing_cached_output(cached_out, out_path, "S5") and not args.force_recompute:
                return
            raise FileNotFoundError("No cached S5 output found and no S5 checkpoint directory available.")
        if model_path is None:
            raise FileNotFoundError("S5 inference requested but no checkpoint path was provided/found.")
        out = run_s5_dataframe(df=df, model_name_or_path=model_path, family_model_name=family_name)
        n_rows = _write_finalized_output(out, out_path, "S5")
        print(f"Saved {n_rows} rows to: {out_path}")
        return

    if args.system == "S6":
        if args.mode == "cached":
            if _use_existing_cached_output(cached_out, out_path, "S6"):
                return
            if _evaluate_raw_s6(project_root, out_path):
                print(f"Saved evaluated cached S6 outputs to: {out_path}")
                return
            raise FileNotFoundError("No cached or raw S6 CSV found.")

        if args.mode == "auto":
            if not args.force_recompute and _use_existing_cached_output(cached_out, out_path, "S6"):
                return

            if _evaluate_raw_s6(project_root, out_path):
                print(f"Saved evaluated cached S6 outputs to: {out_path}")
                return

            checkpoint = _resolve_s6_checkpoint(project_root, args.checkpoint_path)
            if checkpoint is None:
                raise FileNotFoundError(
                    "No S6 cached outputs, raw S6 CSV, or S6 checkpoint found."
                )

        checkpoint = _resolve_s6_checkpoint(project_root, args.checkpoint_path)
        if checkpoint is None:
            raise FileNotFoundError("S6 inference requested but no checkpoint path was provided/found.")

        model_name = args.model_name or "facebook/bart-large-cnn"
        out = run_s6_dataframe(df=df, model_name=model_name, checkpoint_path=checkpoint)
        n_rows = _write_finalized_output(out, out_path, "S6")
        print(f"Saved {n_rows} rows to: {out_path}")
        return

    raise ValueError(f"Unsupported system: {args.system}")


if __name__ == "__main__":
    main()
