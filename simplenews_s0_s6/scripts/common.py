from __future__ import annotations

import gc
import sys
from pathlib import Path
from typing import Optional

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd

from src.config import DEFAULT_CONFIG, ExperimentConfig
from src.dataset_bootstrap import ensure_dataset
from src.io_utils import ensure_dirs, load_csv, standardize_dataframe


def add_project_root_to_path(project_root: Path) -> None:
    project_root = project_root.resolve()
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))


def detect_device() -> str:
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
        if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            return "mps"
    except Exception:
        pass
    return "cpu"


def clear_memory() -> None:
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        elif getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            torch.mps.empty_cache()
    except Exception:
        pass


def ensure_project_dirs(project_root: Path) -> None:
    ensure_dirs(
        project_root / "outputs",
        project_root / "outputs" / "system_runs",
        project_root / "outputs" / "tables",
        project_root / "outputs" / "figures",
        project_root / "outputs" / "cache",
    )


def tuned_experiment_config() -> ExperimentConfig:
    cfg = ExperimentConfig()
    cfg.random_seed = 42
    cfg.selector.max_sentences = 3
    cfg.selector.similarity_threshold = 0.08
    cfg.selector.top_entity_k = 8
    cfg.selector.require_number_coverage = True
    cfg.selector.require_date_coverage = True

    cfg.simplifier.enabled = True
    cfg.simplifier.max_word_length = 8
    cfg.simplifier.min_word_frequency_rank = 2000
    cfg.simplifier.max_replacements_per_doc = 10
    cfg.simplifier.min_replacement_confidence = 0.70
    cfg.simplifier.max_glossary_items = 8
    cfg.simplifier.use_glossary_fallback = True
    cfg.simplifier.use_structural_simplification = False
    return cfg


def ensure_local_dataset(project_root: Path, dataset_name: str = "cnn") -> str:
    data_dir = project_root / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    if dataset_name.lower() not in {"cnn", "cnn_dailymail", "cnn/dailymail"}:
        raise ValueError(f"Unsupported dataset for this workflow: {dataset_name}")
    return ensure_dataset(data_dir=data_dir, auto_download=True, force_download=False)


def load_split_df(project_root: Path, split: str, limit: Optional[int] = None) -> pd.DataFrame:
    ensure_local_dataset(project_root, dataset_name="cnn")
    split_path = project_root / "data" / f"{split}.csv"
    raw, colmap = load_csv(split_path)
    df = standardize_dataframe(raw, colmap)
    df["doc_id"] = df["doc_id"].astype(str)
    if limit is not None:
        df = df.iloc[:limit].copy()
    return df


def standardize_external_outputs(df: pd.DataFrame, system: str) -> pd.DataFrame:
    out = df.copy()

    if "highlights" in out.columns and "reference" not in out.columns:
        out = out.rename(columns={"highlights": "reference"})
    if "id" in out.columns and "doc_id" not in out.columns:
        out["doc_id"] = out["id"]
    if "doc_id" not in out.columns:
        out["doc_id"] = range(len(out))
    if "system" not in out.columns:
        out["system"] = system

    out["doc_id"] = out["doc_id"].astype(str)

    keepable = {
        "doc_id",
        "id",
        "system",
        "system_label",
        "article",
        "reference",
        "output",
        "title",
        "replacements",
        "glossary",
        "rouge1",
        "rouge2",
        "rougel",
        "word_count",
        "sentence_count",
        "avg_sentence_len",
        "flesch_reading_ease",
        "fkgl",
        "entity_coverage",
        "entity_precision",
        "entity_f1",
        "entity_unsupported_ratio",
        "number_coverage",
        "number_precision",
        "number_f1",
        "number_unsupported_ratio",
        "date_coverage",
        "date_precision",
        "date_f1",
        "date_unsupported_ratio",
        "novel_token_ratio",
        "copy_token_ratio",
        "novel_content_token_ratio",
        "s4_scaffold",
        "s4_raw",
        "s4_fallback_used",
        "s4_accept",
        "s4_reject_reasons",
        "s4_src_entities_count",
        "s4_src_numbers_count",
        "s4_src_dates_count",
        "s4_raw_word_count",
        "s4_entity_f1",
        "s4_number_f1",
        "s4_date_f1",
        "s4_entity_unsupported_ratio",
        "s4_number_unsupported_ratio",
        "s4_date_unsupported_ratio",
        "s5_mode",
        "s5_num_chunks",
    }

    keep = [c for c in out.columns if c in keepable]
    out = out[keep].copy()
    return out


LEGACY_OUTPUT_COLUMNS = [
    "doc_id",
    "id",
    "system",
    "article",
    "reference",
    "output",
    "system_label",
    "title",
    "replacements",
    "glossary",
    "rouge1",
    "rouge2",
    "rougel",
    "word_count",
    "sentence_count",
    "avg_sentence_len",
    "flesch_reading_ease",
    "fkgl",
    "entity_coverage",
    "number_coverage",
    "date_coverage",
    "entity_precision",
    "entity_f1",
    "entity_unsupported_ratio",
    "number_precision",
    "number_f1",
    "number_unsupported_ratio",
    "date_precision",
    "date_f1",
    "date_unsupported_ratio",
    "novel_token_ratio",
    "copy_token_ratio",
    "novel_content_token_ratio",
    "s4_scaffold",
    "s4_raw",
    "s4_fallback_used",
    "s4_accept",
    "s4_reject_reasons",
    "s4_src_entities_count",
    "s4_src_numbers_count",
    "s4_src_dates_count",
    "s4_raw_word_count",
    "s4_entity_f1",
    "s4_number_f1",
    "s4_date_f1",
    "s4_entity_unsupported_ratio",
    "s4_number_unsupported_ratio",
    "s4_date_unsupported_ratio",
    "s5_mode",
    "s5_num_chunks",
]

S0_TO_S6 = {"S0", "S1", "S2", "S3", "S4", "S5", "S6"}

SYSTEM_LABEL_DEFAULTS = {
    "S0": "S0 Lead-3",
    "S1": "S1 TextRank",
    "S2": "S2 Coverage-aware TextRank",
    "S3": "S3 Coverage-aware + simplification",
    "S4": np.nan,
    "S5": np.nan,
    "S6": "S6 DPO",
}
LISTLIKE_DEFAULT_SYSTEMS = {"S0", "S1", "S2", "S3", "S6"}
S5_MODE_DEFAULTS = {"S5": "single_pass", "S6": "dpo"}
S5_NUM_CHUNKS_DEFAULTS = {"S5": 1}

BLANK_LEGACY_COLS = {
    "s4_scaffold",
    "s4_raw",
    "s4_fallback_used",
    "s4_accept",
    "s4_reject_reasons",
    "s4_src_entities_count",
    "s4_src_numbers_count",
    "s4_src_dates_count",
    "s4_raw_word_count",
    "s4_entity_f1",
    "s4_number_f1",
    "s4_date_f1",
    "s4_entity_unsupported_ratio",
    "s4_number_unsupported_ratio",
    "s4_date_unsupported_ratio",
}


def _normalize_system_name(system: str | None) -> str:
    return str(system or "").strip().upper()


def _resolve_output_system(df: pd.DataFrame, system: str | None = None) -> str:
    explicit = _normalize_system_name(system)
    if explicit in S0_TO_S6:
        return explicit

    if "system" in df.columns:
        values = sorted({str(v).strip().upper() for v in df["system"].dropna().tolist() if str(v).strip()})
        if len(values) == 1:
            return values[0]
        if len(values) > 1:
            raise ValueError(f"Expected a single-system DataFrame, found multiple systems: {values}")

    return "S6"


def _is_blankish_scalar(value) -> bool:
    if value is None:
        return True
    try:
        if pd.isna(value):
            return True
    except Exception:
        pass
    return isinstance(value, str) and value.strip() == ""


def _fill_blankish_with_default(series: pd.Series, default_value):
    if pd.isna(default_value):
        return series
    mask = series.apply(_is_blankish_scalar)
    return series.where(~mask, default_value)


def _default_value_for_missing_legacy_column(col: str, system: str | None = None):
    resolved_system = _normalize_system_name(system)
    if col in BLANK_LEGACY_COLS:
        return np.nan
    if col in {"replacements", "glossary"}:
        return "[]" if resolved_system in LISTLIKE_DEFAULT_SYSTEMS else np.nan
    if col == "title":
        return ""
    if col == "system_label":
        return SYSTEM_LABEL_DEFAULTS.get(resolved_system, np.nan)
    if col == "s5_mode":
        return S5_MODE_DEFAULTS.get(resolved_system, np.nan)
    if col == "s5_num_chunks":
        return S5_NUM_CHUNKS_DEFAULTS.get(resolved_system, np.nan)
    return np.nan


def pad_to_legacy_output_schema(df: pd.DataFrame, system: str | None = None) -> pd.DataFrame:
    out = df.copy()
    resolved_system = _resolve_output_system(out, system)

    if "system" not in out.columns:
        out["system"] = resolved_system
    else:
        out["system"] = out["system"].fillna(resolved_system).astype(str).str.upper()

    if "doc_id" in out.columns:
        out["doc_id"] = out["doc_id"].astype(str)
    if "id" not in out.columns and "doc_id" in out.columns:
        out["id"] = out["doc_id"].astype(str)

    for col in LEGACY_OUTPUT_COLUMNS:
        if col not in out.columns:
            out[col] = _default_value_for_missing_legacy_column(col, resolved_system)

    out = out[LEGACY_OUTPUT_COLUMNS]
    return out


def validate_legacy_output_schema(df: pd.DataFrame) -> None:
    missing = [c for c in LEGACY_OUTPUT_COLUMNS if c not in df.columns]
    extra = [c for c in df.columns if c not in LEGACY_OUTPUT_COLUMNS]
    if missing:
        raise ValueError(f"Final CSV is missing legacy columns: {missing}")
    if len(df.columns) != len(LEGACY_OUTPUT_COLUMNS):
        raise ValueError(
            f"Final CSV has {len(df.columns)} columns, expected {len(LEGACY_OUTPUT_COLUMNS)}. Extra columns: {extra}"
        )


def has_legacy_output_schema(df: pd.DataFrame) -> bool:
    try:
        validate_legacy_output_schema(df)
        return True
    except Exception:
        return False


def finalize_system_output(df: pd.DataFrame, system: str | None = None) -> pd.DataFrame:
    resolved_system = _resolve_output_system(df, system)
    out = standardize_external_outputs(df, system=resolved_system)

    out["system"] = resolved_system
    out["doc_id"] = out["doc_id"].astype(str)

    if "id" not in out.columns:
        out["id"] = out["doc_id"]
    else:
        id_series = out["id"].copy()
        id_series = id_series.where(~id_series.apply(_is_blankish_scalar), out["doc_id"])
        out["id"] = id_series.astype(str)

    for col in ["title", "system_label", "replacements", "glossary", "s5_mode", "s5_num_chunks"]:
        default_value = _default_value_for_missing_legacy_column(col, resolved_system)
        if col not in out.columns:
            out[col] = default_value
        else:
            out[col] = _fill_blankish_with_default(out[col], default_value)

    out = pad_to_legacy_output_schema(out, system=resolved_system)
    validate_legacy_output_schema(out)
    return out
