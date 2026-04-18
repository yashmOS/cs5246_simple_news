
from __future__ import annotations

import argparse
import ast
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DEFAULT_SYSTEM_ORDER = ["S0", "S1", "S2", "S3", "S4", "S5", "S6"]


def _ordered_systems(values: list[str]) -> list[str]:
    present = [str(v) for v in values if pd.notna(v)]
    seen = set()
    deduped = []
    for v in present:
        if v not in seen:
            deduped.append(v)
            seen.add(v)
    known = [s for s in DEFAULT_SYSTEM_ORDER if s in deduped]
    extras = sorted([s for s in deduped if s not in DEFAULT_SYSTEM_ORDER])
    return known + extras


def _parse_listlike(value) -> list:
    if isinstance(value, list):
        return value
    if value is None:
        return []
    try:
        if pd.isna(value):
            return []
    except Exception:
        pass
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return []
        try:
            parsed = ast.literal_eval(text)
            if isinstance(parsed, list):
                return parsed
        except Exception:
            return []
    return []


def _bootstrap_mean_ci(values, n_boot: int = 1000, alpha: float = 0.05, seed: int = 42) -> tuple[float, float, float]:
    arr = np.asarray(pd.Series(values).dropna(), dtype=float)
    if len(arr) == 0:
        return np.nan, np.nan, np.nan
    if len(arr) == 1:
        val = float(arr[0])
        return val, val, val
    rng = np.random.default_rng(seed)
    n = len(arr)
    means = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        means[i] = rng.choice(arr, size=n, replace=True).mean()
    lower = float(np.quantile(means, alpha / 2))
    upper = float(np.quantile(means, 1 - alpha / 2))
    return float(arr.mean()), lower, upper


def _find_first_existing(candidates: list[Path]) -> Path | None:
    for path in candidates:
        if path.exists():
            return path
    return None


def _find_first_glob(directory: Path, patterns: list[str]) -> Path | None:
    for pattern in patterns:
        matches = sorted(directory.glob(pattern))
        if matches:
            return matches[0]
    return None


def _load_summary(project_root: Path, split: str, combined_path: Path | None) -> tuple[pd.DataFrame, pd.DataFrame]:
    table_dir = project_root / "outputs" / "tables"
    default_summary_path = table_dir / f"{split}_metric_summary_refactored.csv"
    resolved_combined = combined_path if combined_path is not None else project_root / "outputs" / f"{split}_all_systems_combined.csv"

    combined = pd.read_csv(resolved_combined, low_memory=False)

    metric_cols = [c for c in [
        "rougel",
        "flesch_reading_ease",
        "fkgl",
        "entity_f1",
        "number_f1",
        "date_f1",
        "entity_unsupported_ratio",
        "number_unsupported_ratio",
        "date_unsupported_ratio",
        "word_count",
    ] if c in combined.columns]
    summary = combined.groupby("system", dropna=False)[metric_cols].mean(numeric_only=True).reset_index()

    if "fact_f1_mean" not in summary.columns and {"entity_f1", "number_f1", "date_f1"}.issubset(summary.columns):
        summary["fact_f1_mean"] = summary[["entity_f1", "number_f1", "date_f1"]].mean(axis=1)

    if "unsupported_mean" not in summary.columns and {"entity_unsupported_ratio", "number_unsupported_ratio", "date_unsupported_ratio"}.issubset(summary.columns):
        summary["unsupported_mean"] = summary[["entity_unsupported_ratio", "number_unsupported_ratio", "date_unsupported_ratio"]].mean(axis=1)

    summary = summary.copy()
    combined = combined.copy()
    summary["system"] = summary["system"].astype(str)
    combined["system"] = combined["system"].astype(str)

    system_order = _ordered_systems(summary["system"].tolist() or combined["system"].tolist())
    summary = summary.set_index("system").reindex(system_order).reset_index()
    combined = combined.set_index("system").reset_index()
    return summary, combined


def _save_basic_rougel_bar(summary: pd.DataFrame, out_dir: Path, split: str) -> str | None:
    plot_df = summary[["system", "rougel"]].dropna().copy()
    if plot_df.empty:
        return None
    x = np.arange(len(plot_df))
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(x, plot_df["rougel"].to_numpy())
    ax.set_xticks(x)
    ax.set_xticklabels(plot_df["system"].tolist())
    ax.set_title("ROUGE-L by system")
    ax.set_xlabel("System")
    ax.set_ylabel("ROUGE-L")
    fig.tight_layout()
    out_name = f"{split}_rougel_bar_refactored.png"
    fig.savefig(out_dir / out_name, dpi=180)
    plt.close(fig)
    return out_name


def _save_basic_readability_scatter(summary: pd.DataFrame, out_dir: Path, split: str) -> str | None:
    needed = {"system", "flesch_reading_ease", "fact_f1_mean", "word_count"}
    if not needed.issubset(summary.columns):
        return None
    plot_df = summary[list(needed)].dropna(subset=["flesch_reading_ease", "fact_f1_mean"]).copy()
    if plot_df.empty:
        return None
    fig, ax = plt.subplots(figsize=(6.5, 5))
    ax.scatter(
        plot_df["flesch_reading_ease"],
        plot_df["fact_f1_mean"],
        s=plot_df["word_count"].fillna(0) * 3 + 30,
    )
    for _, row in plot_df.iterrows():
        ax.annotate(str(row["system"]), (row["flesch_reading_ease"], row["fact_f1_mean"]))
    ax.set_xlabel("Flesch Reading Ease")
    ax.set_ylabel("Mean fact F1")
    ax.set_title("Readability vs faithfulness")
    fig.tight_layout()
    out_name = f"{split}_readability_vs_faithfulness_refactored.png"
    fig.savefig(out_dir / out_name, dpi=180)
    plt.close(fig)
    return out_name


def _save_output_length(summary: pd.DataFrame, out_dir: Path, split: str) -> str | None:
    plot_df = summary[["system", "word_count"]].dropna().copy() if {"system", "word_count"}.issubset(summary.columns) else pd.DataFrame()
    if plot_df.empty:
        return None
    x = np.arange(len(plot_df))
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(x, plot_df["word_count"].to_numpy())
    ax.set_xticks(x)
    ax.set_xticklabels(plot_df["system"].tolist())
    ax.set_title("Average output length by system")
    ax.set_xlabel("System")
    ax.set_ylabel("Average words")
    fig.tight_layout()
    out_name = f"{split}_output_length_refactored.png"
    fig.savefig(out_dir / out_name, dpi=180)
    plt.close(fig)
    return out_name


def _save_unsupported_mean(summary: pd.DataFrame, out_dir: Path, split: str) -> str | None:
    if not {"system", "unsupported_mean"}.issubset(summary.columns):
        return None
    plot_df = summary[["system", "unsupported_mean"]].dropna().copy()
    if plot_df.empty:
        return None
    x = np.arange(len(plot_df))
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(x, plot_df["unsupported_mean"].to_numpy())
    ax.set_xticks(x)
    ax.set_xticklabels(plot_df["system"].tolist())
    ax.set_title("Unsupported-detail mean by system")
    ax.set_xlabel("System")
    ax.set_ylabel("Mean unsupported ratio")
    fig.tight_layout()
    out_name = f"{split}_unsupported_mean_refactored.png"
    fig.savefig(out_dir / out_name, dpi=180)
    plt.close(fig)
    return out_name


def _save_rougel_bootstrap_ci(combined: pd.DataFrame, table_dir: Path, out_dir: Path, split: str) -> tuple[str | None, str | None]:
    if not {"system", "rougel"}.issubset(combined.columns):
        return None, None
    rows = []
    order = _ordered_systems(combined["system"].tolist())
    for system in order:
        sub = combined.loc[combined["system"] == system, "rougel"]
        if sub.dropna().empty:
            continue
        mean_, lo, hi = _bootstrap_mean_ci(sub)
        rows.append({
            "system": system,
            "rougel_mean": mean_,
            "ci_low": lo,
            "ci_high": hi,
            "err_low": mean_ - lo,
            "err_high": hi - mean_,
            "n_docs": int(sub.notna().sum()),
        })
    ci_df = pd.DataFrame(rows)
    if ci_df.empty:
        return None, None

    csv_name = f"{split}_rougel_bootstrap_ci_refactored.csv"
    ci_df.to_csv(table_dir / csv_name, index=False)

    x = np.arange(len(ci_df))
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    ax.bar(
        x,
        ci_df["rougel_mean"].to_numpy(),
        yerr=np.vstack([ci_df["err_low"].to_numpy(), ci_df["err_high"].to_numpy()]),
        capsize=4,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(ci_df["system"].tolist())
    ax.set_title("ROUGE-L by system with 95% bootstrap confidence intervals")
    ax.set_xlabel("System")
    ax.set_ylabel("ROUGE-L")
    for xpos, mean_ in zip(x, ci_df["rougel_mean"].to_numpy()):
        ax.text(xpos, mean_, f"{mean_:.3f}", ha="center", va="bottom", fontsize=9)
    fig.tight_layout()

    fig_name = f"{split}_rougel_bootstrap_ci_refactored.png"
    fig.savefig(out_dir / fig_name, dpi=220)
    plt.close(fig)
    return fig_name, csv_name


def _save_readability_faithfulness_bubble(summary: pd.DataFrame, out_dir: Path, split: str) -> str | None:
    needed = {"system", "flesch_reading_ease", "fact_f1_mean", "word_count"}
    if not needed.issubset(summary.columns):
        return None
    plot_df = summary[list(needed)].dropna().copy()
    if plot_df.empty:
        return None

    size_base = plot_df["word_count"].astype(float)
    if np.isclose(size_base.max(), size_base.min()):
        bubble_size = np.full(len(size_base), 180.0)
    else:
        bubble_size = 60 + 4 * (size_base - size_base.min())

    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.scatter(
        plot_df["flesch_reading_ease"],
        plot_df["fact_f1_mean"],
        s=bubble_size,
        alpha=0.8,
    )

    for _, row in plot_df.iterrows():
        ax.annotate(
            str(row["system"]),
            (row["flesch_reading_ease"], row["fact_f1_mean"]),
            xytext=(5, 5),
            textcoords="offset points",
        )

    if plot_df["flesch_reading_ease"].notna().any():
        ax.axvline(float(plot_df["flesch_reading_ease"].median()), linestyle="--", linewidth=1)
    if plot_df["fact_f1_mean"].notna().any():
        ax.axhline(float(plot_df["fact_f1_mean"].median()), linestyle="--", linewidth=1)

    ax.set_title("Readability vs faithfulness (bubble size = average output length)")
    ax.set_xlabel("Flesch reading ease (higher = easier)")
    ax.set_ylabel("Mean fact F1 (entity/number/date)")

    legend_vals = sorted(set([
        float(plot_df["word_count"].min()),
        float(plot_df["word_count"].median()),
        float(plot_df["word_count"].max()),
    ]))
    legend_handles = []
    for val in legend_vals:
        if np.isclose(size_base.max(), size_base.min()):
            s = 180.0
        else:
            s = 60 + 4 * (val - float(size_base.min()))
        legend_handles.append(ax.scatter([], [], s=s, label=f"{val:.0f} words"))
    if legend_handles:
        ax.legend(handles=legend_handles, title="Avg output length", loc="best")

    fig.tight_layout()
    fig_name = f"{split}_readability_faithfulness_bubble_refactored.png"
    fig.savefig(out_dir / fig_name, dpi=220)
    plt.close(fig)
    return fig_name


def _save_s4_guard_behaviour(combined: pd.DataFrame, table_dir: Path, out_dir: Path, split: str) -> tuple[str | None, str | None]:
    s4 = combined.loc[combined["system"] == "S4"].copy() if "system" in combined.columns else pd.DataFrame()
    if s4.empty or "s4_fallback_used" not in s4.columns:
        return None, None

    plot_df = pd.DataFrame({
        "metric": ["fallback_true_rate", "fallback_false_rate"],
        "value": [
            float((s4["s4_fallback_used"] == True).mean()),
            float((s4["s4_fallback_used"] == False).mean()),
        ],
        "label": ["Fallback", "Accepted"],
    })

    csv_name = f"{split}_s4_guard_behaviour_refactored.csv"
    plot_df.to_csv(table_dir / csv_name, index=False)

    x = np.arange(len(plot_df))
    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.bar(x, plot_df["value"].to_numpy())
    ax.set_xticks(x)
    ax.set_xticklabels(plot_df["label"].tolist())
    ax.set_title("S4 guard behaviour")
    ax.set_xlabel("Outcome")
    ax.set_ylabel("Rate")
    fig.tight_layout()

    fig_name = f"{split}_s4_guard_behaviour_refactored.png"
    fig.savefig(out_dir / fig_name, dpi=200)
    plt.close(fig)
    return fig_name, csv_name


def _save_s3_effect_summary(combined: pd.DataFrame, table_dir: Path, out_dir: Path, split: str) -> tuple[str | None, str | None]:
    needed = {"system", "doc_id", "output"}
    if not needed.issubset(combined.columns):
        return None, None

    s2 = combined.loc[combined["system"] == "S2", ["doc_id", "output"]].rename(columns={"output": "s2_output"}).copy()
    s3_cols = ["doc_id", "output"]
    for optional_col in ["replacements", "glossary"]:
        if optional_col in combined.columns:
            s3_cols.append(optional_col)
    s3 = combined.loc[combined["system"] == "S3", s3_cols].rename(columns={"output": "s3_output"}).copy()

    if s2.empty or s3.empty or "replacements" not in s3.columns or "glossary" not in s3.columns:
        return None, None

    s3["replacements_list"] = s3["replacements"].apply(_parse_listlike)
    s3["glossary_list"] = s3["glossary"].apply(_parse_listlike)

    merged = s2.merge(s3, on="doc_id", how="inner")
    if merged.empty:
        return None, None

    identical_rate = float((merged["s2_output"] == merged["s3_output"]).mean())
    docs_with_replacement_rate = float((merged["replacements_list"].apply(len) > 0).mean())
    glossary_usage_rate = float((merged["glossary_list"].apply(len) > 0).mean())
    avg_replacements_per_doc = float(merged["replacements_list"].apply(len).mean())

    effect_df = pd.DataFrame({
        "metric": ["S2=S3 identical rate", "Docs with >=1 replacement", "Docs with glossary"],
        "value": [identical_rate, docs_with_replacement_rate, glossary_usage_rate],
    })
    csv_name = f"{split}_s3_effect_summary_refactored.csv"
    effect_df.to_csv(table_dir / csv_name, index=False)

    x = np.arange(len(effect_df))
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.bar(x, effect_df["value"].to_numpy())
    ax.set_xticks(x)
    ax.set_xticklabels(effect_df["metric"].tolist(), rotation=15)
    ax.set_title(f"S3 effect summary (avg replacements/doc = {avg_replacements_per_doc:.2f})")
    ax.set_ylabel("Rate")
    fig.tight_layout()

    fig_name = f"{split}_s3_effect_summary_refactored.png"
    fig.savefig(out_dir / fig_name, dpi=200)
    plt.close(fig)
    return fig_name, csv_name


def _save_s4_selector_sweep_heatmap(project_root: Path, table_dir: Path, out_dir: Path, split: str) -> tuple[str | None, str | None]:
    direct_candidates = [
        table_dir / f"{split}_s4_selector_sweep_results_refactored.csv",
        table_dir / f"{split}_s4_selector_sweep_results.csv",
    ]
    sweep_path = _find_first_existing(direct_candidates)
    if sweep_path is None:
        sweep_path = _find_first_glob(table_dir, ["*s4_selector_sweep*.csv"])

    if sweep_path is None:
        return None, None

    sweep_df = pd.read_csv(sweep_path)
    required = {"max_sentences", "top_entity_k", "objective"}
    if not required.issubset(sweep_df.columns):
        return None, None

    heat_df = (
        sweep_df.groupby(["max_sentences", "top_entity_k"])["objective"]
        .max()
        .reset_index()
        .pivot(index="max_sentences", columns="top_entity_k", values="objective")
        .sort_index()
    )
    if heat_df.empty:
        return None, None

    fig, ax = plt.subplots(figsize=(6, 4.5))
    im = ax.imshow(heat_df.values, aspect="auto")
    ax.set_title("S4 selector sweep (best objective)")
    ax.set_xlabel("top_entity_k")
    ax.set_ylabel("max_sentences")
    ax.set_xticks(range(len(heat_df.columns)))
    ax.set_xticklabels([str(c) for c in heat_df.columns])
    ax.set_yticks(range(len(heat_df.index)))
    ax.set_yticklabels([str(i) for i in heat_df.index])

    for i in range(len(heat_df.index)):
        for j in range(len(heat_df.columns)):
            val = heat_df.values[i, j]
            if pd.notna(val):
                ax.text(j, i, f"{val:.3f}", ha="center", va="center")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()

    fig_name = f"{split}_s4_selector_sweep_heatmap_refactored.png"
    fig.savefig(out_dir / fig_name, dpi=200)
    plt.close(fig)

    top10_name = f"{split}_s4_selector_top10_refactored.csv"
    sweep_df.head(10).to_csv(table_dir / top10_name, index=False)
    return fig_name, top10_name


def main() -> None:
    parser = argparse.ArgumentParser(description="Build report plots from refactored SimpleNews outputs.")
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--combined", type=Path, default=None)
    parser.add_argument("--split", default="test")
    args = parser.parse_args()

    project_root = args.project_root.resolve()
    out_dir = project_root / "outputs" / "figures"
    table_dir = project_root / "outputs" / "tables"
    out_dir.mkdir(parents=True, exist_ok=True)
    table_dir.mkdir(parents=True, exist_ok=True)

    combined_path = args.combined.resolve() if args.combined else None
    summary, combined = _load_summary(project_root, args.split, combined_path)

    generated_figures: list[str] = []
    generated_tables: list[str] = []

    for maybe_name in [
        _save_basic_rougel_bar(summary, out_dir, args.split),
        _save_basic_readability_scatter(summary, out_dir, args.split),
        _save_output_length(summary, out_dir, args.split),
        _save_unsupported_mean(summary, out_dir, args.split),
        _save_readability_faithfulness_bubble(summary, out_dir, args.split),
    ]:
        if maybe_name:
            generated_figures.append(maybe_name)

    fig_name, csv_name = _save_rougel_bootstrap_ci(combined, table_dir, out_dir, args.split)
    if fig_name:
        generated_figures.append(fig_name)
    if csv_name:
        generated_tables.append(csv_name)

    fig_name, csv_name = _save_s4_guard_behaviour(combined, table_dir, out_dir, args.split)
    if fig_name:
        generated_figures.append(fig_name)
    if csv_name:
        generated_tables.append(csv_name)

    fig_name, csv_name = _save_s3_effect_summary(combined, table_dir, out_dir, args.split)
    if fig_name:
        generated_figures.append(fig_name)
    if csv_name:
        generated_tables.append(csv_name)

    fig_name, csv_name = _save_s4_selector_sweep_heatmap(project_root, table_dir, out_dir, args.split)
    if fig_name:
        generated_figures.append(fig_name)
    if csv_name:
        generated_tables.append(csv_name)

    manifest = pd.DataFrame({
        "figure": sorted(set([p.name for p in out_dir.glob(f"{args.split}_*_refactored.png")])),
    })
    manifest_path = out_dir / f"{args.split}_figure_manifest.csv"
    manifest.to_csv(manifest_path, index=False)

    print(f"Wrote plots to: {out_dir}")
    print(f"Wrote manifest to: {manifest_path}")
    if generated_tables:
        print("Wrote supporting tables:")
        for name in generated_tables:
            print(f" - {table_dir / name}")
    missing_optional = []
    if not (out_dir / f"{args.split}_s4_guard_behaviour_refactored.png").exists():
        missing_optional.append("S4 guard behaviour (needs s4_fallback_used in combined outputs)")
    if not (out_dir / f"{args.split}_s3_effect_summary_refactored.png").exists():
        missing_optional.append("S3 effect summary (needs S2/S3 rows plus replacements and glossary columns)")
    if not (out_dir / f"{args.split}_s4_selector_sweep_heatmap_refactored.png").exists():
        missing_optional.append("S4 selector sweep heatmap (needs a selector-sweep CSV in outputs/tables)")
    if missing_optional:
        print("Optional figures skipped:")
        for msg in missing_optional:
            print(f" - {msg}")


if __name__ == "__main__":
    main()
