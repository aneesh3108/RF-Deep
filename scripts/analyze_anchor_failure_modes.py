"""
Summarize anchor failure modes from merged RF-Deep + anchor CSV output.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from ood_utils import compute_95ci_percentile, compute_metrics
from project_paths import ANALYSIS_RESULTS_ROOT


STANDARD_OOD_DATASETS = ["rsna", "covid19", "kits23", "pancreas"]
EXTERNAL_OOD_DATASETS = ["breastc", "covid19a"]

def summarize_metric(values: list[float]) -> dict[str, float]:
    if not values:
        return {
            "mean": np.nan,
            "std": np.nan,
            "ci_lower": np.nan,
            "ci_upper": np.nan,
        }
    mean, std, ci_lower, ci_upper = compute_95ci_percentile(values)
    return {
        "mean": float(mean),
        "std": float(std),
        "ci_lower": float(ci_lower),
        "ci_upper": float(ci_upper),
    }


def require_unique_scan_rows(df: pd.DataFrame) -> pd.DataFrame:
    key_cols = ["method", "run_idx", "seed", "eval_dataset", "train_context", "lodo_holdout_dataset", "dataset", "filename_base"]
    deduped = df.drop_duplicates(subset=key_cols)
    return deduped.reset_index(drop=True)


def add_component_bin(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    def bin_components(value: float) -> str:
        if value == 0:
            return "0"
        if value == 1:
            return "1"
        if 2 <= value <= 5:
            return "2-5"
        return ">5"

    out["component_bin"] = out["component_count_kept"].map(bin_components)
    return out


def add_tiny_flag(df: pd.DataFrame, tiny_threshold_cc: float) -> pd.DataFrame:
    out = df.copy()
    out["tiny_prediction"] = (
        (out["foreground_volume_cc_kept"] > 0) &
        (out["foreground_volume_cc_kept"] <= tiny_threshold_cc)
    )
    return out


def volume_quartile_summary(
    df: pd.DataFrame,
    datasets: list[str],
) -> pd.DataFrame:
    rows = []
    for eval_dataset in datasets:
        ds_df = df[df["eval_dataset"] == eval_dataset].copy()
        if ds_df.empty:
            continue

        for (run_idx, seed, train_context, lodo_holdout_dataset), run_df in ds_df.groupby(
            ["run_idx", "seed", "train_context", "lodo_holdout_dataset"],
            dropna=False,
            sort=False,
        ):
            ood_df = run_df[
                (run_df["dataset"] == eval_dataset) &
                (run_df["foreground_volume_cc_kept"] > 0)
            ].copy()
            id_df = run_df[run_df["dataset"] == "lrad"].copy()
            id_positive = id_df[id_df["foreground_volume_cc_kept"] > 0].copy()
            if len(ood_df) < 4 or id_df.empty or len(id_positive) < 4:
                continue

            quartile_labels = ["Q1", "Q2", "Q3", "Q4"]
            q25, q50, q75 = id_positive["foreground_volume_cc_kept"].quantile(
                [0.25, 0.50, 0.75]
            ).tolist()
            bins = [0.0, float(q25), float(q50), float(q75), np.inf]
            try:
                ood_df["volume_quartile"] = pd.cut(
                    ood_df["foreground_volume_cc_kept"],
                    bins=bins,
                    labels=quartile_labels,
                    include_lowest=True,
                    right=True,
                )
            except ValueError:
                continue
            if ood_df["volume_quartile"].isna().all():
                continue

            for quartile, quart_df in ood_df.groupby("volume_quartile", observed=False):
                if quart_df.empty:
                    continue
                subset = pd.concat([id_df, quart_df], ignore_index=True)
                if subset["label"].nunique() < 2:
                    continue
                auroc, _ = compute_metrics(subset["label"].values, subset["probability"].values)
                rows.append(
                    {
                        "eval_dataset": eval_dataset,
                        "volume_quartile": str(quartile),
                        "run_idx": run_idx,
                        "seed": seed,
                        "train_context": train_context,
                        "lodo_holdout_dataset": lodo_holdout_dataset,
                        "auroc": auroc,
                        "ood_count": int(len(quart_df)),
                        "id_q25_cc": float(q25),
                        "id_q50_cc": float(q50),
                        "id_q75_cc": float(q75),
                    }
                )

    if not rows:
        return pd.DataFrame()

    raw = pd.DataFrame(rows)
    summary_rows = []
    for (eval_dataset, quartile), group in raw.groupby(["eval_dataset", "volume_quartile"], sort=False):
        stats = summarize_metric(group["auroc"].tolist())
        summary_rows.append(
            {
                "eval_dataset": eval_dataset,
                "volume_quartile": quartile,
                "auroc_mean": stats["mean"],
                "auroc_std": stats["std"],
                "auroc_ci_lower": stats["ci_lower"],
                "auroc_ci_upper": stats["ci_upper"],
                "ood_count_mean": float(group["ood_count"].mean()),
                "num_runs": int(len(group)),
            }
        )
    return pd.DataFrame(summary_rows)


def component_distribution_summary(
    df: pd.DataFrame,
    datasets: list[str],
) -> pd.DataFrame:
    rows = []
    dedup = df.drop_duplicates(subset=["dataset", "filename_base"])
    bins = ["0", "1", "2-5", ">5"]
    for dataset in datasets:
        ds_df = dedup[dedup["dataset"] == dataset].copy()
        if ds_df.empty:
            continue
        total = len(ds_df)
        counts = ds_df["component_bin"].value_counts().to_dict()
        rows.append(
            {
                "dataset": dataset,
                "n_scans": total,
                **{
                    f"count_{bin_label}": int(counts.get(bin_label, 0))
                    for bin_label in bins
                },
                **{
                    f"pct_{bin_label}": float(100.0 * counts.get(bin_label, 0) / total)
                    for bin_label in bins
                },
                "no_prediction_count": int(ds_df["no_prediction_kept"].sum()),
                "no_prediction_pct": float(100.0 * ds_df["no_prediction_kept"].mean()),
                "tiny_prediction_count": int(ds_df["tiny_prediction"].sum()),
                "tiny_prediction_pct": float(100.0 * ds_df["tiny_prediction"].mean()),
            }
        )
    return pd.DataFrame(rows)


def component_auroc_summary(
    df: pd.DataFrame,
    datasets: list[str],
) -> pd.DataFrame:
    rows = []
    for eval_dataset in datasets:
        ds_df = df[df["eval_dataset"] == eval_dataset].copy()
        if ds_df.empty:
            continue
        for (run_idx, seed, train_context, lodo_holdout_dataset), run_df in ds_df.groupby(
            ["run_idx", "seed", "train_context", "lodo_holdout_dataset"], dropna=False, sort=False
        ):
            ood_df = run_df[(run_df["dataset"] == eval_dataset) & (run_df["component_count_kept"] >= 1)].copy()
            id_df = run_df[(run_df["dataset"] == "lrad") & (run_df["component_count_kept"] >= 1)].copy()
            if ood_df.empty or id_df.empty:
                continue
            subset = pd.concat([id_df, ood_df], ignore_index=True)
            if subset["label"].nunique() < 2:
                continue
            auroc, _ = compute_metrics(subset["label"].values, subset["probability"].values)
            rows.append(
                {
                    "eval_dataset": eval_dataset,
                    "run_idx": run_idx,
                    "seed": seed,
                    "train_context": train_context,
                    "lodo_holdout_dataset": lodo_holdout_dataset,
                    "auroc": auroc,
                    "ood_count_ge1": int(len(ood_df)),
                    "id_count_ge1": int(len(id_df)),
                }
            )

    if not rows:
        return pd.DataFrame()

    raw = pd.DataFrame(rows)
    summary_rows = []
    for eval_dataset, group in raw.groupby("eval_dataset", sort=False):
        stats = summarize_metric(group["auroc"].tolist())
        summary_rows.append(
            {
                "eval_dataset": eval_dataset,
                "auroc_mean_ge1": stats["mean"],
                "auroc_std_ge1": stats["std"],
                "auroc_ci_lower_ge1": stats["ci_lower"],
                "auroc_ci_upper_ge1": stats["ci_upper"],
                "ood_count_ge1_mean": float(group["ood_count_ge1"].mean()),
                "id_count_ge1_mean": float(group["id_count_ge1"].mean()),
                "num_runs": int(len(group)),
            }
        )
    return pd.DataFrame(summary_rows)


def no_prediction_fallback_summary(
    df: pd.DataFrame,
    datasets: list[str],
    threshold: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    for eval_dataset in datasets:
        ds_df = df[df["eval_dataset"] == eval_dataset].copy()
        if ds_df.empty:
            continue
        for (run_idx, seed, train_context, lodo_holdout_dataset), run_df in ds_df.groupby(
            ["run_idx", "seed", "train_context", "lodo_holdout_dataset"], dropna=False, sort=False
        ):
            subset = run_df[run_df["dataset"].isin(["lrad", eval_dataset])].copy()
            if subset.empty or subset["label"].nunique() < 2:
                continue

            fallback_score = np.where(
                subset["no_prediction_kept"],
                1.0,
                subset["probability"],
            )
            auroc, fpr95 = compute_metrics(subset["label"].values, fallback_score)
            id_subset = subset[subset["dataset"] == "lrad"].copy()
            id_false_pos = int(((id_subset["no_prediction_kept"]) | (id_subset["probability"] >= threshold)).sum())
            rows.append(
                {
                    "eval_dataset": eval_dataset,
                    "run_idx": run_idx,
                    "seed": seed,
                    "train_context": train_context,
                    "lodo_holdout_dataset": lodo_holdout_dataset,
                    "auroc": auroc,
                    "fpr95": fpr95,
                    "id_false_positives": id_false_pos,
                    "id_total": int(len(id_subset)),
                }
            )

    if rows:
        raw = pd.DataFrame(rows)
        summary_rows = []
        for eval_dataset, group in raw.groupby("eval_dataset", sort=False):
            auroc_stats = summarize_metric(group["auroc"].tolist())
            fpr95_stats = summarize_metric(group["fpr95"].tolist())
            fp_stats = summarize_metric(group["id_false_positives"].tolist())
            summary_rows.append(
                {
                    "eval_dataset": eval_dataset,
                    "fallback_auroc_mean": auroc_stats["mean"],
                    "fallback_auroc_ci_lower": auroc_stats["ci_lower"],
                    "fallback_auroc_ci_upper": auroc_stats["ci_upper"],
                    "fallback_fpr95_mean": fpr95_stats["mean"],
                    "fallback_fpr95_ci_lower": fpr95_stats["ci_lower"],
                    "fallback_fpr95_ci_upper": fpr95_stats["ci_upper"],
                    "id_false_positives_mean": fp_stats["mean"],
                    "id_false_positives_ci_lower": fp_stats["ci_lower"],
                    "id_false_positives_ci_upper": fp_stats["ci_upper"],
                    "id_total_mean": float(group["id_total"].mean()),
                    "num_runs": int(len(group)),
                }
            )
        return pd.DataFrame(summary_rows), raw

    return pd.DataFrame(), pd.DataFrame()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze anchor failure modes from merged RF-Deep scan scores."
    )
    parser.add_argument(
        "--merged-csv",
        default=str(ANALYSIS_RESULTS_ROOT / "anchor_summary" / "smit_anchor_scores_merged.csv"),
        help="Merged scan-level RF-Deep + anchor CSV.",
    )
    parser.add_argument(
        "--method",
        default="ensemble",
        choices=["unified", "dataset_specific", "ensemble", "lodo"],
        help="Method to analyze.",
    )
    parser.add_argument(
        "--include-external",
        action="store_true",
        help="Include breastc and covid19a in the output tables.",
    )
    parser.add_argument(
        "--tiny-threshold-cc",
        type=float,
        default=1.0,
        help="Absolute threshold for the 'tiny prediction' indicator.",
    )
    parser.add_argument(
        "--fallback-threshold",
        type=float,
        default=0.5,
        help="RF-Deep probability threshold used in the no-prediction fallback analysis.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(ANALYSIS_RESULTS_ROOT / "anchor_failure"),
        help="Directory for output CSV tables.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.merged_csv)
    df = df[df["method"] == args.method].copy()
    df = require_unique_scan_rows(df)
    df = add_component_bin(df)
    df = add_tiny_flag(df, args.tiny_threshold_cc)

    datasets = list(STANDARD_OOD_DATASETS)
    if args.include_external:
        datasets.extend(EXTERNAL_OOD_DATASETS)

    volume_df = volume_quartile_summary(df, datasets)
    component_dist_df = component_distribution_summary(df, ["lrad", *datasets])
    component_auroc_df = component_auroc_summary(df, datasets)
    fallback_summary_df, fallback_raw_df = no_prediction_fallback_summary(
        df, datasets, args.fallback_threshold
    )

    stem = f"{args.method}_tiny{args.tiny_threshold_cc:g}cc_thr{args.fallback_threshold:g}"
    volume_df.to_csv(output_dir / f"{stem}_volume_quartiles.csv", index=False)
    component_dist_df.to_csv(output_dir / f"{stem}_component_distribution.csv", index=False)
    component_auroc_df.to_csv(output_dir / f"{stem}_component_auroc.csv", index=False)
    fallback_summary_df.to_csv(output_dir / f"{stem}_fallback_summary.csv", index=False)
    fallback_raw_df.to_csv(output_dir / f"{stem}_fallback_raw.csv", index=False)

    print(f"Saved anchor failure tables to: {output_dir}")


if __name__ == "__main__":
    main()
