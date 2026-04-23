"""
Summarize segmentation metric CSVs and compute paired statistical tests.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from project_paths import EXCEL_RECORDS_ROOT


def compute_wilcoxon_results(df, prefix="dice"):
    comparisons = [(0, 1), (0, 2), (1, 2)]
    results = {}
    for i, j in comparisons:
        x = df[f"{prefix}_{i}"]
        y = df[f"{prefix}_{j}"]
        mask = x.notna() & y.notna()
        if mask.sum() >= 10:
            stat, p = wilcoxon(x[mask], y[mask])
            results[f"Model {i} vs Model {j}"] = {"statistic": stat, "p-value": p}
        else:
            results[f"Model {i} vs Model {j}"] = {"statistic": None, "p-value": None}
    return pd.DataFrame(results).T


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize per-model segmentation metrics.")
    parser.add_argument("--data-name", default="test_LRAD")
    parser.add_argument("--threshold", default=0.5, type=float)
    args = parser.parse_args()

    final_paths = [
        EXCEL_RECORDS_ROOT / f"lung_smit_{args.data_name}_{args.threshold}.csv",
        EXCEL_RECORDS_ROOT / f"lung_mim_{args.data_name}_{args.threshold}.csv",
        EXCEL_RECORDS_ROOT / f"lung_ibot_{args.data_name}_{args.threshold}.csv",
        EXCEL_RECORDS_ROOT / f"lung_smitmini_{args.data_name}_{args.threshold}.csv",
        EXCEL_RECORDS_ROOT / f"lung_swinunetr_{args.data_name}_{args.threshold}.csv",
        EXCEL_RECORDS_ROOT / f"lung_swinunetr_10k_{args.data_name}_{args.threshold}.csv",
    ]

    final_dfs = [pd.read_csv(path) for path in final_paths]
    for i, df in enumerate(final_dfs):
        final_dfs[i] = df[["name", "dice", "hd95", "precision", "recall"]].rename(
            columns={
                "dice": f"dice_{i}",
                "hd95": f"hd95_{i}",
                "precision": f"precision_{i}",
                "recall": f"recall_{i}",
            }
        )

    merged_final = final_dfs[0]
    for df in final_dfs[1:]:
        merged_final = pd.merge(merged_final, df, on="name", how="outer")

    for idx, row in merged_final.iterrows():
        dice_values = [row[f"dice_{i}"] for i in range(len(final_paths))]
        if any(pd.notna(val) for val in dice_values):
            for i in range(3):
                if pd.isna(row[f"dice_{i}"]):
                    merged_final.at[idx, f"dice_{i}"] = 0
                if pd.isna(row[f"hd95_{i}"]):
                    merged_final.at[idx, f"hd95_{i}"] = 10

    dice_stats = {f"Model {i}": (merged_final[f"dice_{i}"].mean(), merged_final[f"dice_{i}"].std()) for i in range(len(final_paths))}
    hd95_stats = {f"Model {i}": (merged_final[f"hd95_{i}"].mean(), merged_final[f"hd95_{i}"].std()) for i in range(len(final_paths))}
    precision_means = {f"Model {i}": merged_final[f"precision_{i}"].mean() for i in range(len(final_paths))}
    recall_means = {f"Model {i}": merged_final[f"recall_{i}"].mean() for i in range(len(final_paths))}

    f1_scores = {}
    for i in range(len(final_paths)):
        precision = merged_final[f"precision_{i}"]
        recall = merged_final[f"recall_{i}"]
        f1 = 2 * precision * recall / (precision + recall + 1e-8)
        f1 = f1.replace([np.inf, -np.inf], np.nan)
        f1_scores[f"Model {i}"] = f1.mean()

    print("=== Performance Summary ===\n")
    for i in range(len(final_paths)):
        print(f"Model {i}:")
        print(f"  Dice     : {dice_stats[f'Model {i}'][0]:.3f} ± {dice_stats[f'Model {i}'][1]:.3f}")
        print(f"  HD95     : {hd95_stats[f'Model {i}'][0]:.2f} ± {hd95_stats[f'Model {i}'][1]:.2f}")
        print(f"  Precision: {precision_means[f'Model {i}']:.3f}")
        print(f"  Recall   : {recall_means[f'Model {i}']:.3f}")
        print(f"  F1-score : {f1_scores[f'Model {i}']:.3f}\n")


if __name__ == "__main__":
    main()
