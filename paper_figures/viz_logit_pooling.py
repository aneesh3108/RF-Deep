'''
Plot logit-baseline pooling variants in the style of existing ablations.
'''

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from project_paths import FIGURES_ROOT, LOGIT_BASELINES_RESULTS_ROOT


plt.rcParams.update({"font.size": 16})

DATASET_GROUPS = {
    "main": ["RSNA PE", "MIDRC C19", "KiTS", "PancreasCT"],
    "external": ["Breast Cancer CT", "MIDRC C19+"],
}
DATASET_COLORS = {
    "RSNA PE": "royalblue",
    "MIDRC C19": "darkorange",
    "KiTS": "forestgreen",
    "PancreasCT": "crimson",
    "Breast Cancer CT": "purple",
    "MIDRC C19+": "teal",
}
POOLINGS = ["Mean", "Max", "95th percentile", "Median"]
POOLING_FILE_MAP = {
    "Max": "max",
    "95th percentile": "p95",
    "Median": "median",
}
JSON_DATASET_MAP = {
    "RSNA PE": "RSNA PE",
    "MIDRC C19": "MIDRC C19",
    "KiTS": "KiTS",
    "PancreasCT": "Pancreas",
}
METRIC_LABELS = {
    "maxsoftmax": "MaxSoftmax",
    "maxlogit": "MaxLogit",
    "energy": "Energy",
}


# Mean pooling values from the original all-voxel tables used in the manuscript.
MEAN_RESULTS = {
    "maxsoftmax": {
        "RSNA PE": {"auroc": (88.61, 84.62, 92.36), "fpr95": (38.37, 31.78, 49.24)},
        "MIDRC C19": {"auroc": (86.57, 81.79, 90.14), "fpr95": (49.27, 39.53, 59.78)},
        "KiTS": {"auroc": (95.67, 93.41, 98.06), "fpr95": (23.26, 11.47, 34.19)},
        "PancreasCT": {"auroc": (94.61, 92.61, 97.49), "fpr95": (34.27, 24.80, 49.30)},
        "Breast Cancer CT": {"auroc": (91.86, 87.47, 95.45), "fpr95": (34.62, 24.98, 50.83)},
        "MIDRC C19+": {"auroc": (89.37, 85.12, 92.71), "fpr95": (36.32, 25.36, 52.55)},
    },
    "maxlogit": {
        "RSNA PE": {"auroc": (88.77, 85.09, 92.43), "fpr95": (40.01, 24.80, 52.52)},
        "MIDRC C19": {"auroc": (89.31, 85.05, 93.06), "fpr95": (41.59, 29.84, 53.96)},
        "KiTS": {"auroc": (95.89, 93.58, 98.44), "fpr95": (18.05, 10.41, 27.37)},
        "PancreasCT": {"auroc": (93.53, 90.17, 96.59), "fpr95": (30.87, 16.17, 56.87)},
        "Breast Cancer CT": {"auroc": (94.27, 91.13, 97.11), "fpr95": (23.15, 15.94, 30.82)},
        "MIDRC C19+": {"auroc": (89.89, 85.66, 93.77), "fpr95": (32.81, 22.08, 52.21)},
    },
    "energy": {
        "RSNA PE": {"auroc": (88.59, 84.85, 92.24), "fpr95": (39.86, 24.80, 52.55)},
        "MIDRC C19": {"auroc": (89.30, 84.98, 93.04), "fpr95": (43.04, 29.84, 55.43)},
        "KiTS": {"auroc": (95.80, 93.41, 98.44), "fpr95": (17.47, 10.41, 26.65)},
        "PancreasCT": {"auroc": (93.52, 90.10, 96.53), "fpr95": (29.96, 15.83, 56.87)},
        "Breast Cancer CT": {"auroc": (94.34, 91.36, 97.16), "fpr95": (22.86, 15.56, 30.82)},
        "MIDRC C19+": {"auroc": (89.54, 85.09, 93.62), "fpr95": (32.62, 22.08, 52.17)},
    },
}


def load_json_results(metric: str, results_root: Path, datasets: list[str]):
    results = {pooling: {} for pooling in POOLING_FILE_MAP}
    for pooling, file_suffix in POOLING_FILE_MAP.items():
        json_path = results_root / "global" / f"global_{metric}_{file_suffix}_all.json"
        with json_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        for dataset in datasets:
            dataset_key = JSON_DATASET_MAP[dataset]
            metrics = payload["ood_metrics_bootstrap"][dataset_key]
            results[pooling][dataset] = {
                "auroc": (
                    100 * metrics["auroc_mean"],
                    100 * metrics["auroc_ci"][0],
                    100 * metrics["auroc_ci"][1],
                ),
                "fpr95": (
                    100 * metrics["fpr95_mean"],
                    100 * metrics["fpr95_ci"][0],
                    100 * metrics["fpr95_ci"][1],
                ),
            }
    return results


def build_metric_results(metric: str, results_root: Path, datasets: list[str]):
    metric_results = {"Mean": MEAN_RESULTS[metric]}
    metric_results.update(load_json_results(metric, results_root, datasets))
    return metric_results


def create_subplot(
    ax,
    metric_key: str,
    results_root: Path,
    datasets: list[str],
    show_ylabel_left: bool,
    show_ylabel_right: bool,
):
    metric_results = build_metric_results(metric_key, results_root, datasets)
    x_values = np.arange(len(POOLINGS))

    for dataset in datasets:
        auroc_values = [metric_results[pooling][dataset]["auroc"] for pooling in POOLINGS]
        means = np.array([v[0] for v in auroc_values])
        lowers = np.array([v[1] for v in auroc_values])
        uppers = np.array([v[2] for v in auroc_values])
        ax.errorbar(
            x_values,
            means,
            yerr=[np.maximum(means - lowers, 0), np.maximum(uppers - means, 0)],
            color=DATASET_COLORS[dataset],
            marker="o",
            linewidth=2.5,
            capsize=4,
            capthick=1.5,
            markersize=7,
            alpha=0.9,
        )

    if show_ylabel_left:
        ax.set_ylabel("AUROC", labelpad=-10)
    ax.set_ylim(75, 100)
    ax.set_xticks(x_values)
    ax.set_xticklabels(["Mean", "Max", "P95", "Median"], rotation=0)
    ax.set_title(METRIC_LABELS[metric_key])
    ax.grid(alpha=0.25, linestyle="--", linewidth=0.8)

    ax2 = ax.twinx()
    for dataset in datasets:
        fpr_values = [metric_results[pooling][dataset]["fpr95"] for pooling in POOLINGS]
        means = np.array([v[0] for v in fpr_values])
        lowers = np.array([v[1] for v in fpr_values])
        uppers = np.array([v[2] for v in fpr_values])
        ax2.errorbar(
            x_values,
            means,
            yerr=[np.maximum(means - lowers, 0), np.maximum(uppers - means, 0)],
            color=DATASET_COLORS[dataset],
            marker="x",
            linestyle="--",
            linewidth=2.5,
            capsize=4,
            capthick=1.5,
            markersize=6,
            alpha=0.9,
        )
    if show_ylabel_right:
        ax2.set_ylabel("FPR95", labelpad=-10)
    ax2.set_ylim(0, 100)
    return ax2


def create_figure(results_root: Path, cohort_set: str):
    datasets = DATASET_GROUPS[cohort_set]
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5), sharey=True)
    metric_order = ["maxsoftmax", "maxlogit", "energy"]
    for idx, (ax, metric_key) in enumerate(zip(axes, metric_order)):
        create_subplot(
            ax,
            metric_key,
            results_root=results_root,
            datasets=datasets,
            show_ylabel_left=(idx == 0),
            show_ylabel_right=(idx == len(metric_order) - 1),
        )

    handles = []
    for dataset in datasets:
        handles.append(
            plt.Line2D([0], [0], color=DATASET_COLORS[dataset], marker="o", linewidth=2.5, label=dataset)
        )
    handles.append(plt.Line2D([0], [0], color="black", marker="o", linewidth=2.0, label="AUROC"))
    handles.append(plt.Line2D([0], [0], color="black", marker="x", linestyle="--", linewidth=2.0, label="FPR95"))
    fig.legend(handles=handles, loc="upper center", ncol=max(4, len(datasets) + 2), frameon=False, bbox_to_anchor=(0.5, 1.10))
    fig.tight_layout()
    plt.subplots_adjust(top=0.77, wspace=0.18)
    return fig


def main():
    parser = argparse.ArgumentParser(description="Render pooling-variant figure for logit baselines.")
    parser.add_argument(
        "--results-root",
        default=str(LOGIT_BASELINES_RESULTS_ROOT),
        help="Root directory containing logit-baseline JSON outputs.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output PDF path.",
    )
    parser.add_argument(
        "--cohort-set",
        choices=["main", "external"],
        default="main",
        help="Which dataset group to render.",
    )
    args = parser.parse_args()

    fig = create_figure(Path(args.results_root), args.cohort_set)
    default_output = FIGURES_ROOT / f"logit_pooling_variants_{args.cohort_set}.pdf"
    output_path = Path(args.output) if args.output else default_output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote figure to {output_path}")


if __name__ == "__main__":
    main()
