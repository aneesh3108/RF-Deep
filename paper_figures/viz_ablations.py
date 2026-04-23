'''
Ablation figure generation for paper plots.
'''

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from project_paths import FIGURES_ROOT

plt.rcParams.update({"font.size": 16})

DATASETS = ["RSNA PE", "MIDRC C19$^-$", "KiTS", "PancreasCT"]
COLORS = ["royalblue", "darkorange", "forestgreen", "crimson"]

ABLATIONS = {
    "nrois": {
        "filename": "ablation_nrois2.pdf",
        "xlabel": "Number of ROIs",
        "x_labels": ["1", "2", "4", "8"],
        "auroc": [
            [(87.70, 85.10, 93.40), (93.20, 87.90, 98.00), (95.80, 92.60, 98.50), (95.50, 91.40, 97.00)],
            [(86.40, 78.20, 94.50), (91.50, 85.50, 95.40), (93.30, 89.90, 96.00), (92.50, 85.10, 95.30)],
            [(95.40, 91.70, 98.80), (100.0, 99.90, 100.0), (100.0, 99.9, 100.0), (100.0, 99.8, 100.0)],
            [(97.60, 94.60, 100.0), (100.0, 100.0, 100.0), (100.0, 100.0, 100.0), (100.0, 100.0, 100.0)],
        ],
        "fpr95": [
            [(21.00, 12.80, 27.80), (21.20, 10.20, 25.00), (15.20, 7.10, 23.50), (18.00, 10.70, 27.82)],
            [(34.10, 18.40, 45.10), (30.30, 18.40, 42.50), (25.50, 16.83, 36.20), (28.10, 18.44, 31.16)],
            [(1.25, 0.00, 5.11), (0.00, 0.00, 0.50), (0.10, 0.00, 1.00), (0.10, 0.00, 1.00)],
            [(5.23, 1.00, 12.38), (0.00, 0.00, 0.00), (0.00, 0.00, 0.00), (0.00, 0.00, 0.00)],
        ],
    },
    "cropsize": {
        "filename": "ablation_cropsize.pdf",
        "xlabel": "Crop size",
        "x_labels": ["96", "128", "192", "256"],
        "auroc": [
            [(94.94, 90.18, 97.76), (95.16, 92.60, 98.17), (94.41, 88.37, 97.44), (93.34, 87.76, 96.86)],
            [(91.42, 87.52, 94.69), (93.32, 89.86, 95.02), (92.09, 90.02, 93.98), (92.63, 90.59, 94.55)],
            [(99.82, 99.55, 99.98), (99.94, 99.77, 100.0), (99.98, 99.87, 100.0), (99.99, 99.93, 100.0)],
            [(99.94, 99.79, 100.0), (100.0, 99.96, 100.0), (99.99, 99.89, 100.0), (99.93, 99.71, 100.0)],
        ],
        "fpr95": [
            [(17.01, 8.57, 27.21), (15.20, 7.10, 23.50), (16.50, 7.82, 29.32), (19.19, 7.82, 32.86)],
            [(33.14, 21.43, 57.29), (25.50, 16.83, 36.20), (27.77, 23.71, 37.89), (29.01, 22.86, 36.46)],
            [(0.77, 0.00, 2.86), (0.01, 0.00, 0.00), (0.01, 0.00, 0.00), (0.00, 0.00, 0.00)],
            [(0.20, 0.00, 1.43), (0.00, 0.00, 0.00), (0.00, 0.00, 0.00), (0.40, 0.00, 1.43)],
        ],
    },
    "numexamples": {
        "filename": "ablation_numexamples.pdf",
        "xlabel": "Number of training examples",
        "x_labels": ["5", "10", "20", "25"],
        "auroc": [
            [(88.79, 83.55, 92.29), (93.00, 89.10, 95.71), (95.16, 89.64, 98.17), (95.52, 92.14, 97.70)],
            [(86.17, 79.18, 91.18), (90.32, 86.33, 93.51), (91.92, 87.86, 95.02), (92.49, 89.29, 95.06)],
            [(99.04, 97.60, 99.76), (99.84, 99.60, 99.96), (99.94, 99.77, 100.0), (99.96, 99.86, 100.0)],
            [(99.68, 98.96, 100.0), (99.95, 99.79, 100.0), (100.0, 99.96, 100.0), (100.0, 99.96, 100.0)],
        ],
        "fpr95": [
            [(26.37, 17.86, 35.00), (19.75, 11.96, 27.14), (16.99, 9.25, 26.46), (14.80, 8.93, 22.86)],
            [(40.21, 29.33, 56.04), (35.11, 24.82, 49.29), (31.57, 23.54, 45.04), (30.33, 22.50, 41.07)],
            [(1.76, 0.00, 4.64), (0.29, 0.00, 1.43), (0.01, 0.00, 0.00), (0.01, 0.00, 0.00)],
            [(1.25, 0.00, 4.29), (0.09, 0.00, 0.71), (0.00, 0.00, 0.00), (0.00, 0.00, 0.00)],
        ],
    },
    "stages": {
        "filename": "ablation_stages.pdf",
        "xlabel": "Swin Transformer stage",
        "x_labels": ["PE", "S1", "S2", "S3", "S4"],
        "auroc": [
            [(91.73, 85.95, 95.51), (91.78, 87.34, 95.00), (92.95, 89.25, 95.70), (94.67, 90.79, 97.45), (95.16, 89.64, 98.17)],
            [(88.50, 83.06, 92.44), (88.61, 83.57, 92.55), (89.87, 85.63, 93.27), (91.24, 87.45, 94.29), (91.92, 87.86, 95.02)],
            [(99.59, 98.81, 99.96), (99.74, 99.23, 99.98), (99.83, 99.49, 99.98), (99.90, 99.66, 100.0), (99.94, 99.77, 100.0)],
            [(99.63, 98.75, 100.0), (99.80, 99.29, 100.0), (99.92, 99.64, 100.0), (99.97, 99.86, 100.0), (100.0, 99.96, 100.0)],
        ],
        "fpr95": [
            [(23.21, 12.86, 34.29), (23.74, 15.00, 33.21), (21.02, 12.14, 30.71), (17.36, 9.64, 26.43), (16.99, 9.25, 26.46)],
            [(37.29, 27.14, 51.07), (37.93, 28.93, 50.00), (35.57, 26.07, 47.14), (33.21, 23.57, 42.50), (31.57, 23.54, 45.04)],
            [(0.93, 0.00, 3.57), (0.60, 0.00, 2.14), (0.40, 0.00, 1.43), (0.26, 0.00, 1.43), (0.01, 0.00, 0.00)],
            [(1.31, 0.00, 4.29), (0.54, 0.00, 2.14), (0.11, 0.00, 0.71), (0.03, 0.00, 0.00), (0.00, 0.00, 0.00)],
        ],
    },
    "crop_strategy": {
        "filename": "ablation_cropstrategy.pdf",
        "xlabel": "Crop strategy",
        "x_labels": ["Center", "Spatial", "Anchored"],
        "auroc": [
            [(93.70, 89.00, 97.00), (94.10, 90.70, 97.30), (95.80, 92.60, 98.50)],
            [(92.80, 90.00, 95.20), (93.30, 90.40, 95.60), (93.30, 89.90, 96.00)],
            [(99.90, 99.60, 100.0), (99.90, 99.70, 100.0), (100.0, 99.90, 100.0)],
            [(100.0, 99.80, 100.0), (100.0, 99.90, 100.0), (100.0, 100.0, 100.0)]
        ],
        "fpr95": [
            [(20.00, 10.70, 30.20), (17.80, 9.20, 26.00), (15.20, 7.10, 23.50)],
            [(25.50, 15.80, 36.20), (27.90, 17.80, 34.20), (23.50, 16.80, 32.20)],
            [(0.30, 0.00, 2.60), (0.30, 0.00, 2.60), (0.10, 0.00, 1.00)],
            [(0.10, 0.00, 1.00), (0.00, 0.00, 1.00), (0.00, 0.00, 0.00)]
        ]
    }
}


def create_subplot(
    ax: plt.Axes,
    x_values: np.ndarray,
    x_labels: list[str],
    results_auroc: list[list[tuple[float, float, float]]],
    results_fpr95: list[list[tuple[float, float, float]]],
    datasets: list[str],
    colors: list[str],
    xlabel: str,
    show_ylabel_left: bool = True,
    show_ylabel_right: bool = True,
) -> plt.Axes:
    """Create a single subplot with AUROC and FPR95 on twin y-axes."""
    for i, _dataset in enumerate(datasets):
        means = np.array([d[0] for d in results_auroc[i]])
        lowers = np.array([d[1] for d in results_auroc[i]])
        uppers = np.array([d[2] for d in results_auroc[i]])
        yerr_lower = np.maximum(means - lowers, 0)
        yerr_upper = np.maximum(uppers - means, 0)
        ax.errorbar(
            x_values,
            means,
            yerr=[yerr_lower, yerr_upper],
            color=colors[i],
            marker="o",
            linewidth=2.5,
            capsize=4,
            capthick=1.5,
            markersize=7,
            alpha=0.9,
        )
        for j in range(len(x_values) - 1):
            ax.fill_between(
                x_values[j : j + 2],
                (means - yerr_lower)[j : j + 2],
                (means + yerr_upper)[j : j + 2],
                color=colors[i],
                alpha=0.1,
            )

    if show_ylabel_left:
        ax.set_ylabel("AUROC", labelpad=-10)
    ax.set_ylim(0, 100)
    ax.set_xticks(x_values)
    ax.set_xticklabels(x_labels, rotation=0 if len(x_labels) <= 5 else 15)
    ax.set_xlabel(xlabel)
    ax.tick_params(axis="y", labelcolor="black")
    ax.tick_params(axis="x")
    ax.grid(alpha=0.25, linestyle="--", linewidth=0.8)

    ax2 = ax.twinx()
    for i, _dataset in enumerate(datasets):
        means = np.array([d[0] for d in results_fpr95[i]])
        lowers = np.array([d[1] for d in results_fpr95[i]])
        uppers = np.array([d[2] for d in results_fpr95[i]])
        yerr_lower = np.maximum(means - lowers, 0)
        yerr_upper = np.maximum(uppers - means, 0)
        ax2.errorbar(
            x_values,
            means,
            yerr=[yerr_lower, yerr_upper],
            color=colors[i],
            marker="x",
            linestyle="--",
            linewidth=2.5,
            capsize=4,
            capthick=1.5,
            markersize=6,
            alpha=0.9,
        )
        for j in range(len(x_values) - 1):
            ax2.fill_between(
                x_values[j : j + 2],
                (means - yerr_lower)[j : j + 2],
                (means + yerr_upper)[j : j + 2],
                color=colors[i],
                alpha=0.1,
            )

    if show_ylabel_right:
        ax2.set_ylabel("FPR95", labelpad=-10)
    ax2.set_ylim(0, 100)
    ax2.tick_params(axis="y", labelcolor="black")
    return ax2


def create_ablation_figure(name: str) -> plt.Figure:
    config = ABLATIONS[name]
    fig, ax = plt.subplots(figsize=(5, 6))
    x_values = np.arange(len(config["x_labels"]))
    create_subplot(
        ax=ax,
        x_values=x_values,
        x_labels=config["x_labels"],
        results_auroc=config["auroc"],
        results_fpr95=config["fpr95"],
        datasets=DATASETS,
        colors=COLORS,
        xlabel=config["xlabel"],
    )
    fig.tight_layout()
    return fig


def save_figures(selected: list[str], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for name in selected:
        fig = create_ablation_figure(name)
        fig.savefig(output_dir / ABLATIONS[name]["filename"], dpi=300, bbox_inches="tight")
        plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate ablation paper figures.")
    parser.add_argument(
        "--figure",
        choices=["all", *ABLATIONS.keys()],
        default="all",
        help="Which ablation figure to render.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(FIGURES_ROOT),
        help="Directory for rendered PDF figures.",
    )
    args = parser.parse_args()

    selected = list(ABLATIONS.keys()) if args.figure == "all" else [args.figure]
    save_figures(selected=selected, output_dir=Path(args.output_dir))


if __name__ == "__main__":
    main()
