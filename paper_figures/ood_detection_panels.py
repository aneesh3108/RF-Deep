"""Render OOD score-distribution panels for paper figures."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from project_paths import FIGURES_ROOT, RESULTS_ROOT


plt.rcParams.update(
    {
        "font.size": 16,
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial"],
        "axes.linewidth": 1.0,
        "xtick.major.width": 1.0,
        "ytick.major.width": 1.0,
        "xtick.major.size": 3,
        "ytick.major.size": 3,
        "legend.frameon": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)


def create_figure(dataframe: pd.DataFrame) -> plt.Figure:
    detector_map = {
        "rsna": "RSNA PE",
        "covid19": "MIDRC C19",
        "kits23": "KiTS",
        "pancreas": "PancreasCT",
    }

    fig, axes = plt.subplots(1, 4, figsize=(12, 4), sharey=True)
    color_id = "royalblue"
    color_ood = "darkorange"
    roman = ["(i)", "(ii)", "(iii)", "(iv)"]

    for idx, (ax, detector) in enumerate(zip(axes.flatten(), detector_map)):
        subset = dataframe[dataframe["ood_detector"] == detector]
        scores_id = subset[subset["true_label"] == "ID"]["avg_pred_prob"].values
        scores_ood = subset[subset["true_label"] == "OOD"]["avg_pred_prob"].values

        violins = ax.violinplot(
            [scores_id, scores_ood],
            positions=[0, 1],
            showmeans=False,
            showextrema=True,
            showmedians=True,
        )

        for body_index, body in enumerate(violins["bodies"]):
            body.set_facecolor(color_id if body_index == 0 else color_ood)
            body.set_alpha(0.7)
            body.set_edgecolor("black")
            body.set_linewidth(1.0)

        if "cmedians" in violins:
            violins["cmedians"].set_color("black")
            violins["cmedians"].set_linewidth(1.5)
        for part in ("cbars", "cmins", "cmaxes"):
            if part in violins:
                violins[part].set_color("black")
                violins[part].set_linewidth(1.0)

        ax.set_xticks([0, 1])
        ax.set_xticklabels(["ID", "OOD"])
        ax.set_ylim(-0.05, 1.05)
        ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
        ax.grid(axis="y", alpha=0.3, linestyle="--", linewidth=0.5)
        ax.set_axisbelow(True)

        if idx == 0:
            ax.set_ylabel("OOD detection score ($S_{OOD}$)")

        ax.text(0.02, -0.25, roman[idx], transform=ax.transAxes, va="bottom", ha="left")
        ax.text(0.15, -0.25, detector_map[detector], transform=ax.transAxes, va="bottom", ha="left")

    fig.tight_layout()
    plt.subplots_adjust(hspace=0.4, wspace=0.15, top=0.95)
    return fig


def main() -> None:
    parser = argparse.ArgumentParser(description="Render OOD violin-panel figures.")
    parser.add_argument(
        "--input-csv",
        default=str(RESULTS_ROOT / "per_scan_analysis_smit_runs100_seed2109.csv"),
        help="Per-scan analysis CSV to visualize.",
    )
    parser.add_argument(
        "--output",
        default=str(FIGURES_ROOT / "ood_detection_panels_rfdeep.pdf"),
        help="Output PDF path.",
    )
    args = parser.parse_args()

    df = pd.read_csv(args.input_csv)
    fig = create_figure(df)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote figure to {output_path}")


if __name__ == "__main__":
    main()
