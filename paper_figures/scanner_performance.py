'''
Paper figure utilities for overall and scanner-stratified performance plots.
'''

from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from project_paths import EXCEL_RECORDS_ROOT, FIGURES_ROOT, METADATA_ROOT


MODEL_SPECS = [
    ("SMIT", "lung_smit_test_LRAD_0.5.csv"),
    ("SimMIM", "lung_mim_test_LRAD_0.5.csv"),
    ("iBOT", "lung_ibot_test_LRAD_0.5.csv"),
    ("SMIT Lite", "lung_smitmini_test_LRAD_0.5.csv"),
    ("SwinUNETR", "lung_swinunetr_10k_test_LRAD_0.5.csv"),
    (r"SwinUNETR$^{\dagger}$", "lung_swinunetr_test_LRAD_0.5.csv"),
]

STRATIFICATIONS = [
    ("ManufacturerGroup", ["GE", "Non-GE"], "Manufacturer"),
    ("ContrastGroup", ["Contrast", "Non-Contrast"], "Contrast"),
    ("KernelGroup", ["Recon1", "Recon2", "Recon3"], "Kernel"),
]

RADAR_CATEGORIES = [
    ("ManufacturerGroup", "GE", "GE"),
    ("ManufacturerGroup", "Non-GE", "Non-GE"),
    ("ContrastGroup", "Contrast", "Contrast"),
    ("ContrastGroup", "Non-Contrast", "Non-contrast"),
    ("KernelGroup", "Recon1", "Recon1"),
    ("KernelGroup", "Recon2", "Recon2"),
    ("KernelGroup", "Recon3", "Recon3"),
]


def _mean_se(values: pd.Series) -> tuple[float, float, int]:
    values = values.dropna()
    count = len(values)
    if count == 0:
        return np.nan, np.nan, 0
    mean = values.mean()
    se = values.std(ddof=1) / np.sqrt(count) if count > 1 else 0.0
    return mean, se, count


def _mean_std(values: pd.Series) -> tuple[float, float, int]:
    values = values.dropna()
    count = len(values)
    if count == 0:
        return np.nan, np.nan, 0
    mean = values.mean()
    std = values.std(ddof=1) if count > 1 else 0.0
    return mean, std, count


def _siemens_band(kernel_string: str) -> int | None:
    token = str(kernel_string).upper()
    match = re.search(r"\b[A-Z]\s*([0-9]{2})", token)
    if match:
        return int(match.group(1))
    match = re.search(r"([0-9]{2,})", token)
    if match:
        return int(match.group(1)[:2])
    return None


def _ge_bucket(token: str) -> str | None:
    if "LUNG" in token:
        return "Recon3"
    if "BONE PLUS" in token or "BONEPLUS" in token:
        return "Recon2"
    if "BONE" in token or "STANDARD" in token:
        return "Recon1"
    return None


def _kernel_group(row: pd.Series) -> str:
    manufacturer = str(row.get("ManufacturerUp", "")).upper().strip()
    kernel = str(row.get("KernelRawUp", "")).upper().strip()
    if not kernel or kernel in {"NAN", "NONE"}:
        return "Unknown"

    tokens = [token.strip() for token in re.split(r"[;,/|]+", kernel) if token.strip()]

    def classify_siemens(values: list[str]) -> str | None:
        bands = []
        for value in values:
            band = _siemens_band(value)
            if band is None:
                continue
            if band < 40:
                bands.append("Recon1")
            elif band < 50:
                bands.append("Recon2")
            else:
                bands.append("Recon3")
        if not bands:
            return None
        if "Recon3" in bands:
            return "Recon3"
        if "Recon2" in bands:
            return "Recon2"
        return "Recon1"

    def classify_ge(values: list[str]) -> str | None:
        labels = [label for label in (_ge_bucket(value) for value in values) if label]
        if not labels:
            return None
        if "Recon3" in labels:
            return "Recon3"
        if "Recon2" in labels:
            return "Recon2"
        return "Recon1"

    if "SIEMENS" in manufacturer:
        return classify_siemens(tokens) or classify_ge(tokens) or "Unknown"
    if "GE" in manufacturer or "GENERAL ELECTRIC" in manufacturer:
        return classify_ge(tokens) or classify_siemens(tokens) or "Unknown"
    return classify_ge(tokens) or classify_siemens(tokens) or "Unknown"


def load_scanner_metadata(metadata_path: str | Path) -> pd.DataFrame:
    meta = pd.read_excel(metadata_path)
    meta = meta.rename(
        columns={"ID": "CaseID", "Contras/orNo": "Contrast", "Manufacture": "Manufacturer"}
    )
    meta["CaseID"] = meta["CaseID"].astype(str)
    meta["ManufacturerUp"] = (
        meta["Manufacturer"]
        .astype(str)
        .str.strip()
        .replace({"": "UNKNOWN", "nan": "UNKNOWN", "None": "UNKNOWN", "NONE": "UNKNOWN"})
        .str.upper()
    )
    meta["ManufacturerGroup"] = meta["ManufacturerUp"].apply(
        lambda value: "GE" if value == "GE" else "Non-GE"
    )
    contrast_map = {"CONTRAST": "Contrast", "NONCONTRAST": "Non-Contrast"}
    meta["ContrastGroup"] = meta["Contrast"].map(
        lambda value: contrast_map.get(str(value).strip().upper(), str(value))
    )

    kernel_columns = [
        "Kernel",
        "ConvolutionKernel",
        "ReconKernel",
        "ReconstructionKernel",
        "ConvKernel",
    ]
    kernel_col = next((column for column in kernel_columns if column in meta.columns), None)
    if kernel_col is None:
        raise ValueError(f"No kernel column found in metadata columns: {', '.join(meta.columns)}")

    meta["KernelRawUp"] = meta[kernel_col].astype(str).str.upper().str.strip()
    meta["KernelGroup"] = meta.apply(_kernel_group, axis=1)
    return meta[meta["KernelGroup"] != "Unknown"].copy()


def load_model_scores(scores_dir: str | Path) -> pd.DataFrame:
    scores_dir = Path(scores_dir)
    frames = []
    smit_names: set[str] | None = None

    for index, (_, filename) in enumerate(MODEL_SPECS):
        frame = pd.read_csv(scores_dir / filename)
        frame["CaseID"] = frame["name"].str.replace(".nii.gz", "", regex=False)
        if index == 0:
            smit_names = set(frame["name"].tolist())
        frame = frame[["name", "CaseID", "dice", "hd95", "precision", "recall"]].rename(
            columns={
                "dice": f"dice_{index}",
                "hd95": f"hd95_{index}",
                "precision": f"precision_{index}",
                "recall": f"recall_{index}",
            }
        )
        frames.append(frame)

    merged = frames[0]
    for frame in frames[1:]:
        merged = merged.merge(frame, on=["name", "CaseID"], how="outer")

    merged = merged[merged["name"].isin(smit_names)].reset_index(drop=True)

    for index in range(len(MODEL_SPECS)):
        dice_col = f"dice_{index}"
        hd95_col = f"hd95_{index}"
        merged[dice_col] = merged[dice_col].fillna(0.0)
        merged[hd95_col] = merged[hd95_col].where(pd.notna(merged[hd95_col]), np.nan)

    return merged


def prepare_scanner_analysis(scores_dir: str | Path, metadata_path: str | Path) -> pd.DataFrame:
    merged = load_model_scores(scores_dir)
    metadata = load_scanner_metadata(metadata_path)
    return merged.merge(
        metadata[["CaseID", "ManufacturerGroup", "ContrastGroup", "KernelGroup"]],
        on="CaseID",
        how="inner",
    )


def create_overall_figure(merged: pd.DataFrame) -> plt.Figure:
    model_labels = [spec[0].replace("$^{\\dagger}$", "†") for spec in MODEL_SPECS]
    dsc_mean = []
    dsc_std = []
    hd95_mean = []
    hd95_std = []

    for index in range(len(MODEL_SPECS)):
        mean, std, _ = _mean_std(merged[f"dice_{index}"])
        dsc_mean.append(mean)
        dsc_std.append(std)
        mean, std, _ = _mean_std(merged[f"hd95_{index}"])
        hd95_mean.append(mean)
        hd95_std.append(std)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(5, 10))
    y = np.arange(len(model_labels))
    colors = sns.color_palette("colorblind", n_colors=len(MODEL_SPECS))
    colors[-1] = sns.color_palette("colorblind")[7]

    ax1.barh(
        y,
        dsc_mean,
        0.6,
        xerr=dsc_std,
        capsize=5,
        color=colors,
        alpha=0.8,
        error_kw={"linewidth": 2, "elinewidth": 1.5},
    )
    ax1.set_yticks(y)
    ax1.set_yticklabels(model_labels, fontsize=16)
    ax1.set_xlim([0.5, 0.8])
    ax1.grid(axis="x", alpha=0.3, linestyle="--", linewidth=0.7)
    ax1.axvline(x=dsc_mean[0], color=colors[0], linestyle="--", alpha=0.5, linewidth=2)
    ax1.invert_yaxis()
    ax1.tick_params(axis="x", labelsize=16)
    ax1.set_xlabel("DSC", fontsize=16)

    ax2.barh(
        y,
        hd95_mean,
        0.6,
        xerr=hd95_std,
        capsize=5,
        color=colors,
        alpha=0.8,
        error_kw={"linewidth": 2, "elinewidth": 1.5},
    )
    ax2.set_yticks(y)
    ax2.set_yticklabels(model_labels, fontsize=16)
    ax2.set_xlim([4, 8])
    ax2.grid(axis="x", alpha=0.3, linestyle="--", linewidth=0.7)
    ax2.axvline(x=hd95_mean[0], color=colors[0], linestyle="--", alpha=0.5, linewidth=2)
    ax2.invert_yaxis()
    ax2.tick_params(axis="x", labelsize=16)
    ax2.set_xlabel("HD95 (mm)", fontsize=16)

    fig.tight_layout()
    return fig


def create_radar_plot(merged: pd.DataFrame) -> plt.Figure:
    model_data = {}
    for index, (label, _) in enumerate(MODEL_SPECS):
        values = []
        for column, group, _ in RADAR_CATEGORIES:
            subset = merged[merged[column] == group]
            values.append(subset[f"dice_{index}"].mean())
        model_data[label] = [0.0 if np.isnan(value) else value for value in values]

    fig = plt.figure(figsize=(18, 10))
    grid = fig.add_gridspec(1, 2, width_ratios=[0.8, 1])
    ax = fig.add_subplot(grid[0, 0], projection="polar")
    filler = fig.add_subplot(grid[0, 1])
    filler.axis("off")

    labels = [label for _, _, label in RADAR_CATEGORIES]
    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
    angles += angles[:1]

    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    colors = sns.color_palette("colorblind", n_colors=len(MODEL_SPECS))
    colors[-1] = sns.color_palette("colorblind")[7]

    for index, (label, values) in enumerate(model_data.items()):
        closed_values = values + values[:1]
        ax.plot(
            angles,
            closed_values,
            "o-",
            linewidth=2.5,
            label=label,
            color=colors[index],
            markersize=8,
            alpha=0.9,
        )

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, size=20)
    ax.set_ylim(0, 1.0)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(["0.2", "0.4", "0.6", "0.8", "1.0"], size=24)
    ax.grid(True, linestyle="--", alpha=0.7, linewidth=1.5, color="gray")
    ax.set_axisbelow(True)
    ax.legend(
        loc="lower right",
        bbox_to_anchor=(1.3, -0.05),
        fontsize=20,
        frameon=True,
        fancybox=True,
        shadow=True,
    )

    fig.tight_layout()
    return fig


def create_supplemental_figure(merged: pd.DataFrame) -> plt.Figure:
    fig = plt.figure(figsize=(18, 8))
    grid = fig.add_gridspec(2, 3, height_ratios=[1, 1], hspace=0.2, wspace=0.2)
    colors = sns.color_palette("colorblind", n_colors=len(MODEL_SPECS))
    handles = None
    labels = None

    for metric_index, metric in enumerate(["dice", "hd95"]):
        for strat_index, (column, groups, title) in enumerate(STRATIFICATIONS):
            ax = fig.add_subplot(grid[metric_index, strat_index])
            bar_width = 0.13
            x_positions = np.arange(len(groups))

            for model_index, (label, _) in enumerate(MODEL_SPECS):
                means = []
                errors = []
                for group in groups:
                    subset = merged[merged[column] == group]
                    mean, se, _ = _mean_se(subset[f"{metric}_{model_index}"])
                    means.append(0.0 if np.isnan(mean) else mean)
                    errors.append(0.0 if np.isnan(se) else se)

                offset = (model_index - len(MODEL_SPECS) / 2 + 0.5) * bar_width
                ax.bar(
                    x_positions + offset,
                    means,
                    bar_width,
                    yerr=errors,
                    capsize=4,
                    alpha=0.8,
                    label=label,
                    color=colors[model_index],
                    error_kw={"linewidth": 1.5, "elinewidth": 1.5},
                )

            if handles is None:
                handles, labels = ax.get_legend_handles_labels()

            ax.set_xticks(x_positions)
            ax.set_xticklabels(groups, fontsize=18)
            ax.tick_params(axis="y", labelsize=18)
            if strat_index == 0:
                ax.set_ylabel("DSC" if metric == "dice" else "HD95 (mm)", fontsize=18)
            ax.grid(axis="y", linestyle="--", alpha=0.3)
            ax.set_ylim(0, 1.0 if metric == "dice" else 10)

            if metric_index == 1:
                letter = chr(97 + strat_index)
                ax.text(
                    0.5,
                    -0.25,
                    f"({letter}) {title}",
                    transform=ax.transAxes,
                    fontsize=20,
                    ha="center",
                    va="top",
                )

    fig.legend(handles, labels, loc="lower center", ncol=6, fontsize=18, framealpha=0.9)
    fig.tight_layout(rect=[0, 0.06, 1, 1])
    return fig


def create_scanner_performance_figures(
    scores_dir: str | Path = EXCEL_RECORDS_ROOT,
    metadata_path: str | Path = METADATA_ROOT / "scanner_meta_info_LRad.xlsx",
) -> dict[str, plt.Figure]:
    merged = prepare_scanner_analysis(scores_dir=scores_dir, metadata_path=metadata_path)
    return {
        "overall": create_overall_figure(merged),
        "radar": create_radar_plot(merged),
        "supplemental": create_supplemental_figure(merged),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate overall and scanner-stratified paper figures.")
    parser.add_argument("--scores-dir", default=str(EXCEL_RECORDS_ROOT), help="Directory with per-model metric CSVs.")
    parser.add_argument(
        "--metadata-path",
        default=str(METADATA_ROOT / "scanner_meta_info_LRad.xlsx"),
        help="Scanner metadata Excel file.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(FIGURES_ROOT),
        help="Directory to save generated figures.",
    )
    args = parser.parse_args()

    figures = create_scanner_performance_figures(
        scores_dir=args.scores_dir,
        metadata_path=args.metadata_path,
    )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    figures["overall"].savefig(output_dir / "id_segmentation_performance.pdf", bbox_inches="tight", facecolor="white")
    figures["radar"].savefig(output_dir / "scanner_strata_radar.pdf", format="pdf", bbox_inches="tight", dpi=300)
    figures["supplemental"].savefig(
        output_dir / "scanner_strata_supplemental.pdf",
        format="pdf",
        bbox_inches="tight",
        dpi=300,
    )


if __name__ == "__main__":
    main()
