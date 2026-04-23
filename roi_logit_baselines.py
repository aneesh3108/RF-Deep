"""
ROI-restricted logit baselines using the same crop protocol as RF-Deep.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import os.path as osp
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
from scipy.ndimage import zoom
from logit_baselines import (
    compute_all_ood_metrics,
    compute_all_ood_metrics_bootstrap,
    compute_energy_score,
    compute_max_softmax,
    print_summary_statistics,
)
from project_paths import JSONS_ROOT, LOGITS_ROOT, LOGIT_BASELINES_RESULTS_ROOT, load_manifest_entries


DATASET_CONFIGS = {
    "lrad": {
        "display_name": "ID (Radiogenomics)",
        "orientation": "RAS",
        "result_subdir": Path("smit_main") / "lung_test_LRAD_srcnorm",
    },
    "rsna": {
        "display_name": "RSNA PE",
        "orientation": "PLI",
        "result_subdir": Path("smit_farood") / "lung_test_rsna_srcnorm",
    },
    "covid19": {
        "display_name": "MIDRC C19",
        "orientation": "RAS",
        "result_subdir": Path("smit_farood") / "lung_test_covid19_srcnorm",
    },
    "kits23": {
        "display_name": "KiTS",
        "orientation": "RAS",
        "result_subdir": Path("smit_farood") / "lung_test_kits23_srcnorm",
    },
    "pancreas": {
        "display_name": "Pancreas",
        "orientation": "RAS",
        "result_subdir": Path("smit_farood") / "lung_test_pancreas_srcnorm",
    },
    "breastc": {
        "display_name": "Breast Cancer CT",
        "orientation": "RAS",
        "result_subdir": Path("smit_farood") / "lung_test_breastc_srcnorm",
    },
    "covid19a": {
        "display_name": "MIDRC C19+",
        "orientation": "RAS",
        "result_subdir": Path("smit_farood") / "lung_test_covid19a_srcnorm",
    },
}

DEFAULT_DATASETS = ["lrad", "rsna", "covid19", "kits23", "pancreas", "breastc", "covid19a"]
ROI_SIZE = (128, 128, 128)
N_ROIS = 4
TARGET_SPACING = (1.0, 1.0, 1.0)


def _to_jsonable(value):
    if isinstance(value, dict):
        return {k: _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    return value


def result_root(dataset_key: str) -> Path:
    return Path(LOGITS_ROOT) / DATASET_CONFIGS[dataset_key]["result_subdir"]


def resolve_scan_entry(model_name: str, dataset_key: str, entry: dict) -> dict:
    filename = osp.basename(entry["image"])
    label_path = result_root(dataset_key) / "nii" / filename
    logits_path = result_root(dataset_key) / "numpy" / filename.replace(".nii.gz", ".npy")
    if not label_path.exists():
        raise FileNotFoundError(f"Anchor NIfTI not found for {filename}: {label_path}")
    if not logits_path.exists():
        raise FileNotFoundError(f"Logits not found for {filename}: {logits_path}")
    return {
        "label": str(label_path),
        "logits": str(logits_path),
        "filename": filename,
        "dataset_key": dataset_key,
        "dataset": DATASET_CONFIGS[dataset_key]["display_name"],
    }


def sample_roi_slices(mask: np.ndarray, roi_size: tuple[int, int, int], n_rois: int, seed: int):
    rng = np.random.default_rng(seed)
    coords = np.argwhere(mask > 0)
    if coords.size == 0:
        return []

    chosen_indices = rng.choice(len(coords), size=n_rois, replace=len(coords) < n_rois)
    slices = []
    for center in coords[chosen_indices]:
        roi_slices = []
        for axis, dim in enumerate(roi_size):
            half = dim // 2
            start = int(center[axis]) - half
            end = start + dim
            if start < 0:
                start = 0
                end = min(dim, mask.shape[axis])
            if end > mask.shape[axis]:
                end = mask.shape[axis]
                start = max(0, end - dim)
            roi_slices.append(slice(start, end))
        slices.append(tuple(roi_slices))
    return slices


def crop_with_padding(array: np.ndarray, roi_slices: tuple[slice, slice, slice], roi_size: tuple[int, int, int]):
    cropped = array[(...,) + roi_slices] if array.ndim == 4 else array[roi_slices]
    pad_width = []
    if array.ndim == 4:
        pad_width.append((0, 0))
    for axis, target in enumerate(roi_size):
        actual = cropped.shape[axis + (1 if array.ndim == 4 else 0)]
        pad_width.append((0, max(0, target - actual)))
    if any(pad != (0, 0) for pad in pad_width):
        cropped = np.pad(cropped, pad_width, mode="constant", constant_values=0)
    return cropped


def resample_mask_and_logits(mask_img: nib.Nifti1Image, logits: np.ndarray):
    mask = np.asanyarray(mask_img.dataobj) > 0
    src_spacing = tuple(float(v) for v in mask_img.header.get_zooms()[:3])
    zoom_factors = tuple(src / tgt for src, tgt in zip(src_spacing, TARGET_SPACING))

    # If already ~1 mm isotropic, avoid needless interpolation.
    if all(abs(f - 1.0) < 1e-6 for f in zoom_factors):
        return mask, logits, src_spacing

    resampled_mask = zoom(mask.astype(np.float32), zoom_factors, order=0) > 0
    resampled_logits = np.stack(
        [zoom(logits[c], zoom_factors, order=1) for c in range(logits.shape[0])],
        axis=0,
    ).astype(np.float32)
    return resampled_mask, resampled_logits, src_spacing


def summarize_scan(job):
    entry, min_component_size, roi_seed = job
    try:
        mask_img = nib.load(entry["label"])
        logits = np.load(entry["logits"]).astype(np.float32)
        if mask_img.shape != logits.shape[1:]:
            raise ValueError(
                f"Shape mismatch for {entry['filename']}: "
                f"mask {mask_img.shape} vs logits {logits.shape[1:]}"
            )
        mask, logits, src_spacing = resample_mask_and_logits(mask_img, logits)
        roi_slices_list = sample_roi_slices(mask, ROI_SIZE, N_ROIS, roi_seed)

        roi_rows = []
        for roi_idx, roi_slices in enumerate(roi_slices_list):
            roi_mask = crop_with_padding(mask.astype(np.uint8), roi_slices, ROI_SIZE).astype(bool)
            if roi_mask.sum() < min_component_size:
                continue
            roi_logits = crop_with_padding(logits, roi_slices, ROI_SIZE)
            maxlogit = np.max(roi_logits, axis=0)
            maxsoftmax = compute_max_softmax(roi_logits)
            energy = compute_energy_score(roi_logits)

            roi_rows.append(
                {
                    "maxlogit": float(maxlogit[roi_mask].mean()),
                    "maxsoftmax": float(maxsoftmax[roi_mask].mean()),
                    "energy": float(energy[roi_mask].mean()),
                    "roi_idx": int(roi_idx),
                }
            )

        if not roi_rows:
            return {
                "scan_name": entry["filename"].replace(".nii.gz", ".npy"),
                "dataset": entry["dataset"],
                "status": "no_segmentation",
            }

        return {
            "scan_name": entry["filename"].replace(".nii.gz", ".npy"),
            "dataset": entry["dataset"],
            "status": "ok",
            "row": {
                "scan_name": entry["filename"].replace(".nii.gz", ".npy"),
                "dataset": entry["dataset"],
                "maxlogit": float(np.mean([r["maxlogit"] for r in roi_rows])),
                "maxsoftmax": float(np.mean([r["maxsoftmax"] for r in roi_rows])),
                "energy": float(np.mean([r["energy"] for r in roi_rows])),
                "n_valid_rois": int(len(roi_rows)),
                "source_spacing": tuple(src_spacing),
            },
        }
    except Exception as exc:
        return {
            "scan_name": entry["filename"].replace(".nii.gz", ".npy"),
            "dataset": entry["dataset"],
            "status": "error",
            "error": str(exc),
        }


def compute_dataset_statistics_roi(
    model_name: str,
    dataset_key: str,
    json_dir: Path,
    n_jobs: int = 4,
    base_seed: int = 42,
    min_component_size: int = 10,
) -> pd.DataFrame:
    json_path = Path(json_dir) / f"{model_name}_{dataset_key}_src.json"
    entries = load_manifest_entries(json_path, "validation")
    jobs = []
    for idx, entry in enumerate(entries):
        scan_entry = resolve_scan_entry(model_name, dataset_key, entry)
        jobs.append((scan_entry, min_component_size, base_seed + idx))

    rows = []
    n_processed = 0
    if jobs:
        with ProcessPoolExecutor(max_workers=n_jobs) as executor:
            for result in executor.map(summarize_scan, jobs):
                if result["status"] == "ok":
                    rows.append(result["row"])
                    n_processed += 1
                    print(f"  {result['scan_name']}... ✓")
                elif result["status"] == "no_segmentation":
                    print(f"  {result['scan_name']}... no valid ROI")
                else:
                    print(f"  {result['scan_name']}... ERROR: {result['error']}")

    print(f"\n  {DATASET_CONFIGS[dataset_key]['display_name']}: {n_processed} scans processed")
    return pd.DataFrame(rows)


def write_results_json(output_path: Path, payload: dict) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(_to_jsonable(payload), handle, indent=2)
    print(f"\nSaved ROI logit-baseline JSON to: {output_path}")


def build_parser():
    parser = argparse.ArgumentParser(
        description="ROI-restricted logit baselines using predicted-mask-guided 1 mm isotropic ROI crops."
    )
    parser.add_argument("--model-name", default="smit")
    parser.add_argument(
        "--metric",
        default="maxlogit",
        choices=["maxlogit", "maxsoftmax", "energy"],
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=DEFAULT_DATASETS,
        choices=DEFAULT_DATASETS,
        help="Datasets to include; lrad should remain first as the ID reference.",
    )
    parser.add_argument("--json-dir", default=str(JSONS_ROOT))
    parser.add_argument("--n-bootstrap", type=int, default=100)
    parser.add_argument("--sample-fraction", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-jobs", type=int, default=4)
    parser.add_argument("--min-component-size", type=int, default=10)
    parser.add_argument("--save-json", action="store_true")
    parser.add_argument("--json-path", default=None)
    return parser


def main():
    args = build_parser().parse_args()
    datasets = list(args.datasets)
    if not datasets or datasets[0] != "lrad":
        datasets = ["lrad"] + [ds for ds in datasets if ds != "lrad"]

    all_dfs = []
    dataset_summaries = {}
    print("\nUsing ROI-restricted aggregation: resample to 1 mm isotropic, then mean across 4 RF-Deep-style ROIs")
    for dataset_key in datasets:
        display_name = DATASET_CONFIGS[dataset_key]["display_name"]
        print(f"\n{'=' * 80}\nProcessing: {display_name}\n{'=' * 80}")
        df = compute_dataset_statistics_roi(
            model_name=args.model_name,
            dataset_key=dataset_key,
            json_dir=Path(args.json_dir),
            n_jobs=args.n_jobs,
            base_seed=args.seed,
            min_component_size=args.min_component_size,
        )
        all_dfs.append(df)
        dataset_summaries[display_name] = {
            "n_scans": int(len(df)),
            "maxlogit_mean": float(df["maxlogit"].mean()) if not df.empty else np.nan,
            "maxlogit_std": float(df["maxlogit"].std()) if not df.empty else np.nan,
            "maxsoftmax_mean": float(df["maxsoftmax"].mean()) if not df.empty else np.nan,
            "maxsoftmax_std": float(df["maxsoftmax"].std()) if not df.empty else np.nan,
            "energy_mean": float(df["energy"].mean()) if not df.empty else np.nan,
            "energy_std": float(df["energy"].std()) if not df.empty else np.nan,
        }
        print_summary_statistics(df)

    id_df = all_dfs[0]
    ood_dfs = all_dfs[1:]

    ood_results = compute_all_ood_metrics(id_df, ood_dfs, metric_name=args.metric)
    print(f"\nRunning bootstrap (n={args.n_bootstrap}, seed={args.seed}) ...")
    bootstrap_results = compute_all_ood_metrics_bootstrap(
        id_df,
        ood_dfs,
        metric_name=args.metric,
        n_bootstrap=args.n_bootstrap,
        sample_fraction=args.sample_fraction,
        random_seed=args.seed,
    )

    if args.save_json:
        dataset_part = "_".join(datasets)
        output_path = Path(args.json_path) if args.json_path else (
            LOGIT_BASELINES_RESULTS_ROOT / "roi" /
            f"roi_{args.metric}_{dataset_part}.json"
        )
        write_results_json(
            output_path,
            {
                "metric": args.metric,
                "aggregation": "roi_mean_over_4_crops_1mm_isotropic",
                "selected_datasets": datasets,
                "n_bootstrap": int(args.n_bootstrap),
                "sample_fraction": float(args.sample_fraction),
                "seed": int(args.seed),
                "n_jobs": int(args.n_jobs),
                "min_component_size": int(args.min_component_size),
                "target_spacing": list(TARGET_SPACING),
                "dataset_summaries": dataset_summaries,
                "ood_metrics": ood_results,
                "ood_metrics_bootstrap": bootstrap_results,
            },
        )


if __name__ == "__main__":
    main()
