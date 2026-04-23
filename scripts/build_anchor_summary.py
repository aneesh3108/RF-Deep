"""
Build a per-scan anchor summary table from prediction masks and logits.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import sys
from typing import Any, Optional

import nibabel as nib
import numpy as np
import pandas as pd
from scipy.ndimage import label as label_connected_components
from scipy.ndimage import generate_binary_structure
from tqdm import tqdm

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from project_paths import ANALYSIS_RESULTS_ROOT, RESULTS_ROOT, encode_manifest_path


DEFAULT_DATASETS = ["lrad", "rsna", "covid19", "kits23", "pancreas", "breastc", "covid19a"]


def prediction_root(model_name: str, dataset: str) -> Path:
    if dataset == "lrad":
        return RESULTS_ROOT / f"{model_name}_main" / "lung_test_LRAD_srcnorm"
    return RESULTS_ROOT / f"{model_name}_farood" / f"lung_test_{dataset}_srcnorm"


def component_summary(mask: np.ndarray, min_component_size: int) -> dict[str, float]:
    structure = generate_binary_structure(3, 3)

    raw_labels, raw_count = label_connected_components(mask.astype(np.uint8), structure=structure)
    raw_sizes = np.bincount(raw_labels.ravel())
    raw_component_sizes = raw_sizes[1:]

    keep_labels = np.where(raw_sizes >= min_component_size)[0]
    keep_labels = keep_labels[keep_labels != 0]
    kept_component_sizes = raw_sizes[keep_labels] if keep_labels.size else np.array([], dtype=raw_sizes.dtype)

    total_voxels = int(mask.sum())
    kept_count = int(len(keep_labels))
    kept_voxels = int(kept_component_sizes.sum()) if kept_component_sizes.size else 0
    largest_raw = int(raw_component_sizes.max()) if raw_component_sizes.size else 0
    largest_kept = int(kept_component_sizes.max()) if kept_component_sizes.size else 0

    return {
        "foreground_voxels_raw": total_voxels,
        "foreground_voxels_kept": kept_voxels,
        "component_count_raw": int(raw_count),
        "component_count_kept": int(kept_count),
        "largest_component_voxels_raw": largest_raw,
        "largest_component_voxels_kept": largest_kept,
        "largest_component_fraction_raw": float(largest_raw / total_voxels) if total_voxels else np.nan,
        "largest_component_fraction_kept": float(largest_kept / kept_voxels) if kept_voxels else np.nan,
        "no_prediction_raw": bool(total_voxels == 0),
        "no_prediction_kept": bool(kept_voxels == 0),
    }


def summarize_scan(
    dataset: str,
    nii_path: Path,
    min_component_size: int,
) -> dict[str, Any]:
    image = nib.load(str(nii_path))
    seg = np.asanyarray(image.dataobj) > 0
    zooms = image.header.get_zooms()[:3]
    voxel_volume_mm3 = float(np.prod(zooms))
    comp = component_summary(seg, min_component_size=min_component_size)

    row = {
        "dataset": dataset,
        "filename": nii_path.name,
        "case_id": nii_path.name.replace(".nii.gz", ""),
        "nii_path": encode_manifest_path(nii_path),
        "shape_x": int(seg.shape[0]),
        "shape_y": int(seg.shape[1]),
        "shape_z": int(seg.shape[2]),
        "spacing_x": float(zooms[0]),
        "spacing_y": float(zooms[1]),
        "spacing_z": float(zooms[2]),
        "voxel_volume_mm3": voxel_volume_mm3,
        "foreground_volume_cc_raw": float(comp["foreground_voxels_raw"] * voxel_volume_mm3 / 1000.0),
        "foreground_volume_cc_kept": float(comp["foreground_voxels_kept"] * voxel_volume_mm3 / 1000.0),
        "largest_component_volume_cc_raw": float(comp["largest_component_voxels_raw"] * voxel_volume_mm3 / 1000.0),
        "largest_component_volume_cc_kept": float(comp["largest_component_voxels_kept"] * voxel_volume_mm3 / 1000.0),
        "min_component_size": int(min_component_size),
        **comp,
    }

    return row


def _worker(args: tuple[str, Path, int]) -> dict[str, Any]:
    dataset, nii_path, min_component_size = args
    return summarize_scan(dataset, nii_path, min_component_size)


def build_anchor_summary(
    model_name: str,
    datasets: list[str],
    min_component_size: int,
    num_workers: int,
) -> pd.DataFrame:
    jobs: list[tuple[str, Path, int]] = []
    for dataset in datasets:
        root = prediction_root(model_name, dataset)
        nii_dir = root / "nii"
        if not nii_dir.is_dir():
            raise FileNotFoundError(f"Prediction directory not found: {nii_dir}")

        nii_files = sorted(nii_dir.glob("*.nii.gz"))
        print(f"{dataset}: {len(nii_files)} scans queued")
        for nii_path in nii_files:
            jobs.append((dataset, nii_path, min_component_size))

    print(f"\nProcessing {len(jobs)} scans across {num_workers} worker(s)...\n")

    if num_workers == 1:
        rows = []
        for job in tqdm(jobs, desc="Anchor summary", unit="scan"):
            rows.append(_worker(job))
        return pd.DataFrame(rows)

    rows: list[Optional[dict[str, Any]]] = [None] * len(jobs)
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        future_to_idx = {
            executor.submit(_worker, job): idx for idx, job in enumerate(jobs)
        }
        with tqdm(total=len(jobs), desc="Anchor summary", unit="scan") as pbar:
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                dataset, nii_path, _ = jobs[idx]
                try:
                    rows[idx] = future.result()
                except Exception as exc:
                    print(f"\nERROR {dataset}/{nii_path.name}: {exc}")
                    rows[idx] = {
                        "dataset": dataset,
                        "filename": nii_path.name,
                        "case_id": nii_path.name.replace(".nii.gz", ""),
                        "nii_path": encode_manifest_path(nii_path),
                        "error": str(exc),
                    }
                pbar.update(1)

    return pd.DataFrame([row for row in rows if row is not None])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a per-scan anchor summary table from predictions.")
    parser.add_argument("--model-name", default="smit")
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=DEFAULT_DATASETS,
        help="Datasets to summarize.",
    )
    parser.add_argument(
        "--min-component-size",
        type=int,
        default=300,
        help="Minimum voxels for the filtered/kept connected-component summary.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=8,
        help="Number of worker processes for per-scan summarization.",
    )
    parser.add_argument(
        "--output-path",
        default=str(ANALYSIS_RESULTS_ROOT / "anchor_summary" / "smit_anchor_summary.csv"),
        help="CSV output path.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = build_anchor_summary(
        model_name=args.model_name,
        datasets=args.datasets,
        min_component_size=args.min_component_size,
        num_workers=args.num_workers,
    )
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"\nSaved {len(df)} rows to: {output_path}")


if __name__ == "__main__":
    main()
