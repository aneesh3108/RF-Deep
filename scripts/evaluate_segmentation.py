"""
Evaluate segmentation outputs against reference labels and export per-scan metrics.
"""

import argparse
import json
import os
import os.path as osp
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
from scipy.ndimage import generate_binary_structure
from scipy.ndimage import label as label_connected_components
from tqdm import tqdm

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from scripts.segmentation_metrics_utils import compute_segmentation_scores, detect_lesions
from project_paths import DATA_ROOT, EXCEL_RECORDS_ROOT, RESULTS_ROOT


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate lung segmentation outputs against ground truth.")
    parser.add_argument("--model", default="smit", help="Name of the model")
    parser.add_argument("--subdataset", default="test_LRAD")
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()

    json_file_path = DATA_ROOT / "Trainval_Set_full.json"
    with json_file_path.open("r") as json_file:
        data = json.load(json_file)

    workingset = data[args.subdataset]
    score_records = []

    for sample in tqdm(workingset):
        lesion_detection_stats = {"TP": 0, "FP": 0, "FN": 0}
        file_gt = sample["label"]
        file_gt_full = DATA_ROOT / file_gt

        if args.subdataset == "test_tcia5rater":
            instance_name = file_gt.split("/")[3].replace("label.nii", ".nii.gz")
        else:
            instance_name = os.path.basename(file_gt).replace("_label", "")

        pred_dir = RESULTS_ROOT / f"{args.model}_main" / f"lung_{args.subdataset}_srcnorm"
        file_pred = pred_dir / "nii" / instance_name

        file_gt_nii_img = nib.load(str(file_gt_full))
        vox_spacing = file_gt_nii_img.header.get_zooms()
        gt_volume = file_gt_nii_img.get_fdata()

        file_pred_nii_img = nib.load(str(file_pred))
        pred_volume = file_pred_nii_img.get_fdata()

        pred_mask_lesion, num_predicted = label_connected_components(
            pred_volume > 0,
            structure=generate_binary_structure(3, 3),
            output=np.int16,
        )

        component_counts = np.bincount(pred_mask_lesion.ravel())
        remove_mask = component_counts < 1000
        remove_mask[0] = False
        pred_mask_lesion[remove_mask[pred_mask_lesion]] = 0

        pred_mask_lesion, num_predicted = label_connected_components(
            pred_mask_lesion > 0,
            structure=generate_binary_structure(3, 3),
            output=np.int16,
        )
        true_mask_lesion, num_reference = label_connected_components(
            gt_volume == 1,
            structure=generate_binary_structure(3, 3),
            output=np.int16,
        )

        detected_mask_lesion, mod_ref_mask, num_detected = detect_lesions(
            prediction_mask=pred_mask_lesion,
            reference_mask=true_mask_lesion,
            min_overlap=args.threshold,
        )

        lesion_detection_stats["TP"] += num_detected
        lesion_detection_stats["FP"] += num_predicted - num_detected
        lesion_detection_stats["FN"] += num_reference - num_detected

        tp = lesion_detection_stats["TP"]
        fp = lesion_detection_stats["FP"]
        fn = lesion_detection_stats["FN"]
        precision = float(tp) / (tp + fp) if tp + fp else 0
        recall = float(tp) / (tp + fn) if tp + fn else 0

        if num_detected > 0:
            lesion_scores = compute_segmentation_scores(
                prediction_mask=detected_mask_lesion,
                reference_mask=mod_ref_mask,
                voxel_spacing=vox_spacing,
            )
        else:
            lesion_scores = {"dice": [np.nan], "hd95": [np.nan]}

        score_records.append(
            {
                "name": instance_name,
                "dice": np.nanmean(lesion_scores["dice"]),
                "hd95": np.nanmean(lesion_scores["hd95"]),
                "precision": precision,
                "recall": recall,
            }
        )

    EXCEL_RECORDS_ROOT.mkdir(parents=True, exist_ok=True)
    output_path = EXCEL_RECORDS_ROOT / f"lung_{args.model}_{args.subdataset}_{args.threshold}.csv"
    pd.DataFrame(score_records).to_csv(output_path, index=False)


if __name__ == "__main__":
    main()
