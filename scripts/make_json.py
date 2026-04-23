"""
Build MONAI-style validation manifests from saved segmentation outputs.
"""

import argparse
import json
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from project_paths import (
    BREASTC_ROOT,
    COVID19A_ROOT,
    JSONS_ROOT,
    NSCLC_ROOT,
    RESULTS_ROOT,
    RSNA_ROOT,
    COVID19_ROOT,
    KITS23_ROOT,
    PANCREAS_ROOT,
    LRAD_R01_ROOT,
    LRAD_AMC_ROOT,
    encode_manifest_path,
)


def build_result_dir(model: str, dataset: str, dataset_type: str) -> Path:
    if dataset in {"rsna", "covid19", "covid19a", "kits23", "pancreas", "breastc"}:
        return RESULTS_ROOT / f"{model}_farood" / f"lung_test_{dataset}_{dataset_type}norm" / "nii"
    if dataset == "lrad":
        return RESULTS_ROOT / f"{model}_main" / f"lung_test_{dataset}_{dataset_type}norm" / "nii"
    return RESULTS_ROOT / f"{model}_main" / "lung_validation_srcnorm" / "nii"


def build_image_path(dataset: str, filename: str) -> Path:
    if dataset == "rsna":
        return RSNA_ROOT / "train_nii" / filename
    if dataset == "covid19":
        return COVID19_ROOT / filename
    if dataset == "covid19a":
        return COVID19A_ROOT / filename
    if dataset == "kits23":
        return KITS23_ROOT / "images" / filename
    if dataset == "pancreas":
        return PANCREAS_ROOT / filename
    if dataset == "breastc":
        return BREASTC_ROOT / "imgs" / filename
    if dataset == "validation":
        return NSCLC_ROOT / "image" / filename
    if dataset == "lrad":
        if "R01" in filename:
            return LRAD_R01_ROOT / "image" / filename
        if "AMC" in filename:
            return LRAD_AMC_ROOT / "image" / filename
        raise ValueError(f"Unable to infer LRAD source for {filename}")
    raise ValueError(f"Unsupported dataset: {dataset}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build MONAI-style validation JSONs from segmentation outputs.")
    parser.add_argument("--model", default="swinunetr_10k")
    parser.add_argument("--dataset", default="lrad")
    parser.add_argument("--dataset-type", default="src")
    args = parser.parse_args()

    result_dir = build_result_dir(args.model, args.dataset, args.dataset_type)
    files = sorted(file for file in result_dir.iterdir() if file.name != ".DS_Store")

    data = [
        {
            "image": encode_manifest_path(build_image_path(args.dataset, file.name)),
            "label": encode_manifest_path(file),
        }
        for file in files
    ]
    JSONS_ROOT.mkdir(parents=True, exist_ok=True)
    output_path = JSONS_ROOT / f"{args.model}_{args.dataset}_{args.dataset_type}.json"
    with output_path.open("w") as json_file:
        json.dump({"validation": data}, json_file, indent=4)


if __name__ == "__main__":
    main()
