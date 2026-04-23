"""
Generate radiomics CSVs into the canonical radiomics_features directory.
Parallelised with ProcessPoolExecutor.
"""
from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial

from cerr import plan_container as pc
from cerr.radiomics import ibsi1
from tqdm import tqdm

from project_paths import (
    JSONS_ROOT,
    METADATA_ROOT,
    RADIOMICS_FEATURES_ROOT,
    load_manifest_entries,
)


# ── single-item worker (top-level so it's picklable) ────────────────────────
def _process_one(item: dict, settings_path: str) -> dict:
    """Load a single scan + segmentation and return its feature dict."""
    image_path = item["image"]
    seg_path = item["label"]

    plan_c = pc.loadNiiScan(image_path, "CT SCAN", direction="RAI")
    plan_c = pc.loadNiiStructure(seg_path, 0, plan_c, {1: "tumor"})

    row: dict = {"id": os.path.basename(image_path)}
    try:
        features, _ = ibsi1.computeScalarFeatures(0, 0, settings_path, plan_c)
        row.update(features)
    except ValueError as exc:
        print(f"[WARN] {row['id']}: {exc}")

    return row


# ── driver ───────────────────────────────────────────────────────────────────
def build_features(
    json_name: str,
    split: str,
    settings_path: str,
    max_workers: int | None = None,
) -> list[dict]:
    json_path = JSONS_ROOT / f"{json_name}.json"
    split_entries = load_manifest_entries(json_path, split)

    worker_fn = partial(_process_one, settings_path=settings_path)
    feature_rows: list[dict] = [{}] * len(split_entries)  # pre-allocate to preserve order

    with ProcessPoolExecutor(max_workers=max_workers) as pool:
        # map each future back to its original index so output order is stable
        future_to_idx = {
            pool.submit(worker_fn, item): idx
            for idx, item in enumerate(split_entries)
        }

        for future in tqdm(
            as_completed(future_to_idx),
            total=len(future_to_idx),
            desc="Radiomics",
        ):
            idx = future_to_idx[future]
            try:
                feature_rows[idx] = future.result()
            except Exception as exc:
                # Catch anything the child didn't handle (segfault wrapper, OOM, …)
                entry_id = os.path.basename(split_entries[idx]["image"])
                print(f"[ERROR] {entry_id}: {exc}")
                feature_rows[idx] = {"id": entry_id}

    return feature_rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute radiomics features from a dataset JSON split."
    )
    parser.add_argument(
        "--json-name",
        default="swinunetr_pancreas_src",
        help="JSON stem under jsons/.",
    )
    parser.add_argument(
        "--split",
        default="validation",
        help="Split key inside the JSON file.",
    )
    parser.add_argument(
        "--settings",
        default=str(METADATA_ROOT / "ibsi1.json"),
        help="IBSI settings file path.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Max parallel workers (default: CPU count).",
    )
    args = parser.parse_args()

    feature_rows = build_features(
        args.json_name, args.split, args.settings, max_workers=args.workers
    )

    RADIOMICS_FEATURES_ROOT.mkdir(parents=True, exist_ok=True)
    output_path = RADIOMICS_FEATURES_ROOT / f"{args.json_name}.csv"
    ibsi1.writeFeaturesToFile(feature_rows, str(output_path))
    print(f"Wrote radiomics CSV to {output_path}")


if __name__ == "__main__":
    main()
    
# """
# Generate radiomics CSVs into the canonical radiomics_features directory.
# """

# from __future__ import annotations

# import argparse
# import os

# from cerr import plan_container as pc
# from cerr.radiomics import ibsi1
# from tqdm import tqdm

# from project_paths import JSONS_ROOT, METADATA_ROOT, RADIOMICS_FEATURES_ROOT, load_manifest_entries


# def build_features(json_name: str, split: str, settings_path: str) -> list[dict]:
#     json_path = JSONS_ROOT / f"{json_name}.json"
#     split_entries = load_manifest_entries(json_path, split)

#     feature_rows: list[dict] = []
#     for item in tqdm(split_entries):
#         image_path = item["image"]
#         seg_path = item["label"]

#         plan_c = pc.loadNiiScan(image_path, "CT SCAN", direction="RAI")
#         plan_c = pc.loadNiiStructure(seg_path, 0, plan_c, {1: "tumor"})

#         row = {"id": os.path.basename(image_path)}
#         try:
#             features, _ = ibsi1.computeScalarFeatures(0, 0, settings_path, plan_c)
#             row.update(features)
#         except ValueError as exc:
#             print(exc)
#         feature_rows.append(row)

#     return feature_rows


# def main() -> None:
#     parser = argparse.ArgumentParser(description="Compute radiomics features from a dataset JSON split.")
#     parser.add_argument("--json-name", default="swinunetr_pancreas_src", help="JSON stem under jsons/.")
#     parser.add_argument("--split", default="validation", help="Split key inside the JSON file.")
#     parser.add_argument(
#         "--settings",
#         default=str(METADATA_ROOT / "ibsi1.json"),
#         help="IBSI settings file path.",
#     )
#     args = parser.parse_args()

#     feature_rows = build_features(args.json_name, args.split, args.settings)
#     RADIOMICS_FEATURES_ROOT.mkdir(parents=True, exist_ok=True)
#     output_path = RADIOMICS_FEATURES_ROOT / f"{args.json_name}.csv"
#     ibsi1.writeFeaturesToFile(feature_rows, str(output_path))
#     print(f"Wrote radiomics CSV to {output_path}")


# if __name__ == "__main__":
#     main()
