"""
Metadata-stratified internal holdout experiments for RF-Deep.

This script tests whether acquisition factors inside the NSCLC/LRAD ID cohort
are spuriously flagged as OOD. It reuses the same feature vectors and RF-Deep
pipeline used in ood_rfdeep.py, but replaces the random ID test split with a
metadata-defined held-out ID subgroup.

Supported evaluation modes:
  - dataset_specific: one classifier per OOD dataset
  - ensemble: aggregate per-dataset classifiers (average)
  - lodo: leave-one-OOD-dataset-out
"""

from __future__ import annotations

import argparse
import json
import os.path as osp
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.model_selection import train_test_split

from ood_utils import (
    aggregate_predictions_by_filename,
    build_filename_to_dataset_mapping,
    compute_95ci_percentile,
    compute_metrics,
    compute_per_dataset_metrics,
    extract_expanded_features,
    filter_nan_files,
    load_feature_vectors,
    prepare_datasets,
    train_model,
)
from project_paths import ANALYSIS_RESULTS_ROOT, METADATA_ROOT, encode_manifest_path, ensure_output_dirs


STANDARD_OOD_DATASETS = ["rsna", "covid19", "kits23", "pancreas"]
METHOD_ALIASES = {
    "dataset_specific": "dataset_specific",
    "separate": "dataset_specific",
    "ensemble": "ensemble",
    "lodo": "lodo",
}
FACTOR_LABELS = {
    "manufacturer": "Manufacturer",
    "contrast": "Contrast",
    "kernel": "Kernel",
}


def _normalize_case_id(filename: str) -> str:
    return osp.basename(filename).replace(".nii.gz", "")


def _clean_kernel_label(value: str) -> str:
    token = str(value).strip().upper().replace("_", " ")
    if token in {"", "NAN", "NONE", "UNKNOWN"}:
        return "Unknown"
    token = token.replace("RECON ", "Recon")
    token = token.replace("RECON", "Recon")
    return token


def load_holdout_metadata(metadata_path: str | Path) -> pd.DataFrame:
    """Load the LRAD metadata sheet and standardize the grouping columns."""
    meta = pd.read_excel(metadata_path).copy()
    meta = meta.rename(
        columns={"ID": "CaseID", "Contras/orNo": "ContrastRaw", "Manufacture": "ManufacturerRaw"}
    )
    meta["CaseID"] = meta["CaseID"].astype(str).str.strip()

    meta["Manufacturer"] = (
        meta["ManufacturerRaw"]
        .astype(str)
        .str.upper()
        .str.strip()
        .replace({"NAN": "Unknown", "NONE": "Unknown", "": "Unknown"})
    )
    meta["Contrast"] = (
        meta["ContrastRaw"]
        .astype(str)
        .str.upper()
        .str.strip()
        .map({"CONTRAST": "Contrast", "NONCONTRAST": "Non-Contrast"})
        .fillna("Unknown")
    )

    if "recon_kernel" in meta.columns:
        meta["Kernel"] = meta["recon_kernel"].map(_clean_kernel_label)
    elif "Kernel" in meta.columns:
        meta["Kernel"] = meta["Kernel"].map(_clean_kernel_label)
    else:
        meta["Kernel"] = "Unknown"

    return meta[["CaseID", "Manufacturer", "Contrast", "Kernel"]].drop_duplicates()


def build_id_metadata_table(
    src_data: dict[str, np.ndarray],
    metadata_df: pd.DataFrame,
) -> tuple[pd.DataFrame, list[str]]:
    """Join LRAD feature-vector keys to metadata rows."""
    rows = []
    missing = []

    metadata_by_case = metadata_df.set_index("CaseID")
    for filename in src_data.keys():
        case_id = _normalize_case_id(filename)
        if case_id not in metadata_by_case.index:
            missing.append(case_id)
            continue
        row = metadata_by_case.loc[case_id]
        rows.append(
            {
                "filename": osp.basename(filename),
                "CaseID": case_id,
                "Manufacturer": row["Manufacturer"],
                "Contrast": row["Contrast"],
                "Kernel": row["Kernel"],
            }
        )

    return pd.DataFrame(rows), sorted(set(missing))


def build_group_specs(
    id_metadata: pd.DataFrame,
    factor: str,
    min_group_size: int,
) -> list[dict[str, object]]:
    """Create metadata holdout groups with adequate support."""
    column = FACTOR_LABELS[factor]
    usable = id_metadata[~id_metadata[column].isin(["Unknown", "", None])].copy()
    counts = usable[column].value_counts()

    specs = []
    for group_name, count in counts.items():
        if count < min_group_size:
            continue
        held_out = sorted(usable.loc[usable[column] == group_name, "filename"].tolist())
        train_pool = sorted(usable.loc[usable[column] != group_name, "filename"].tolist())
        specs.append(
            {
                "factor": factor,
                "column": column,
                "group_name": str(group_name),
                "held_out_filenames": held_out,
                "train_pool_filenames": train_pool,
                "held_out_count": len(held_out),
                "train_pool_count": len(train_pool),
            }
        )
    return specs


def split_ood_datasets(
    ood_data_dict: dict[str, dict[str, np.ndarray]],
    seed: int,
    train_fraction: float,
) -> dict[str, dict[str, list[str]]]:
    """Create train/test splits for standard OOD datasets."""
    splits = {}
    for ds_name, ds_data in ood_data_dict.items():
        filenames = list(ds_data.keys())
        ds_train, ds_test = train_test_split(
            filenames,
            train_size=train_fraction,
            random_state=seed,
        )
        splits[ds_name] = {"train": list(ds_train), "test": list(ds_test)}
    return splits


def safe_sample(filenames: list[str], requested_size: int, seed: int | None = None) -> np.ndarray:
    """Sample up to requested_size filenames without replacement."""
    available = len(filenames)
    actual_size = min(requested_size, available)
    if seed is not None:
        np.random.seed(seed)
    if actual_size == available:
        return np.array(filenames)
    return np.random.choice(filenames, actual_size, replace=False)


def summarize_series(values: list[float]) -> dict[str, object]:
    mean, std, ci_lower, ci_upper = compute_95ci_percentile(values)
    return {
        "mean": float(mean),
        "std": float(std),
        "ci_lower": float(ci_lower),
        "ci_upper": float(ci_upper),
        "all_values": list(values),
    }


def build_filename_mapping_for_holdout(
    heldout_id_filenames: list[str],
    ood_data_dict: dict[str, dict[str, np.ndarray]],
) -> dict[str, str]:
    filename_to_dataset = {osp.basename(fname): "heldout_id" for fname in heldout_id_filenames}
    filename_to_dataset.update(build_filename_to_dataset_mapping({}, ood_data_dict, id_name="heldout_id"))
    return filename_to_dataset


def false_ood_rate(df_avg_probs: pd.DataFrame, threshold: float) -> tuple[float, int, int]:
    heldout_df = df_avg_probs[df_avg_probs["dataset"] == "heldout_id"].copy()
    if heldout_df.empty:
        return np.nan, 0, 0
    flagged = int((heldout_df["probability"] >= threshold).sum())
    total = int(len(heldout_df))
    return flagged / total, flagged, total


def evaluate_binary_problem(
    model,
    heldout_id_data: dict[str, np.ndarray],
    heldout_id_filenames: list[str],
    ood_test_data: dict[str, np.ndarray],
    filename_to_dataset: dict[str, str],
    threshold: float,
) -> tuple[pd.DataFrame, float, float, float, int, int]:
    """Evaluate one model on held-out ID plus one or more OOD datasets."""
    filenames_id, X_id = extract_expanded_features(
        heldout_id_data, heldout_id_filenames, mode="test"
    )

    ood_test_filenames = []
    X_ood_parts = []
    for ds_name, ds_data in ood_test_data.items():
        ds_filenames = list(ds_data.keys())
        fnames_ds, X_ds = extract_expanded_features(ds_data, ds_filenames, mode="test")
        ood_test_filenames.extend(fnames_ds)
        X_ood_parts.append(X_ds)

    X_test = np.vstack([X_id, *X_ood_parts])
    y_test = np.hstack([np.zeros(len(X_id)), np.ones(sum(len(x) for x in X_ood_parts))])
    filenames_test = filenames_id + ood_test_filenames

    y_probs = model.predict_proba(X_test)[:, 1]
    df_avg = aggregate_predictions_by_filename(y_probs, y_test, filenames_test, filename_to_dataset)
    auroc, fpr95 = compute_metrics(df_avg["label"].values, df_avg["probability"].values)
    forate, flagged, total = false_ood_rate(df_avg, threshold)
    return df_avg, auroc, fpr95, forate, flagged, total


def train_id_features(
    src_data: dict[str, np.ndarray],
    train_pool_filenames: list[str],
    train_size: int,
    seed: int,
) -> np.ndarray:
    sampled = safe_sample(train_pool_filenames, train_size, seed=seed)
    _, features = extract_expanded_features(src_data, sampled, mode="train", seed=seed)
    return features


def run_single_iteration_dataset_specific(
    run_idx: int,
    seed: int,
    src_data: dict[str, np.ndarray],
    heldout_id_filenames: list[str],
    train_pool_filenames: list[str],
    ood_data_dict: dict[str, dict[str, np.ndarray]],
    train_size: int,
    trainer: str,
    train_fraction: float,
    threshold: float,
) -> tuple[dict[str, dict[str, float]], list[dict[str, object]]]:
    splits = split_ood_datasets(ood_data_dict, seed, train_fraction)
    heldout_id_data = {fname: src_data[fname] for fname in heldout_id_filenames}
    src_train_feat = train_id_features(src_data, train_pool_filenames, train_size, seed)

    results = {}
    raw_rows: list[dict[str, object]] = []
    for ds_name, ds_data in ood_data_dict.items():
        ds_train_all = splits[ds_name]["train"]
        ds_test_all = splits[ds_name]["test"]
        ds_sample_seed = seed + hash(ds_name) % 1000
        ds_sampled = safe_sample(ds_train_all, train_size, seed=ds_sample_seed)
        _, ds_train_feat = extract_expanded_features(ds_data, ds_sampled, mode="train", seed=ds_sample_seed)

        X_train = np.vstack([src_train_feat, ds_train_feat])
        y_train = np.hstack([np.zeros(len(src_train_feat)), np.ones(len(ds_train_feat))])
        model = train_model(X_train, y_train, trainer=trainer, seed=seed)

        ood_test_data = {ds_name: {fname: ds_data[fname] for fname in ds_test_all}}
        filename_to_dataset = build_filename_mapping_for_holdout(heldout_id_filenames, ood_test_data)
        df_avg, auroc, fpr95, forate, flagged, total = evaluate_binary_problem(
            model,
            heldout_id_data,
            heldout_id_filenames,
            ood_test_data,
            filename_to_dataset,
            threshold,
        )
        raw_rows.extend(
            df_avg.assign(
                eval_dataset=ds_name,
                run_idx=run_idx,
                seed=seed,
            ).to_dict("records")
        )
        results[ds_name] = {
            "auroc": float(auroc),
            "fpr95": float(fpr95),
            "false_ood_rate": float(forate),
            "false_ood_flagged": int(flagged),
            "false_ood_total": int(total),
            "ood_test_count": int(len(ds_test_all)),
        }
    return results, raw_rows


def run_single_iteration_ensemble(
    run_idx: int,
    seed: int,
    src_data: dict[str, np.ndarray],
    heldout_id_filenames: list[str],
    train_pool_filenames: list[str],
    ood_data_dict: dict[str, dict[str, np.ndarray]],
    train_size: int,
    trainer: str,
    train_fraction: float,
    threshold: float,
    aggregation: str,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    splits = split_ood_datasets(ood_data_dict, seed, train_fraction)
    heldout_id_data = {fname: src_data[fname] for fname in heldout_id_filenames}
    src_train_feat = train_id_features(src_data, train_pool_filenames, train_size, seed)

    filenames_id, X_id = extract_expanded_features(heldout_id_data, heldout_id_filenames, mode="test")
    filenames_test = list(filenames_id)
    X_test_parts = [X_id]
    test_ood_dict = {}

    for ds_name, ds_data in ood_data_dict.items():
        ds_test_all = splits[ds_name]["test"]
        test_ood_dict[ds_name] = {fname: ds_data[fname] for fname in ds_test_all}
        fnames_ds, X_ds = extract_expanded_features(test_ood_dict[ds_name], ds_test_all, mode="test")
        filenames_test.extend(fnames_ds)
        X_test_parts.append(X_ds)

    X_test = np.vstack(X_test_parts)
    y_test = np.hstack([
        np.zeros(len(X_id)),
        *[np.ones(len(X_part)) for X_part in X_test_parts[1:]],
    ])

    all_predictions = []
    for ds_name, ds_data in ood_data_dict.items():
        ds_train_all = splits[ds_name]["train"]
        ds_sample_seed = seed + hash(ds_name) % 1000
        ds_sampled = safe_sample(ds_train_all, train_size, seed=ds_sample_seed)
        _, ds_train_feat = extract_expanded_features(ds_data, ds_sampled, mode="train", seed=ds_sample_seed)

        X_train = np.vstack([src_train_feat, ds_train_feat])
        y_train = np.hstack([np.zeros(len(src_train_feat)), np.ones(len(ds_train_feat))])
        model = train_model(X_train, y_train, trainer=trainer, seed=seed)
        all_predictions.append(model.predict_proba(X_test)[:, 1])

    all_predictions = np.array(all_predictions)
    if aggregation == "avg":
        y_probs = np.mean(all_predictions, axis=0)
    elif aggregation == "max":
        y_probs = np.max(all_predictions, axis=0)
    else:
        raise ValueError(f"Unsupported aggregation: {aggregation}")

    filename_to_dataset = build_filename_mapping_for_holdout(heldout_id_filenames, test_ood_dict)
    df_avg = aggregate_predictions_by_filename(y_probs, y_test, filenames_test, filename_to_dataset)
    auroc, fpr95 = compute_metrics(df_avg["label"].values, df_avg["probability"].values)
    per_dataset = compute_per_dataset_metrics(df_avg, id_dataset_name="heldout_id")
    forate, flagged, total = false_ood_rate(df_avg, threshold)

    raw_rows = df_avg.assign(
        eval_dataset="all_ood",
        run_idx=run_idx,
        seed=seed,
    ).to_dict("records")

    return {
        "global": {
            "auroc": float(auroc),
            "fpr95": float(fpr95),
            "false_ood_rate": float(forate),
            "false_ood_flagged": int(flagged),
            "false_ood_total": int(total),
        },
        "per_dataset": {
            ds_name: {
                "auroc": float(metrics["auroc"]),
                "fpr95": float(metrics["fpr95"]),
            }
            for ds_name, metrics in per_dataset.items()
        },
    }, raw_rows


def run_single_iteration_lodo(
    run_idx: int,
    seed: int,
    src_data: dict[str, np.ndarray],
    heldout_id_filenames: list[str],
    train_pool_filenames: list[str],
    ood_data_dict: dict[str, dict[str, np.ndarray]],
    train_size: int,
    trainer: str,
    train_fraction: float,
    threshold: float,
) -> tuple[dict[str, dict[str, float]], list[dict[str, object]]]:
    splits = split_ood_datasets(ood_data_dict, seed, train_fraction)
    heldout_id_data = {fname: src_data[fname] for fname in heldout_id_filenames}
    src_train_feat = train_id_features(src_data, train_pool_filenames, train_size, seed)

    results = {}
    raw_rows: list[dict[str, object]] = []
    for heldout_ds in ood_data_dict.keys():
        ood_train_features = []
        for ds_name, ds_data in ood_data_dict.items():
            if ds_name == heldout_ds:
                continue
            ds_train_all = splits[ds_name]["train"]
            ds_sample_seed = seed + hash(ds_name) % 1000
            ds_sampled = safe_sample(ds_train_all, train_size, seed=ds_sample_seed)
            _, ds_train_feat = extract_expanded_features(ds_data, ds_sampled, mode="train", seed=ds_sample_seed)
            ood_train_features.append(ds_train_feat)

        ood_train_combined = np.vstack(ood_train_features)
        X_train = np.vstack([src_train_feat, ood_train_combined])
        y_train = np.hstack([np.zeros(len(src_train_feat)), np.ones(len(ood_train_combined))])
        model = train_model(X_train, y_train, trainer=trainer, seed=seed)

        test_ds_data = ood_data_dict[heldout_ds]
        ds_test_all = splits[heldout_ds]["test"]
        ood_test_data = {heldout_ds: {fname: test_ds_data[fname] for fname in ds_test_all}}
        filename_to_dataset = build_filename_mapping_for_holdout(heldout_id_filenames, ood_test_data)
        df_avg, auroc, fpr95, forate, flagged, total = evaluate_binary_problem(
            model,
            heldout_id_data,
            heldout_id_filenames,
            ood_test_data,
            filename_to_dataset,
            threshold,
        )
        raw_rows.extend(
            df_avg.assign(
                eval_dataset=heldout_ds,
                run_idx=run_idx,
                seed=seed,
            ).to_dict("records")
        )
        results[heldout_ds] = {
            "auroc": float(auroc),
            "fpr95": float(fpr95),
            "false_ood_rate": float(forate),
            "false_ood_flagged": int(flagged),
            "false_ood_total": int(total),
            "ood_test_count": int(len(ds_test_all)),
        }
    return results, raw_rows


def summarize_dataset_results(results_list: list[dict[str, dict[str, float]]]) -> dict[str, dict[str, object]]:
    dataset_names = sorted({key for result in results_list for key in result.keys()})
    summary = {}
    for ds_name in dataset_names:
        aurocs = [result[ds_name]["auroc"] for result in results_list if ds_name in result]
        fpr95s = [result[ds_name]["fpr95"] for result in results_list if ds_name in result]
        false_rates = [result[ds_name]["false_ood_rate"] for result in results_list if ds_name in result]
        flagged = [result[ds_name]["false_ood_flagged"] for result in results_list if ds_name in result]
        totals = [result[ds_name]["false_ood_total"] for result in results_list if ds_name in result]
        ood_counts = [result[ds_name]["ood_test_count"] for result in results_list if ds_name in result]
        summary[ds_name] = {
            "auroc": summarize_series(aurocs),
            "fpr95": summarize_series(fpr95s),
            "false_ood_rate": summarize_series(false_rates),
            "false_ood_flagged_mean": float(np.mean(flagged)),
            "false_ood_total_mean": float(np.mean(totals)),
            "ood_test_count_mean": float(np.mean(ood_counts)),
        }
    return summary


def summarize_ensemble_results(results_list: list[dict[str, object]]) -> dict[str, object]:
    global_aurocs = [result["global"]["auroc"] for result in results_list]
    global_fpr95s = [result["global"]["fpr95"] for result in results_list]
    global_false_rates = [result["global"]["false_ood_rate"] for result in results_list]
    global_flagged = [result["global"]["false_ood_flagged"] for result in results_list]
    global_totals = [result["global"]["false_ood_total"] for result in results_list]

    per_dataset_list = [result["per_dataset"] for result in results_list]
    per_dataset = {}
    dataset_names = sorted({key for result in per_dataset_list for key in result.keys()})
    for ds_name in dataset_names:
        aurocs = [result[ds_name]["auroc"] for result in per_dataset_list if ds_name in result]
        fpr95s = [result[ds_name]["fpr95"] for result in per_dataset_list if ds_name in result]
        per_dataset[ds_name] = {
            "auroc": summarize_series(aurocs),
            "fpr95": summarize_series(fpr95s),
        }

    return {
        "global": {
            "auroc": summarize_series(global_aurocs),
            "fpr95": summarize_series(global_fpr95s),
            "false_ood_rate": summarize_series(global_false_rates),
            "false_ood_flagged_mean": float(np.mean(global_flagged)),
            "false_ood_total_mean": float(np.mean(global_totals)),
        },
        "per_dataset": per_dataset,
    }


def run_method_for_group(
    method: str,
    src_data: dict[str, np.ndarray],
    ood_data_dict: dict[str, dict[str, np.ndarray]],
    spec: dict[str, object],
    num_runs: int,
    base_seed: int,
    train_size: int,
    trainer: str,
    train_fraction: float,
    threshold: float,
    n_jobs: int,
    collect_raw: bool = False,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    heldout_id_filenames = spec["held_out_filenames"]
    train_pool_filenames = spec["train_pool_filenames"]

    if len(train_pool_filenames) == 0 or len(heldout_id_filenames) == 0:
        raise ValueError("Empty train/test ID split for metadata holdout.")

    if method == "dataset_specific":
        worker = lambda run_idx: run_single_iteration_dataset_specific(
            run_idx,
            base_seed + run_idx,
            src_data,
            heldout_id_filenames,
            train_pool_filenames,
            ood_data_dict,
            train_size,
            trainer,
            train_fraction,
            threshold,
        )
        parser = summarize_dataset_results
    elif method == "ensemble":
        worker = lambda run_idx: run_single_iteration_ensemble(
            run_idx,
            base_seed + run_idx,
            src_data,
            heldout_id_filenames,
            train_pool_filenames,
            ood_data_dict,
            train_size,
            trainer,
            train_fraction,
            threshold,
            "avg",
        )
        parser = summarize_ensemble_results
    elif method == "lodo":
        worker = lambda run_idx: run_single_iteration_lodo(
            run_idx,
            base_seed + run_idx,
            src_data,
            heldout_id_filenames,
            train_pool_filenames,
            ood_data_dict,
            train_size,
            trainer,
            train_fraction,
            threshold,
        )
        parser = summarize_dataset_results
    else:
        raise ValueError(f"Unsupported method: {method}")

    if n_jobs == 1:
        run_outputs = [worker(run_idx) for run_idx in range(num_runs)]
    else:
        run_outputs = Parallel(n_jobs=n_jobs, verbose=5)(
            delayed(worker)(run_idx) for run_idx in range(num_runs)
        )

    results_list = [result for result, _ in run_outputs]
    raw_rows: list[dict[str, object]] = []
    if collect_raw:
        for _, rows in run_outputs:
            raw_rows.extend(rows)

    return parser(results_list), raw_rows


def convert_to_serializable(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {key: convert_to_serializable(value) for key, value in obj.items()}
    if isinstance(obj, list):
        return [convert_to_serializable(value) for value in obj]
    return obj


def run_experiment(
    model_name: str,
    img_size: int,
    methods: list[str],
    factors: list[str],
    metadata_path: str | Path,
    num_runs: int,
    base_seed: int,
    train_size: int,
    trainer: str,
    train_fraction: float,
    threshold: float,
    min_group_size: int,
    remove_nan: bool,
    n_jobs: int,
    save_raw_probs: bool = False,
) -> tuple[dict[str, object], pd.DataFrame | None]:
    print(f"{'=' * 80}")
    print("METADATA HOLDOUT OOD EXPERIMENT")
    print(f"{'=' * 80}")
    print(f"Model: {model_name}, Image size: {img_size}")
    print(f"Methods: {methods}")
    print(f"Factors: {factors}")
    print(f"Runs: {num_runs}, Base seed: {base_seed}")
    print(f"Train size per class: {train_size}")
    print(f"OOD train fraction: {train_fraction}")
    print(f"False-OOD threshold: {threshold}")
    print(f"Minimum held-out subgroup size: {min_group_size}")
    print(f"NaN handling: {'REMOVED' if remove_nan else 'KEPT'}")
    print(f"{'=' * 80}\n")

    nan_info = prepare_datasets(model_name, remove_nan=remove_nan)
    feature_vectors = load_feature_vectors(model_name, img_size)

    src_data = feature_vectors["lrad"]
    if remove_nan:
        src_data = filter_nan_files(src_data, nan_info["nan_files_lrad"])

    ood_data_dict = {}
    for ds_name in STANDARD_OOD_DATASETS:
        ds_data = feature_vectors[ds_name]
        if remove_nan:
            ds_data = filter_nan_files(ds_data, nan_info["nan_files_ood"])
        ood_data_dict[ds_name] = ds_data

    metadata_df = load_holdout_metadata(metadata_path)
    id_metadata, missing_cases = build_id_metadata_table(src_data, metadata_df)
    if missing_cases:
        print(f"Metadata missing for {len(missing_cases)} LRAD scans: {missing_cases[:10]}")

    results = {
        "config": {
            "model_name": model_name,
            "img_size": img_size,
            "methods": methods,
            "factors": factors,
            "metadata_path": encode_manifest_path(Path(metadata_path)),
            "num_runs": num_runs,
            "base_seed": base_seed,
            "train_size": train_size,
            "trainer": trainer,
            "train_fraction": train_fraction,
            "threshold": threshold,
            "min_group_size": min_group_size,
            "remove_nan": remove_nan,
            "n_jobs": n_jobs,
        },
        "metadata_coverage": {
            "usable_id_scans": int(len(id_metadata)),
            "missing_metadata_count": int(len(missing_cases)),
        },
        "factors": {},
    }
    all_raw_rows: list[dict[str, object]] = []

    for factor in factors:
        specs = build_group_specs(id_metadata, factor, min_group_size)
        factor_results = {
            "groups": {},
            "group_counts": {
                spec["group_name"]: int(spec["held_out_count"]) for spec in specs
            },
        }
        print(f"\nFactor: {factor}")
        for spec in specs:
            group_name = spec["group_name"]
            print(
                f"  Held-out {group_name}: {spec['held_out_count']} test scans, "
                f"{spec['train_pool_count']} train-pool scans"
            )
            group_result = {
                "held_out_count": int(spec["held_out_count"]),
                "train_pool_count": int(spec["train_pool_count"]),
                "methods": {},
            }
            for method in methods:
                print(f"    Running {method}...")
                method_result, method_raw_rows = run_method_for_group(
                    method,
                    src_data,
                    ood_data_dict,
                    spec,
                    num_runs,
                    base_seed,
                    train_size,
                    trainer,
                    train_fraction,
                    threshold,
                    n_jobs,
                    collect_raw=save_raw_probs,
                )
                group_result["methods"][method] = method_result
                if save_raw_probs:
                    for row in method_raw_rows:
                        row.update(
                            {
                                "factor": factor,
                                "group_name": group_name,
                                "method": method,
                                "held_out_count": int(spec["held_out_count"]),
                                "train_pool_count": int(spec["train_pool_count"]),
                            }
                        )
                    all_raw_rows.extend(method_raw_rows)
            factor_results["groups"][group_name] = group_result
        results["factors"][factor] = factor_results

    raw_df = pd.DataFrame(all_raw_rows) if save_raw_probs and all_raw_rows else None
    return results, raw_df


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run metadata-based internal ID holdout experiments for RF-Deep."
    )
    parser.add_argument(
        "--model-name",
        default="smit",
        help="Model name used in feature-vector pickles and radiomics CSVs.",
    )
    parser.add_argument(
        "--img-size",
        type=int,
        default=128,
        help="Image size key used in feature-vector pickles.",
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        default=["dataset_specific", "ensemble", "lodo"],
        choices=sorted(METHOD_ALIASES.keys()),
        help="Evaluation methods to run.",
    )
    parser.add_argument(
        "--factors",
        nargs="+",
        default=["manufacturer", "contrast", "kernel"],
        choices=["manufacturer", "contrast", "kernel"],
        help="Metadata factors to evaluate.",
    )
    parser.add_argument(
        "--metadata-path",
        default=str(METADATA_ROOT / "scanner_meta_info_LRad.xlsx"),
        help="Path to the LRAD metadata Excel file.",
    )
    parser.add_argument("--num-runs", type=int, default=100)
    parser.add_argument("--base-seed", type=int, default=2109)
    parser.add_argument("--train-size", type=int, default=20)
    parser.add_argument("--trainer", default="random_forest", choices=["random_forest", "simple_mlp"])
    parser.add_argument(
        "--train-fraction",
        type=float,
        default=0.3,
        help="Fraction of each OOD dataset reserved for training.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Probability threshold used to count false OOD calls on held-out ID.",
    )
    parser.add_argument(
        "--min-group-size",
        type=int,
        default=20,
        help="Minimum held-out subgroup size required to run an experiment.",
    )
    parser.add_argument("--keep-nan", action="store_true", help="Keep NaN-tagged scans instead of filtering them.")
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument(
        "--save-raw-probs",
        action="store_true",
        help="Also save raw per-scan probabilities for threshold sensitivity analysis.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(ANALYSIS_RESULTS_ROOT / "metadata_holdout"),
        help="Directory for the JSON results file.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    methods = [METHOD_ALIASES[method] for method in args.methods]
    results, raw_df = run_experiment(
        model_name=args.model_name,
        img_size=args.img_size,
        methods=methods,
        factors=args.factors,
        metadata_path=args.metadata_path,
        num_runs=args.num_runs,
        base_seed=args.base_seed,
        train_size=args.train_size,
        trainer=args.trainer,
        train_fraction=args.train_fraction,
        threshold=args.threshold,
        min_group_size=args.min_group_size,
        remove_nan=not args.keep_nan,
        n_jobs=args.n_jobs,
        save_raw_probs=args.save_raw_probs,
    )

    ensure_output_dirs()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    method_tag = "-".join(methods)
    factor_tag = "-".join(args.factors)
    output_path = output_dir / (
        f"{args.model_name}_size{args.img_size}_{method_tag}_{factor_tag}_"
        f"runs{args.num_runs}_seed{args.base_seed}.json"
    )
    output_path.write_text(json.dumps(convert_to_serializable(results), indent=2))
    print(f"\nSaved results to: {output_path}")

    if args.save_raw_probs and raw_df is not None:
        raw_output_path = output_path.with_suffix(".raw_probs.csv")
        raw_df.to_csv(raw_output_path, index=False)
        print(f"Saved raw probabilities to: {raw_output_path}")


if __name__ == "__main__":
    main()
