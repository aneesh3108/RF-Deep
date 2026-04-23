"""
Export scan-level RF-Deep probabilities and merge them with anchor summaries.
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import sys
import warnings
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from tqdm import tqdm

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from ood_rfdeep import create_splits_for_all_methods, safe_sample
from ood_utils import (
    aggregate_predictions_by_filename,
    build_filename_to_dataset_mapping,
    extract_expanded_features,
    filter_nan_files,
    load_feature_vectors,
    prepare_datasets,
    train_model,
)
from project_paths import ANALYSIS_RESULTS_ROOT


STANDARD_OOD_DATASETS = ["rsna", "covid19", "kits23", "pancreas"]
TEST_ONLY_DATASETS = ["breastc", "covid19a"]


def stable_dataset_seed(base_seed: int, dataset_name: str) -> int:
    """Stable per-dataset offset so runs are reproducible across Python sessions."""
    digest = hashlib.md5(dataset_name.encode("utf-8")).hexdigest()
    return base_seed + (int(digest[:8], 16) % 1000)


def prepare_feature_data(model_name: str, img_size: int, remove_nan: bool):
    nan_info = prepare_datasets(model_name, remove_nan=remove_nan)
    feature_vectors = load_feature_vectors(model_name, img_size)

    src_data = feature_vectors["lrad"]
    if remove_nan:
        src_data = filter_nan_files(src_data, nan_info["nan_files_lrad"])

    ood_data_dict = {}
    ood_filenames_dict = {}
    for ds_name in STANDARD_OOD_DATASETS:
        ds_data = feature_vectors[ds_name]
        if remove_nan:
            ds_data = filter_nan_files(ds_data, nan_info["nan_files_ood"])
        ood_data_dict[ds_name] = ds_data
        ood_filenames_dict[ds_name] = list(ds_data.keys())

    test_only_data_dict = {}
    for ds_name in TEST_ONLY_DATASETS:
        if ds_name in feature_vectors:
            test_only_data_dict[ds_name] = feature_vectors[ds_name]

    return src_data, ood_data_dict, ood_filenames_dict, test_only_data_dict


def annotate_df(
    df_avg: pd.DataFrame,
    method: str,
    run_idx: int,
    seed: int,
    eval_dataset: str,
    train_context: str,
    lodo_holdout_dataset: str | None = None,
) -> pd.DataFrame:
    df = df_avg.copy()
    df["method"] = method
    df["run_idx"] = run_idx
    df["seed"] = seed
    # Backward-compatible alias: dataset used for the binary evaluation slice.
    df["comparison_dataset"] = eval_dataset
    df["eval_dataset"] = eval_dataset
    df["train_context"] = train_context
    df["lodo_holdout_dataset"] = lodo_holdout_dataset
    return df.rename(columns={"filename": "filename_base"})


def dataset_specific_run(
    run_idx: int,
    base_seed: int,
    src_data,
    ood_data_dict,
    ood_filenames_dict,
    test_only_data_dict,
    train_size: int,
    trainer: str,
) -> pd.DataFrame:
    seed = base_seed + run_idx
    splits = create_splits_for_all_methods(
        list(src_data.keys()),
        ood_data_dict,
        ood_filenames_dict,
        seed,
    )
    src_train_all = splits["id"]["train"]
    src_test_all = splits["id"]["test"]

    src_train_sampled = safe_sample(src_train_all, train_size, seed=seed)
    _, src_train_feat = extract_expanded_features(
        src_data, src_train_sampled, mode="train", seed=seed
    )
    filenames_test_id, X_test_id = extract_expanded_features(
        src_data, src_test_all, mode="test"
    )

    frames = []
    trained_models = {}
    for ds_name, ds_data in ood_data_dict.items():
        ds_train_all = splits[ds_name]["train"]
        ds_test_all = splits[ds_name]["test"]
        ds_seed = stable_dataset_seed(seed, ds_name)
        ds_train_sampled = safe_sample(ds_train_all, train_size, seed=ds_seed)
        _, ds_train_feat = extract_expanded_features(
            ds_data, ds_train_sampled, mode="train", seed=ds_seed
        )
        filenames_test_ood, X_test_ood = extract_expanded_features(
            ds_data, ds_test_all, mode="test"
        )

        X_train = np.vstack([src_train_feat, ds_train_feat])
        y_train = np.hstack([np.zeros(len(src_train_feat)), np.ones(len(ds_train_feat))])
        model = train_model(X_train, y_train, trainer=trainer, seed=seed)
        trained_models[ds_name] = model

        X_test = np.vstack([X_test_id, X_test_ood])
        y_test = np.hstack([np.zeros(len(X_test_id)), np.ones(len(X_test_ood))])
        filenames_test = filenames_test_id + filenames_test_ood
        filename_to_dataset = build_filename_to_dataset_mapping(src_data, {ds_name: ds_data}, id_name="lrad")
        y_probs = model.predict_proba(X_test)[:, 1]
        df_avg = aggregate_predictions_by_filename(y_probs, y_test, filenames_test, filename_to_dataset)
        frames.append(
            annotate_df(
                df_avg, "dataset_specific", run_idx, seed,
                eval_dataset=ds_name, train_context=f"one_vs_{ds_name}",
            )
        )

    # Also score unseen test-only datasets with each per-dataset classifier.
    for train_ds, model in trained_models.items():
        for eval_ds, eval_data in test_only_data_dict.items():
            eval_filenames = list(eval_data.keys())
            filenames_test_ood, X_test_ood = extract_expanded_features(
                eval_data, eval_filenames, mode="test"
            )
            X_test = np.vstack([X_test_id, X_test_ood])
            y_test = np.hstack([np.zeros(len(X_test_id)), np.ones(len(X_test_ood))])
            filenames_test = filenames_test_id + filenames_test_ood
            y_probs = model.predict_proba(X_test)[:, 1]
            filename_to_dataset = build_filename_to_dataset_mapping(
                src_data, {eval_ds: eval_data}, id_name="lrad"
            )
            df_avg = aggregate_predictions_by_filename(y_probs, y_test, filenames_test, filename_to_dataset)
            frames.append(
                annotate_df(
                    df_avg, "dataset_specific", run_idx, seed,
                    eval_dataset=eval_ds, train_context=f"one_vs_{train_ds}",
                )
            )

    return pd.concat(frames, ignore_index=True)


def ensemble_run(
    run_idx: int,
    base_seed: int,
    src_data,
    ood_data_dict,
    ood_filenames_dict,
    test_only_data_dict,
    train_size: int,
    trainer: str,
) -> pd.DataFrame:
    seed = base_seed + run_idx
    splits = create_splits_for_all_methods(
        list(src_data.keys()),
        ood_data_dict,
        ood_filenames_dict,
        seed,
    )
    src_train_all = splits["id"]["train"]
    src_test_all = splits["id"]["test"]
    src_train_sampled = safe_sample(src_train_all, train_size, seed=seed)
    _, src_train_feat = extract_expanded_features(
        src_data, src_train_sampled, mode="train", seed=seed
    )
    filenames_test_id, X_test_id = extract_expanded_features(
        src_data, src_test_all, mode="test"
    )

    trained_models = {}
    for ds_name, ds_data in ood_data_dict.items():
        ds_train_all = splits[ds_name]["train"]
        ds_seed = stable_dataset_seed(seed, ds_name)
        ds_train_sampled = safe_sample(ds_train_all, train_size, seed=ds_seed)
        _, ds_train_feat = extract_expanded_features(
            ds_data, ds_train_sampled, mode="train", seed=ds_seed
        )
        X_train = np.vstack([src_train_feat, ds_train_feat])
        y_train = np.hstack([np.zeros(len(src_train_feat)), np.ones(len(ds_train_feat))])
        trained_models[ds_name] = train_model(X_train, y_train, trainer=trainer, seed=seed)

    all_eval_datasets = {**ood_data_dict, **test_only_data_dict}
    frames = []
    for ds_name, ds_data in all_eval_datasets.items():
        if ds_name in splits:
            eval_filenames = splits[ds_name]["test"]
        else:
            eval_filenames = list(ds_data.keys())
        filenames_test_ood, X_test_ood = extract_expanded_features(
            ds_data, eval_filenames, mode="test"
        )
        X_test = np.vstack([X_test_id, X_test_ood])
        y_test = np.hstack([np.zeros(len(X_test_id)), np.ones(len(X_test_ood))])
        filenames_test = filenames_test_id + filenames_test_ood
        all_preds = [model.predict_proba(X_test)[:, 1] for model in trained_models.values()]
        y_probs = np.mean(np.array(all_preds), axis=0)
        filename_to_dataset = build_filename_to_dataset_mapping(src_data, {ds_name: ds_data}, id_name="lrad")
        df_avg = aggregate_predictions_by_filename(y_probs, y_test, filenames_test, filename_to_dataset)
        frames.append(
            annotate_df(
                df_avg, "ensemble", run_idx, seed,
                eval_dataset=ds_name, train_context="ensemble",
            )
        )

    return pd.concat(frames, ignore_index=True)


def unified_run(
    run_idx: int,
    base_seed: int,
    src_data,
    ood_data_dict,
    ood_filenames_dict,
    test_only_data_dict,
    train_size: int,
    trainer: str,
) -> pd.DataFrame:
    seed = base_seed + run_idx
    splits = create_splits_for_all_methods(
        list(src_data.keys()),
        ood_data_dict,
        ood_filenames_dict,
        seed,
    )
    src_train_all = splits["id"]["train"]
    src_test_all = splits["id"]["test"]

    pooled_ood_train = []
    for ds_name in ood_data_dict.keys():
        pooled_ood_train.extend(splits[ds_name]["train"])

    pooled_ood_data = {}
    for ds_name, ds_data in ood_data_dict.items():
        pooled_ood_data.update(ds_data)

    src_train_sampled = safe_sample(src_train_all, train_size, seed=seed)
    ood_train_sampled = safe_sample(pooled_ood_train, train_size, seed=seed)
    _, src_train_feat = extract_expanded_features(
        src_data, src_train_sampled, mode="train", seed=seed
    )
    _, ood_train_feat = extract_expanded_features(
        pooled_ood_data, ood_train_sampled, mode="train", seed=seed
    )
    X_train = np.vstack([src_train_feat, ood_train_feat])
    y_train = np.hstack([np.zeros(len(src_train_feat)), np.ones(len(ood_train_feat))])
    model = train_model(X_train, y_train, trainer=trainer, seed=seed)

    filenames_test_id, X_test_id = extract_expanded_features(
        src_data, src_test_all, mode="test"
    )

    all_eval_datasets = {**ood_data_dict, **test_only_data_dict}
    frames = []
    for ds_name, ds_data in all_eval_datasets.items():
        if ds_name in splits:
            eval_filenames = splits[ds_name]["test"]
        else:
            eval_filenames = list(ds_data.keys())
        filenames_test_ood, X_test_ood = extract_expanded_features(
            ds_data, eval_filenames, mode="test"
        )
        X_test = np.vstack([X_test_id, X_test_ood])
        y_test = np.hstack([np.zeros(len(X_test_id)), np.ones(len(X_test_ood))])
        filenames_test = filenames_test_id + filenames_test_ood
        y_probs = model.predict_proba(X_test)[:, 1]
        filename_to_dataset = build_filename_to_dataset_mapping(src_data, {ds_name: ds_data}, id_name="lrad")
        df_avg = aggregate_predictions_by_filename(y_probs, y_test, filenames_test, filename_to_dataset)
        frames.append(
            annotate_df(
                df_avg, "unified", run_idx, seed,
                eval_dataset=ds_name, train_context="pooled_ood",
            )
        )

    return pd.concat(frames, ignore_index=True)


def lodo_run(
    run_idx: int,
    base_seed: int,
    src_data,
    ood_data_dict,
    ood_filenames_dict,
    test_only_data_dict,
    train_size: int,
    trainer: str,
) -> pd.DataFrame:
    seed = base_seed + run_idx
    splits = create_splits_for_all_methods(
        list(src_data.keys()),
        ood_data_dict,
        ood_filenames_dict,
        seed,
    )
    src_train_all = splits["id"]["train"]
    src_test_all = splits["id"]["test"]
    src_train_sampled = safe_sample(src_train_all, train_size, seed=seed)
    _, src_train_feat = extract_expanded_features(
        src_data, src_train_sampled, mode="train", seed=seed
    )
    filenames_test_id, X_test_id = extract_expanded_features(
        src_data, src_test_all, mode="test"
    )

    frames = []
    for heldout_ds in ood_data_dict.keys():
        ood_train_features = []
        for ds_name, ds_data in ood_data_dict.items():
            if ds_name == heldout_ds:
                continue
            ds_seed = stable_dataset_seed(seed, ds_name)
            ds_train_sampled = safe_sample(splits[ds_name]["train"], train_size, seed=ds_seed)
            _, ds_train_feat = extract_expanded_features(
                ds_data, ds_train_sampled, mode="train", seed=ds_seed
            )
            ood_train_features.append(ds_train_feat)
        ood_train_combined = np.vstack(ood_train_features)
        X_train = np.vstack([src_train_feat, ood_train_combined])
        y_train = np.hstack([
            np.zeros(len(src_train_feat)),
            np.ones(len(ood_train_combined)),
        ])
        model = train_model(X_train, y_train, trainer=trainer, seed=seed)

        eval_sets = {heldout_ds: {fname: ood_data_dict[heldout_ds][fname] for fname in splits[heldout_ds]["test"]}}
        for to_name, to_data in test_only_data_dict.items():
            eval_sets[f"{to_name}__via__{heldout_ds}"] = to_data

        for eval_name, eval_data in eval_sets.items():
            ds_label, _, _ = eval_name.partition("__via__")
            eval_filenames = list(eval_data.keys())
            filenames_test_ood, X_test_ood = extract_expanded_features(
                eval_data, eval_filenames, mode="test"
            )
            X_test = np.vstack([X_test_id, X_test_ood])
            y_test = np.hstack([np.zeros(len(X_test_id)), np.ones(len(X_test_ood))])
            filenames_test = filenames_test_id + filenames_test_ood
            y_probs = model.predict_proba(X_test)[:, 1]
            filename_to_dataset = build_filename_to_dataset_mapping(src_data, {ds_label: eval_data}, id_name="lrad")
            df_avg = aggregate_predictions_by_filename(y_probs, y_test, filenames_test, filename_to_dataset)
            frames.append(
                annotate_df(
                    df_avg, "lodo", run_idx, seed,
                    eval_dataset=ds_label,
                    train_context=f"lodo_holdout_{heldout_ds}",
                    lodo_holdout_dataset=heldout_ds,
                )
            )

    return pd.concat(frames, ignore_index=True)


def export_scores(
    methods: list[str],
    model_name: str,
    img_size: int,
    num_runs: int,
    base_seed: int,
    train_size: int,
    trainer: str,
    remove_nan: bool,
    n_jobs: int,
) -> pd.DataFrame:
    src_data, ood_data_dict, ood_filenames_dict, test_only_data_dict = prepare_feature_data(
        model_name, img_size, remove_nan
    )

    def run_all_methods_for_index(run_idx: int) -> pd.DataFrame:
        warnings.filterwarnings("once", message="Requested.*samples", category=UserWarning)

        frames = []
        if "unified" in methods:
            frames.append(
                unified_run(
                    run_idx, base_seed, src_data, ood_data_dict,
                    ood_filenames_dict, test_only_data_dict, train_size, trainer,
                )
            )
        if "dataset_specific" in methods:
            frames.append(
                dataset_specific_run(
                    run_idx, base_seed, src_data, ood_data_dict,
                    ood_filenames_dict, test_only_data_dict, train_size, trainer,
                )
            )
        if "ensemble" in methods:
            frames.append(
                ensemble_run(
                    run_idx, base_seed, src_data, ood_data_dict,
                    ood_filenames_dict, test_only_data_dict, train_size, trainer,
                )
            )
        if "lodo" in methods:
            frames.append(
                lodo_run(
                    run_idx, base_seed, src_data, ood_data_dict,
                    ood_filenames_dict, test_only_data_dict, train_size, trainer,
                )
            )
        return pd.concat(frames, ignore_index=True)

    if n_jobs == 1:
        results = [
            run_all_methods_for_index(run_idx)
            for run_idx in tqdm(range(num_runs), desc="RF-Deep score export", unit="run")
        ]
    else:
        results = Parallel(n_jobs=n_jobs, verbose=10)(
            delayed(run_all_methods_for_index)(run_idx) for run_idx in range(num_runs)
        )

    return pd.concat(results, ignore_index=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export scan-level RF-Deep probabilities and merge with anchor summary."
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        default=["ensemble"],
        choices=["unified", "dataset_specific", "ensemble", "lodo"],
    )
    parser.add_argument("--model-name", default="smit")
    parser.add_argument("--img-size", type=int, default=128)
    parser.add_argument("--num-runs", type=int, default=100)
    parser.add_argument("--base-seed", type=int, default=2109)
    parser.add_argument("--train-size", type=int, default=20)
    parser.add_argument("--trainer", default="random_forest", choices=["random_forest", "simple_mlp"])
    parser.add_argument("--keep-nan", action="store_true")
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=1,
        help="Parallel workers across runs. Use -1 for all cores.",
    )
    parser.add_argument(
        "--anchor-csv",
        default=str(ANALYSIS_RESULTS_ROOT / "anchor_summary" / "smit_anchor_summary.csv"),
        help="Anchor summary CSV to merge on dataset + filename.",
    )
    parser.add_argument(
        "--output-path",
        default=str(ANALYSIS_RESULTS_ROOT / "anchor_summary" / "smit_anchor_scores_merged.csv"),
        help="Merged CSV output path.",
    )
    return parser.parse_args()


def main() -> None:
    warnings.filterwarnings("module", message="Requested.*samples", category=UserWarning)
    args = parse_args()
    scores_df = export_scores(
        methods=args.methods,
        model_name=args.model_name,
        img_size=args.img_size,
        num_runs=args.num_runs,
        base_seed=args.base_seed,
        train_size=args.train_size,
        trainer=args.trainer,
        remove_nan=not args.keep_nan,
        n_jobs=args.n_jobs,
    )

    anchor_df = pd.read_csv(args.anchor_csv)
    duplicate_mask = anchor_df.duplicated(subset=["dataset", "filename"], keep=False)
    if duplicate_mask.any():
        duplicates = anchor_df.loc[duplicate_mask, ["dataset", "filename"]].drop_duplicates()
        sample_str = duplicates.head(5).to_string(index=False)
        raise ValueError(
            f"Anchor CSV has {len(duplicates)} duplicate (dataset, filename) pairs.\n{sample_str}"
        )
    merged = scores_df.merge(
        anchor_df,
        left_on=["dataset", "filename_base"],
        right_on=["dataset", "filename"],
        how="left",
        validate="many_to_one",
    )

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(output_path, index=False)
    print(f"Saved {len(merged)} rows to: {output_path}")


if __name__ == "__main__":
    main()
