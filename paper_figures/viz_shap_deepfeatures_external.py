'''
SHAP visualizations for external test-only OOD datasets.

This utility is intentionally separate from ood_rfdeep.py. It trains RF-Deep
style classifiers on the standard OOD datasets, then visualizes SHAP patterns
for completely external held-out datasets such as breastc and covid19a.
'''

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import shap
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ood_utils import (
    extract_expanded_features,
    filter_nan_files,
    load_feature_vectors,
    prepare_datasets,
    train_model,
)
from project_paths import FIGURES_ROOT


STANDARD_OOD_DATASETS = ["rsna", "covid19", "kits23", "pancreas"]


def safe_sample(filenames: list[str], requested_size: int, seed: int | None = None) -> np.ndarray:
    """Sample up to requested_size filenames without replacement."""
    actual_size = min(requested_size, len(filenames))
    if seed is not None:
        np.random.seed(seed)
    if actual_size == len(filenames):
        return np.array(filenames)
    return np.random.choice(filenames, actual_size, replace=False)


def prepare_data(model_name: str, img_size: int, remove_nan: bool):
    nan_info = prepare_datasets(model_name, remove_nan=remove_nan)
    feature_vectors = load_feature_vectors(model_name=model_name, img_size=img_size)

    src_data = feature_vectors["lrad"]
    if remove_nan:
        src_data = filter_nan_files(src_data, nan_info["nan_files_lrad"])

    ood_data_dict = {}
    for ds_name in STANDARD_OOD_DATASETS:
        ds_data = feature_vectors[ds_name]
        if remove_nan:
            ds_data = filter_nan_files(ds_data, nan_info["nan_files_ood"])
        ood_data_dict[ds_name] = ds_data

    return src_data, ood_data_dict, feature_vectors


def create_id_split(src_data: dict[str, np.ndarray], seed: int, train_fraction: float):
    src_train, src_test = train_test_split(
        list(src_data.keys()),
        train_size=train_fraction,
        random_state=seed,
    )
    return list(src_train), list(src_test)


def train_dataset_specific_models(
    src_data: dict[str, np.ndarray],
    ood_data_dict: dict[str, dict[str, np.ndarray]],
    src_train_all: list[str],
    seed: int,
    train_fraction: float,
    train_size: int,
    trainer: str,
):
    models = {}
    _, src_train_feat = extract_expanded_features(
        src_data,
        safe_sample(src_train_all, train_size, seed=seed),
        mode="train",
        seed=seed,
    )

    for ds_name, ds_data in ood_data_dict.items():
        ds_train_all, _ = train_test_split(
            list(ds_data.keys()),
            train_size=train_fraction,
            random_state=seed,
        )
        ds_seed = seed + hash(ds_name) % 1000
        _, ds_train_feat = extract_expanded_features(
            ds_data,
            safe_sample(list(ds_train_all), train_size, seed=ds_seed),
            mode="train",
            seed=ds_seed,
        )
        X_train = np.vstack([src_train_feat, ds_train_feat])
        y_train = np.hstack([np.zeros(len(src_train_feat)), np.ones(len(ds_train_feat))])
        models[ds_name] = {
            "model": train_model(X_train, y_train, trainer=trainer, seed=seed),
            "X_train": X_train,
        }
    return models


def build_external_test_matrix(
    src_data: dict[str, np.ndarray],
    src_test_all: list[str],
    external_data: dict[str, np.ndarray],
):
    _, src_test_feat = extract_expanded_features(src_data, src_test_all, mode="test")
    external_filenames = list(external_data.keys())
    _, external_test_feat = extract_expanded_features(external_data, external_filenames, mode="test")
    X_test = np.vstack([src_test_feat, external_test_feat])
    feature_names = [f"Ft {i}" for i in range(X_test.shape[1])]
    return X_test, feature_names


def plot_dataset_specific_shap(
    model_bundle: dict[str, object],
    X_test: np.ndarray,
    feature_names: list[str],
    output_path: Path,
):
    explainer = shap.TreeExplainer(
        model_bundle["model"],
        model_bundle["X_train"],
        model_output="probability",
    )
    shap_values = explainer(X_test)
    plt.figure()
    shap.summary_plot(
        shap_values[:, :, 1],
        X_test,
        feature_names=feature_names,
        plot_size=(5, 7),
        show=False,
    )
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_ensemble_shap(
    model_bundles: dict[str, dict[str, object]],
    X_test: np.ndarray,
    feature_names: list[str],
    output_path: Path,
    aggregation: str,
):
    shap_arrays = []
    for bundle in model_bundles.values():
        explainer = shap.TreeExplainer(
            bundle["model"],
            bundle["X_train"],
            model_output="probability",
        )
        shap_arrays.append(explainer(X_test).values[:, :, 1])

    shap_stack = np.stack(shap_arrays, axis=0)
    if aggregation == "avg":
        shap_values = shap_stack.mean(axis=0)
    elif aggregation == "max":
        shap_values = shap_stack.max(axis=0)
    else:
        raise ValueError(f"Unsupported aggregation: {aggregation}")

    plt.figure()
    shap.summary_plot(
        shap_values,
        X_test,
        feature_names=feature_names,
        plot_size=(5, 7),
        show=False,
    )
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def run_external_shap(
    model_name: str,
    img_size: int,
    external_datasets: list[str],
    include_standard_ood: bool,
    mode: str,
    train_size: int,
    seed: int,
    trainer: str,
    train_fraction: float,
    remove_nan: bool,
    output_dir: str | Path,
):
    src_data, ood_data_dict, feature_vectors = prepare_data(model_name, img_size, remove_nan)
    src_train_all, src_test_all = create_id_split(src_data, seed, train_fraction)
    model_bundles = train_dataset_specific_models(
        src_data,
        ood_data_dict,
        src_train_all,
        seed,
        train_fraction,
        train_size,
        trainer,
    )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    datasets_to_plot = list(external_datasets)
    if include_standard_ood:
        for ds_name in STANDARD_OOD_DATASETS:
            if ds_name not in datasets_to_plot:
                datasets_to_plot.append(ds_name)

    for external_name in datasets_to_plot:
        if external_name not in feature_vectors:
            raise KeyError(
                f"External dataset '{external_name}' not found in feature vectors. "
                f"Available datasets: {list(feature_vectors.keys())}"
            )
        external_data = feature_vectors[external_name]
        X_test, feature_names = build_external_test_matrix(src_data, src_test_all, external_data)

        if mode == "dataset_specific":
            for standard_name, model_bundle in model_bundles.items():
                output_path = output_dir / (
                    f"deepfeatures_shap_external_{model_name}_{external_name}_via_{standard_name}.pdf"
                )
                print(f"Saving {output_path.name}")
                plot_dataset_specific_shap(model_bundle, X_test, feature_names, output_path)
        elif mode == "ensemble":
            output_path = output_dir / (
                f"deepfeatures_shap_external_{model_name}_{external_name}_{mode}.pdf"
            )
            print(f"Saving {output_path.name}")
            plot_ensemble_shap(model_bundles, X_test, feature_names, output_path, "avg")
        else:
            raise ValueError(f"Unsupported mode: {mode}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Create SHAP plots for external held-out deep-feature OOD datasets."
    )
    parser.add_argument("--model-name", default="smit")
    parser.add_argument("--img-size", type=int, default=128)
    parser.add_argument(
        "--external-datasets",
        nargs="+",
        default=["breastc", "covid19a"],
        help="External test-only datasets to visualize.",
    )
    parser.add_argument(
        "--include-standard-ood",
        action="store_true",
        help="Also generate SHAP plots for rsna, covid19, kits23, and pancreas.",
    )
    parser.add_argument(
        "--mode",
        choices=["dataset_specific", "ensemble"],
        default="ensemble",
        help="How to generate SHAP plots.",
    )
    parser.add_argument("--train-size", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--trainer", choices=["random_forest", "simple_mlp"], default="random_forest")
    parser.add_argument("--train-fraction", type=float, default=0.5)
    parser.add_argument("--keep-nan", action="store_true")
    parser.add_argument(
        "--output-dir",
        default=str(FIGURES_ROOT),
        help="Directory to save SHAP figures.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    run_external_shap(
        model_name=args.model_name,
        img_size=args.img_size,
        external_datasets=args.external_datasets,
        include_standard_ood=args.include_standard_ood,
        mode=args.mode,
        train_size=args.train_size,
        seed=args.seed,
        trainer=args.trainer,
        train_fraction=args.train_fraction,
        remove_nan=not args.keep_nan,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
