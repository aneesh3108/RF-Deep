"""
Mahalanobis distance OOD detection (MD-Deep).

Fits mean + covariance on ID training features only, scores all test
samples by squared Mahalanobis distance.  Reports global and per-dataset
metrics.  No OOD exposure.

Optional normalization (fitted on ID, applied before everything):
  --normalize none      No normalization (default)
  --normalize zscore    Per-dimension z-score standardization

Optional feature transforms (applied after normalization, before scoring):
  --transform none    Plain Mahalanobis (default)
  --transform react   ReAct: clip features at ID percentile (Sun et al., NeurIPS 2021)
  --transform ash     ASH: zero out weak per-sample activations (Djurisic et al., ICLR 2023)

Usage:
  python ood_maha.py
  python ood_maha.py --normalize zscore
  python ood_maha.py --transform react --percentile 90 95 99
  python ood_maha.py --normalize zscore --transform none react ash
"""

import argparse
import json

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.covariance import EmpiricalCovariance, LedoitWolf
from joblib import Parallel, delayed

from project_paths import ANALYSIS_RESULTS_ROOT, ensure_output_dirs
from ood_utils import (
    load_feature_vectors,
    prepare_datasets,
    filter_nan_files,
    build_filename_to_dataset_mapping,
    compute_95ci_percentile,
    extract_expanded_features,
    aggregate_predictions_by_filename,
    compute_metrics,
    compute_per_dataset_metrics
)


# ============================================
# FEATURE TRANSFORMS
# ============================================

class NoNormalization:
    """Identity — no normalization."""
    def fit(self, X_id):
        pass
    def __call__(self, X):
        return X


class ZScoreNormalization:
    """Per-dimension z-score normalization fitted on ID data."""
    def __init__(self):
        self.mean_ = None
        self.std_ = None
    def fit(self, X_id):
        self.mean_ = np.mean(X_id, axis=0)
        self.std_ = np.std(X_id, axis=0)
        self.std_[self.std_ < 1e-10] = 1.0  # avoid division by zero
    def __call__(self, X):
        return (X - self.mean_) / self.std_


def make_normalizer(name):
    if name == "none":
        return NoNormalization()
    elif name == "zscore":
        return ZScoreNormalization()
    else:
        raise ValueError(f"Unknown normalizer: {name}")


class NoTransform:
    """Identity — plain Mahalanobis."""
    def fit(self, X_id):
        pass
    def __call__(self, X):
        return X


class ReActTransform:
    """Clip each feature dimension at the p-th percentile of ID data."""
    def __init__(self, percentile=95):
        self.percentile = percentile
        self.clip_val_ = None
    def fit(self, X_id):
        self.clip_val_ = np.percentile(X_id, self.percentile, axis=0)
    def __call__(self, X):
        return np.minimum(X, self.clip_val_)


class ASHTransform:
    """Zero out per-sample dimensions below the p-th percentile of |activation|."""
    def __init__(self, percentile=90):
        self.percentile = percentile
    def fit(self, X_id):
        pass  # ASH is per-sample, no ID fitting needed
    def __call__(self, X):
        thresholds = np.percentile(np.abs(X), self.percentile, axis=1, keepdims=True)
        mask = np.abs(X) >= thresholds
        return X * mask


def make_transform(name, percentile=90):
    if name == "none":
        return NoTransform()
    elif name == "react":
        return ReActTransform(percentile)
    elif name == "ash":
        return ASHTransform(percentile)
    else:
        raise ValueError(f"Unknown transform: {name}")


# ============================================
# MAHALANOBIS DETECTOR
# ============================================

class MahalanobisDetector:
    """
    OOD detector using squared Mahalanobis distance.
    Only requires ID training data.  Optionally applies a feature
    transform before fitting and scoring.
    """
    def __init__(self, covariance_type='ledoit_wolf', regularization=1e-5,
                 transform=None, normalizer=None):
        self.covariance_type = covariance_type
        self.regularization = regularization
        self.normalizer = normalizer or NoNormalization()
        self.transform = transform or NoTransform()
        self.mean_ = None
        self.precision_ = None

    def fit(self, X_train_id):
        # Normalize first, then apply transform
        self.normalizer.fit(X_train_id)
        X = self.normalizer(X_train_id)
        self.transform.fit(X)
        X = self.transform(X)

        self.mean_ = np.mean(X, axis=0)

        if self.covariance_type == 'empirical':
            cov_estimator = EmpiricalCovariance()
        elif self.covariance_type == 'ledoit_wolf':
            cov_estimator = LedoitWolf()
        else:
            raise ValueError(f"Unknown covariance type: {self.covariance_type}")

        cov_estimator.fit(X)
        cov = cov_estimator.covariance_
        cov += self.regularization * np.eye(cov.shape[0])

        try:
            self.precision_ = np.linalg.inv(cov)
        except np.linalg.LinAlgError:
            print("Warning: Covariance matrix is singular, using pseudo-inverse")
            self.precision_ = np.linalg.pinv(cov)

        return self

    def score_samples(self, X):
        """Squared Mahalanobis distance. Higher = more OOD."""
        if self.mean_ is None or self.precision_ is None:
            raise ValueError("Detector not fitted yet!")

        X = self.normalizer(X)
        X = self.transform(X)
        diff = X - self.mean_
        return np.sum(diff @ self.precision_ * diff, axis=1)


# ============================================
# SHARED UTILITIES
# ============================================

def create_splits(filenames, seed, train_fraction=0.3):
    """Create train/test split."""
    train, test = train_test_split(
        filenames,
        train_size=train_fraction,
        random_state=seed
    )
    return {'train': train, 'test': test}


def safe_sample(filenames, requested_size, seed=None):
    """Sample from filenames, using all if fewer than requested."""
    available = len(filenames)
    actual_size = min(requested_size, available)

    if seed is not None:
        np.random.seed(seed)

    if actual_size == available:
        return np.array(filenames)
    else:
        return np.random.choice(filenames, actual_size, replace=False)


def parse_unified_results(results_list, ood_dataset_names):
    """Parse results with global and per-dataset metrics."""
    global_aurocs = []
    global_fpr95s = []
    dataset_results = {ds: {'aurocs': [], 'fpr95s': []} for ds in ood_dataset_names}

    for auroc, fpr95, per_ds in results_list:
        global_aurocs.append(auroc)
        global_fpr95s.append(fpr95)

        for ds, metrics in per_ds.items():
            dataset_results[ds]['aurocs'].append(metrics['auroc'])
            dataset_results[ds]['fpr95s'].append(metrics['fpr95'])

    return format_results(global_aurocs, global_fpr95s, dataset_results, ood_dataset_names)


def format_results(global_aurocs, global_fpr95s, dataset_results, ood_dataset_names):
    """Format results dictionary with global and per-dataset metrics."""
    results = {}

    if global_aurocs is not None:
        auroc_mean, auroc_std, auroc_lower, auroc_upper = compute_95ci_percentile(global_aurocs)
        fpr95_mean, fpr95_std, fpr95_lower, fpr95_upper = compute_95ci_percentile(global_fpr95s)

        results['global'] = {
            'auroc_mean': auroc_mean,
            'auroc_std': auroc_std,
            'auroc_ci_lower': auroc_lower,
            'auroc_ci_upper': auroc_upper,
            'fpr95_mean': fpr95_mean,
            'fpr95_std': fpr95_std,
            'fpr95_ci_lower': fpr95_lower,
            'fpr95_ci_upper': fpr95_upper,
        }

    results['per_dataset'] = {}
    for ds in ood_dataset_names:
        if len(dataset_results[ds]['aurocs']) == 0:
            continue

        auroc_mean, auroc_std, auroc_lower, auroc_upper = compute_95ci_percentile(dataset_results[ds]['aurocs'])
        fpr95_mean, fpr95_std, fpr95_lower, fpr95_upper = compute_95ci_percentile(dataset_results[ds]['fpr95s'])

        results['per_dataset'][ds] = {
            'auroc_mean': auroc_mean,
            'auroc_std': auroc_std,
            'auroc_ci_lower': auroc_lower,
            'auroc_ci_upper': auroc_upper,
            'fpr95_mean': fpr95_mean,
            'fpr95_std': fpr95_std,
            'fpr95_ci_lower': fpr95_lower,
            'fpr95_ci_upper': fpr95_upper,
        }

    return results


# ============================================
# CORE EXPERIMENT LOGIC
# ============================================

def run_single_iteration(
    seed, src_data, ood_data_dict, train_size,
    covariance_type='ledoit_wolf', regularization=1e-5,
    transform_name='none', percentile=90,
    normalize='none',
    filename_to_dataset=None, id_dataset_name="lrad"
):
    """Single iteration: fit on ID, score ID + all OOD, return global + per-dataset."""
    src_filenames = list(src_data.keys())
    src_splits = create_splits(src_filenames, seed)

    src_train_sampled = safe_sample(src_splits['train'], train_size, seed=seed)
    _, src_train_feat = extract_expanded_features(
        src_data, src_train_sampled, mode="train", seed=seed
    )

    filenames_test_id, X_test_id = extract_expanded_features(
        src_data, src_splits['test'], mode="test"
    )

    normalizer = make_normalizer(normalize)
    transform = make_transform(transform_name, percentile)
    detector = MahalanobisDetector(
        covariance_type=covariance_type,
        regularization=regularization,
        transform=transform,
        normalizer=normalizer,
    )
    detector.fit(src_train_feat)

    filenames_test = filenames_test_id.copy()
    X_test_list = [X_test_id]

    for ds_data in ood_data_dict.values():
        ds_splits = create_splits(list(ds_data.keys()), seed)
        filenames_test_ood, X_test_ood = extract_expanded_features(
            ds_data, ds_splits['test'], mode="test"
        )
        filenames_test.extend(filenames_test_ood)
        X_test_list.append(X_test_ood)

    X_test = np.vstack(X_test_list)
    y_test = np.hstack([
        np.zeros(len(X_test_id)),
        *[np.ones(len(X_test_list[i])) for i in range(1, len(X_test_list))]
    ])

    scores = detector.score_samples(X_test)

    df_avg_probs = aggregate_predictions_by_filename(
        scores, y_test, filenames_test, filename_to_dataset
    )

    global_auroc, global_fpr95 = compute_metrics(
        df_avg_probs["label"].values, df_avg_probs["probability"].values
    )
    per_dataset_metrics = compute_per_dataset_metrics(df_avg_probs, id_dataset_name)

    return global_auroc, global_fpr95, per_dataset_metrics


# ============================================
# MAIN
# ============================================

DEFAULT_OOD_DATASETS = ['rsna', 'covid19', 'kits23', 'pancreas']
ALL_OOD_DATASETS = ['rsna', 'covid19', 'kits23', 'pancreas', 'breastc', 'covid19a']


def run_experiment(
    model_name='smit',
    img_size=128,
    num_runs=100,
    base_seed=3108,
    train_size=20,
    covariance_type='ledoit_wolf',
    regularization=1e-5,
    transform_name='none',
    percentile=90,
    normalize='none',
    ood_datasets=None,
    remove_nan=True,
    n_jobs=-1
):
    """Run Mahalanobis OOD detection experiment."""
    ood_dataset_names = ood_datasets or DEFAULT_OOD_DATASETS

    parts = ["Mahalanobis"]
    if normalize != "none":
        parts.append(f"normalize={normalize}")
    if transform_name != "none":
        parts.append(transform_name.upper())
    label = " + ".join(parts)

    print(f"{'='*80}")
    print(f"{label} OOD DETECTION")
    print(f"{'='*80}")
    print(f"Model: {model_name}, Image size: {img_size}")
    print(f"Runs: {num_runs}, Training size: {train_size}")
    print(f"Covariance: {covariance_type}, Regularization: {regularization}")
    print(f"OOD datasets: {ood_dataset_names}")
    if normalize != "none":
        print(f"Normalize: {normalize}")
    if transform_name != "none":
        print(f"Transform: {transform_name}, Percentile: {percentile}")
    print(f"{'='*80}\n")

    print("Loading data...")
    # NaN filtering only available for the original 4 OOD datasets
    nan_info = prepare_datasets(model_name, remove_nan=remove_nan)
    feature_vectors = load_feature_vectors(model_name, img_size)

    src_data = feature_vectors["lrad"]
    src_total = len(src_data)
    if remove_nan:
        src_data = filter_nan_files(src_data, nan_info['nan_files_lrad'])
        print(f"  ID: {len(src_data)} / {src_total} scans kept (removed {src_total - len(src_data)})")

    # Datasets with radiomics NaN info available
    nan_filterable = {'rsna', 'covid19', 'kits23', 'pancreas'}

    ood_data_dict = {}
    for ds_name in ood_dataset_names:
        ds_data = feature_vectors[ds_name]
        ds_total = len(ds_data)
        if remove_nan and ds_name in nan_filterable:
            ds_data = filter_nan_files(ds_data, nan_info['nan_files_ood'])
            print(f"  {ds_name.upper()}: {len(ds_data)} / {ds_total} scans kept (removed {ds_total - len(ds_data)})")
        else:
            print(f"  {ds_name.upper()}: {len(ds_data)} scans")
        ood_data_dict[ds_name] = ds_data

    filename_to_dataset = build_filename_to_dataset_mapping(
        src_data, ood_data_dict, id_name="lrad"
    )

    print(f"\nRunning {num_runs} iterations of {label}...")

    results_list = Parallel(n_jobs=n_jobs, verbose=10)(
        delayed(run_single_iteration)(
            base_seed + run_idx, src_data, ood_data_dict, train_size,
            covariance_type, regularization, transform_name, percentile,
            normalize, filename_to_dataset, "lrad"
        ) for run_idx in range(num_runs)
    )

    results = parse_unified_results(results_list, ood_dataset_names)

    print_results(results, label)

    results["config"] = {
        "model_name": model_name,
        "img_size": img_size,
        "num_runs": num_runs,
        "base_seed": base_seed,
        "train_size": train_size,
        "covariance_type": covariance_type,
        "regularization": regularization,
        "transform": transform_name,
        "percentile": percentile if transform_name != "none" else None,
        "normalize": normalize,
        "remove_nan": remove_nan,
    }

    def convert_to_serializable(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, dict):
            return {key: convert_to_serializable(value) for key, value in obj.items()}
        if isinstance(obj, list):
            return [convert_to_serializable(item) for item in obj]
        if isinstance(obj, (np.floating, np.integer)):
            return obj.item()
        return obj

    ensure_output_dirs()
    output_dir = ANALYSIS_RESULTS_ROOT / "ood_maha"
    output_dir.mkdir(parents=True, exist_ok=True)

    fname_parts = [model_name, f"size{img_size}"]
    if normalize != "none":
        fname_parts.append(f"norm-{normalize}")
    if transform_name != "none":
        fname_parts.append(f"{transform_name}_p{percentile}")
    fname_parts.append(f"cov-{covariance_type}")
    fname_parts.append(f"runs{num_runs}_seed{base_seed}")
    fname = "_".join(fname_parts) + ".json"

    output_path = output_dir / fname
    output_path.write_text(json.dumps(convert_to_serializable(results), indent=2))
    print(f"\nSaved results to: {output_path}")

    return results


def print_results(results, label="Mahalanobis"):
    """Print results."""
    print("\n" + "=" * 80)
    print(f"RESULTS — {label}")
    print("=" * 80)

    if 'global' in results:
        g = results['global']
        print(f"\nGLOBAL:")
        print(f"  AUROC: {g['auroc_mean']:.4f} ± {g['auroc_std']:.4f} | 95% CI: [{g['auroc_ci_lower']:.4f}, {g['auroc_ci_upper']:.4f}]")
        print(f"  FPR95: {g['fpr95_mean']:.4f} ± {g['fpr95_std']:.4f} | 95% CI: [{g['fpr95_ci_lower']:.4f}, {g['fpr95_ci_upper']:.4f}]")

    if 'per_dataset' in results:
        print(f"\nPER-DATASET:")
        for ds, metrics in results['per_dataset'].items():
            print(f"\n{ds.upper()}:")
            print(f"  AUROC: {metrics['auroc_mean']:.4f} ± {metrics['auroc_std']:.4f} | 95% CI: [{metrics['auroc_ci_lower']:.4f}, {metrics['auroc_ci_upper']:.4f}]")
            print(f"  FPR95: {metrics['fpr95_mean']:.4f} ± {metrics['fpr95_std']:.4f} | 95% CI: [{metrics['fpr95_ci_lower']:.4f}, {metrics['fpr95_ci_upper']:.4f}]")


def build_parser():
    parser = argparse.ArgumentParser(
        description="Mahalanobis distance OOD detection with optional feature transforms.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python ood_maha.py
  python ood_maha.py --normalize zscore
  python ood_maha.py --transform react --percentile 90 95 99
  python ood_maha.py --transform ash --percentile 80 90
  python ood_maha.py --normalize zscore --transform none react ash
  python ood_maha.py --covariance-type empirical ledoit_wolf
""",
    )
    parser.add_argument(
        "--normalize", nargs="+",
        choices=["none", "zscore"],
        default=["none"],
        help="Feature normalization(s) fitted on ID data, applied before transforms.",
    )
    parser.add_argument(
        "--transform", nargs="+",
        choices=["none", "react", "ash"],
        default=["none"],
        help="Feature transform(s) applied before Mahalanobis scoring.",
    )
    parser.add_argument(
        "--percentile", nargs="+", type=float, default=[90],
        help="Percentile(s) for react/ash transforms (ignored for none).",
    )
    parser.add_argument(
        "--covariance-type", nargs="+",
        choices=["empirical", "ledoit_wolf"],
        default=["ledoit_wolf"],
        help="One or more covariance estimators to evaluate.",
    )
    parser.add_argument(
        "--ood-datasets", nargs="+",
        default=None,
        help=f"OOD datasets to evaluate. Default: {DEFAULT_OOD_DATASETS}. "
             f"Available: {ALL_OOD_DATASETS}",
    )
    parser.add_argument("--model-name", default="smit", help="Feature model name.")
    parser.add_argument("--img-size", type=int, default=128, help="Feature image size.")
    parser.add_argument("--num-runs", type=int, default=100, help="Number of repeated runs.")
    parser.add_argument("--base-seed", type=int, default=2109, help="Base random seed.")
    parser.add_argument("--train-size", type=int, default=20,
                        help="Training samples per class for each repeated run.")
    parser.add_argument("--regularization", type=float, default=1e-5,
                        help="Diagonal regularization added to the covariance matrix.")
    parser.add_argument("--remove-nan", dest="remove_nan", action="store_true",
                        help="Filter scans with unusable radiomics rows before evaluation.")
    parser.add_argument("--keep-nan", dest="remove_nan", action="store_false",
                        help="Keep scans with unusable radiomics rows.")
    parser.set_defaults(remove_nan=True)
    parser.add_argument("--n-jobs", type=int, default=-1, help="Parallel workers.")
    return parser


def main():
    args = build_parser().parse_args()

    for norm in args.normalize:
        for transform in args.transform:
            percentiles = args.percentile if transform != "none" else [None]
            for pct in percentiles:
                for cov_type in args.covariance_type:
                    print(f"\n\n{'#'*80}")
                    print(f"# NORM: {norm.upper()}"
                          + f"  TRANSFORM: {transform.upper()}"
                          + (f"  PERCENTILE: {pct}" if pct else "")
                          + f"  COV: {cov_type.upper()}")
                    print(f"{'#'*80}\n")

                    run_experiment(
                        model_name=args.model_name,
                        img_size=args.img_size,
                        num_runs=args.num_runs,
                        base_seed=args.base_seed,
                        train_size=args.train_size,
                        covariance_type=cov_type,
                        regularization=args.regularization,
                        transform_name=transform,
                        percentile=pct if pct else 90,
                        normalize=norm,
                        ood_datasets=args.ood_datasets,
                        remove_nan=args.remove_nan,
                        n_jobs=args.n_jobs,
                    )


if __name__ == '__main__':
    main()
