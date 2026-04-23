"""
Main script to run all OOD detection experiments for RF-Deep.
Supports: separate (dataset-specific), ensemble, unified, LODO, and LODO+
All methods use identical train/test splits for fair comparison.
"""

import argparse
import json
import warnings

import numpy as np
from joblib import Parallel, delayed
from sklearn.model_selection import train_test_split

from project_paths import ANALYSIS_RESULTS_ROOT, ensure_output_dirs
from ood_utils import (
    load_feature_vectors,
    prepare_datasets,
    filter_nan_files,
    build_filename_to_dataset_mapping,
    compute_95ci_percentile,
    extract_expanded_features,
    train_model,
    aggregate_predictions_by_filename,
    compute_metrics,
    compute_per_dataset_metrics,
)

_safe_sample_warned = False
_TRAIN_N_CROPS = 4  # module-level default; overridden by run_experiment()

# ============================================
# SPLIT CREATION (SHARED ACROSS ALL METHODS)
# ============================================

def create_splits_for_all_methods(
    src_filenames,
    ood_data_dict,
    ood_filenames_dict,
    seed,
    train_fraction=0.3
):
    """
    Create train/test splits that are consistent across all methods.
    
    Args:
        src_filenames: List of ID filenames
        ood_data_dict: Dict mapping dataset name to data
        ood_filenames_dict: Dict mapping dataset name to list of filenames
        seed: Random seed
        train_fraction: Fraction for training (0.3 = 30% train, 70% test)
    
    Returns:
        Dictionary with splits for ID and each OOD dataset
    """
    splits = {}
    
    # Split ID data
    src_train, src_test = train_test_split(
        src_filenames, 
        train_size=train_fraction, 
        random_state=seed
    )
    splits['id'] = {'train': src_train, 'test': src_test}
    
    # Split each OOD dataset individually (with same seed for consistency)
    for ds_name in ood_data_dict.keys():
        ds_filenames = ood_filenames_dict[ds_name]
        ds_train, ds_test = train_test_split(
            ds_filenames,
            train_size=train_fraction,
            random_state=seed
        )
        splits[ds_name] = {'train': ds_train, 'test': ds_test}
    
    return splits


def safe_sample(filenames, requested_size, seed=None):
    global _safe_sample_warned
    available = len(filenames)
    actual_size = min(requested_size, available)

    if actual_size < requested_size and not _safe_sample_warned:
        warnings.warn(
            f"Requested {requested_size} samples but only {available} available. Using all {available}.",
            UserWarning,
            stacklevel=2,
        )
        _safe_sample_warned = True

    if seed is not None:
        np.random.seed(seed)

    if actual_size == available:
        return np.array(filenames)
    else:
        return np.random.choice(filenames, actual_size, replace=False)


# ============================================
# METHOD IMPLEMENTATIONS
# ============================================

def unified_ensemble(
    seed,
    src_data,
    ood_data_dict,
    splits,
    train_size,
    trainer='random_forest',
    filename_to_dataset=None,
    id_dataset_name="lrad",
    return_models=False,
    rf_params=None,
):
    """Unified Ensemble: Train single classifier on ID vs all OOD pooled together."""
    # Get pre-computed splits
    src_train_all = splits['id']['train']
    src_test_all = splits['id']['test']
    
    # Combine all OOD train/test from splits
    ood_train_all = []
    ood_test_all = []
    ood_data = {}
    
    for ds_name in ood_data_dict.keys():
        ood_train_all.extend(splits[ds_name]['train'])
        ood_test_all.extend(splits[ds_name]['test'])
        
        # Combine OOD data dictionaries
        ood_data.update(ood_data_dict[ds_name])
    
    # Sample training data
    src_train_sampled = safe_sample(src_train_all, train_size, seed=seed)
    ood_train_sampled = safe_sample(ood_train_all, train_size, seed=seed)
    
    # Extract training features
    _, src_train_feat = extract_expanded_features(
        src_data, src_train_sampled, mode="train", n_crops=_TRAIN_N_CROPS, seed=seed
    )
    _, ood_train_feat = extract_expanded_features(
        ood_data, ood_train_sampled, mode="train", n_crops=_TRAIN_N_CROPS, seed=seed
    )
    
    # Extract test features
    filenames_test_id, X_test_id = extract_expanded_features(
        src_data, src_test_all, mode="test"
    )
    filenames_test_ood, X_test_ood = extract_expanded_features(
        ood_data, ood_test_all, mode="test"
    )
    
    # Combine
    X_train = np.vstack([src_train_feat, ood_train_feat])
    y_train = np.hstack([np.zeros(len(src_train_feat)), np.ones(len(ood_train_feat))])
    
    X_test = np.vstack([X_test_id, X_test_ood])
    y_test = np.hstack([np.zeros(len(X_test_id)), np.ones(len(X_test_ood))])
    filenames_test = filenames_test_id + filenames_test_ood
    
    # Train model
    model = train_model(X_train, y_train, trainer=trainer, seed=seed, rf_params=rf_params)
    
    # Predict
    y_probs = model.predict_proba(X_test)[:, 1]
    
    # Aggregate by filename
    df_avg_probs = aggregate_predictions_by_filename(
        y_probs, y_test, filenames_test, filename_to_dataset
    )
    
    # Compute metrics
    y_probs_avg = df_avg_probs["probability"].values
    y_test_avg = df_avg_probs["label"].values
    
    global_auroc, global_fpr95 = compute_metrics(y_test_avg, y_probs_avg)
    per_dataset_metrics = compute_per_dataset_metrics(df_avg_probs, id_dataset_name)
    
    if return_models:
        return global_auroc, global_fpr95, per_dataset_metrics, model, src_test_all
    return global_auroc, global_fpr95, per_dataset_metrics


def separate_classifiers(
    seed,
    src_data,
    ood_data_dict,
    splits,
    train_size,
    trainer='random_forest',
    filename_to_dataset=None,
    id_dataset_name="lrad",
    return_models=False,
    rf_params=None,
):
    """Separate Classifiers: Train one classifier per OOD dataset."""
    # Get pre-computed ID splits
    src_train_all = splits['id']['train']
    src_test_all = splits['id']['test']
    
    # Sample ID training data
    src_train_sampled = safe_sample(src_train_all, train_size, seed=seed)
    _, src_train_feat = extract_expanded_features(
        src_data, src_train_sampled, mode="train", n_crops=_TRAIN_N_CROPS, seed=seed
    )
    
    # Extract ID test features (same for all)
    filenames_test_id, X_test_id = extract_expanded_features(
        src_data, src_test_all, mode="test"
    )
    
    results = {}
    trained_models = {}
    
    # Train one classifier per OOD dataset
    for ds_name, ds_data in ood_data_dict.items():
        # Get pre-computed splits for this OOD dataset
        ds_train_all = splits[ds_name]['train']
        ds_test_all = splits[ds_name]['test']
        
        # Sample OOD training data
        ds_sample_seed = seed + hash(ds_name) % 1000
        ds_train_sampled = safe_sample(ds_train_all, train_size, seed=ds_sample_seed)
        _, ds_train_feat = extract_expanded_features(
            ds_data, ds_train_sampled, mode="train", n_crops=_TRAIN_N_CROPS, seed=ds_sample_seed
        )
        
        # Extract OOD test features
        filenames_test_ood, X_test_ood = extract_expanded_features(
            ds_data, ds_test_all, mode="test"
        )
        
        # Train classifier for this OOD dataset
        X_train = np.vstack([src_train_feat, ds_train_feat])
        y_train = np.hstack([np.zeros(len(src_train_feat)), np.ones(len(ds_train_feat))])
        
        model = train_model(X_train, y_train, trainer=trainer, seed=seed, rf_params=rf_params)
        trained_models[ds_name] = model
        
        # Test on ID + this OOD dataset only
        X_test = np.vstack([X_test_id, X_test_ood])
        y_test = np.hstack([np.zeros(len(X_test_id)), np.ones(len(X_test_ood))])
        filenames_test = filenames_test_id + filenames_test_ood
        
        y_probs = model.predict_proba(X_test)[:, 1]
        
        # Aggregate by filename
        df_avg_probs = aggregate_predictions_by_filename(
            y_probs, y_test, filenames_test, filename_to_dataset
        )
        
        y_probs_avg = df_avg_probs["probability"].values
        y_test_avg = df_avg_probs["label"].values
        
        auroc, fpr95 = compute_metrics(y_test_avg, y_probs_avg)
        results[ds_name] = {'auroc': auroc, 'fpr95': fpr95}
    
    if return_models:
        return results, trained_models, src_test_all
    return results


def ensemble_average(
    seed,
    src_data,
    ood_data_dict,
    splits,
    train_size,
    trainer='random_forest',
    filename_to_dataset=None,
    id_dataset_name="lrad",
    aggregation='avg',
    return_models=False,
    rf_params=None,
):
    """Ensemble: Train one classifier per OOD dataset, aggregate predictions."""
    # Get pre-computed ID splits
    src_train_all = splits['id']['train']
    src_test_all = splits['id']['test']
    
    # Sample ID training data
    src_train_sampled = safe_sample(src_train_all, train_size, seed=seed)
    _, src_train_feat = extract_expanded_features(
        src_data, src_train_sampled, mode="train", n_crops=_TRAIN_N_CROPS, seed=seed
    )
    
    # Extract ID test features
    filenames_test_id, X_test_id = extract_expanded_features(
        src_data, src_test_all, mode="test"
    )
    
    # Extract test features for each OOD dataset using pre-computed splits
    ood_test_data = {}
    for ds_name, ds_data in ood_data_dict.items():
        ds_test_all = splits[ds_name]['test']
        
        filenames_test_ood, X_test_ood = extract_expanded_features(
            ds_data, ds_test_all, mode="test"
        )
        
        ood_test_data[ds_name] = {
            'filenames': filenames_test_ood,
            'features': X_test_ood
        }
    
    # Combine all test data
    filenames_test = filenames_test_id.copy()
    X_test_list = [X_test_id]
    
    for ds_name in ood_data_dict.keys():
        filenames_test.extend(ood_test_data[ds_name]['filenames'])
        X_test_list.append(ood_test_data[ds_name]['features'])
    
    X_test = np.vstack(X_test_list)
    y_test = np.hstack([
        np.zeros(len(X_test_id)),
        *[np.ones(len(ood_test_data[ds]['features'])) for ds in ood_data_dict.keys()]
    ])
    
    # Train one classifier per OOD dataset and collect predictions
    all_predictions = []
    trained_models = {}
    
    for ds_name, ds_data in ood_data_dict.items():
        # Get pre-computed splits
        ds_train_all = splits[ds_name]['train']
        
        # Sample OOD training data
        ds_sample_seed = seed + hash(ds_name) % 1000
        ds_train_sampled = safe_sample(ds_train_all, train_size, seed=ds_sample_seed)
        _, ds_train_feat = extract_expanded_features(
            ds_data, ds_train_sampled, mode="train", n_crops=_TRAIN_N_CROPS, seed=ds_sample_seed
        )
        
        # Train classifier
        X_train = np.vstack([src_train_feat, ds_train_feat])
        y_train = np.hstack([np.zeros(len(src_train_feat)), np.ones(len(ds_train_feat))])
        
        model = train_model(X_train, y_train, trainer=trainer, seed=seed, rf_params=rf_params)
        trained_models[ds_name] = model
        
        # Predict on ALL test data
        y_probs = model.predict_proba(X_test)[:, 1]
        all_predictions.append(y_probs)
    
    # Aggregate predictions
    all_predictions = np.array(all_predictions)
    
    if aggregation == 'avg':
        y_probs_aggregated = np.mean(all_predictions, axis=0)
    elif aggregation == 'max':
        y_probs_aggregated = np.max(all_predictions, axis=0)
    else:
        raise ValueError(f"Unknown aggregation method: {aggregation}")
    
    # Aggregate by filename
    df_avg_probs = aggregate_predictions_by_filename(
        y_probs_aggregated, y_test, filenames_test, filename_to_dataset
    )
    
    # Compute metrics
    y_probs_avg = df_avg_probs["probability"].values
    y_test_avg = df_avg_probs["label"].values
    
    global_auroc, global_fpr95 = compute_metrics(y_test_avg, y_probs_avg)
    per_dataset_metrics = compute_per_dataset_metrics(df_avg_probs, id_dataset_name)
    
    if return_models:
        return global_auroc, global_fpr95, per_dataset_metrics, trained_models, src_test_all
    return global_auroc, global_fpr95, per_dataset_metrics


def lodo(
    seed,
    src_data,
    ood_data_dict,
    splits,
    train_size,
    test_dataset,
    trainer='random_forest',
    filename_to_dataset=None,
    id_dataset_name="lrad",
    test_only_datasets=None,
    rf_params=None,
):
    """LODO: Train on ID + all OOD except one, test on held-out OOD.
    
    Test-only datasets are never used for training. They are additionally
    evaluated using the LODO classifier, with per-dataset metrics returned.
    """
    if test_only_datasets is None:
        test_only_datasets = set()

    # Get pre-computed ID splits
    src_train_all = splits['id']['train']
    src_test_all = splits['id']['test']
    
    # Sample ID training data
    src_train_sampled = safe_sample(src_train_all, train_size, seed=seed)
    _, src_train_feat = extract_expanded_features(
        src_data, src_train_sampled, mode="train", n_crops=_TRAIN_N_CROPS, seed=seed
    )
    
    # Extract ID test features
    filenames_test_id, X_test_id = extract_expanded_features(
        src_data, src_test_all, mode="test"
    )
    
    # Collect training data from all OOD datasets EXCEPT test_dataset and test-only
    ood_train_features = []
    
    for ds_name, ds_data in ood_data_dict.items():
        if ds_name == test_dataset:
            continue  # Skip the held-out dataset
        if ds_name in test_only_datasets:
            continue  # Skip test-only datasets (they have no train split)
        
        # Get pre-computed splits for this OOD dataset
        ds_train_all = splits[ds_name]['train']
        
        # Sample OOD training data
        ds_sample_seed = seed + hash(ds_name) % 1000
        ds_train_sampled = safe_sample(ds_train_all, train_size, seed=ds_sample_seed)
        _, ds_train_feat = extract_expanded_features(
            ds_data, ds_train_sampled, mode="train", n_crops=_TRAIN_N_CROPS, seed=ds_sample_seed
        )
        
        ood_train_features.append(ds_train_feat)
    
    # Combine all OOD training data
    ood_train_combined = np.vstack(ood_train_features)
    
    # Train on ID + (all OOD except test_dataset and test-only)
    X_train = np.vstack([src_train_feat, ood_train_combined])
    y_train = np.hstack([
        np.zeros(len(src_train_feat)),
        np.ones(len(ood_train_combined))
    ])
    
    model = train_model(X_train, y_train, trainer=trainer, seed=seed, rf_params=rf_params)
    
    # Test on ID + held-out OOD dataset (using pre-computed split)
    test_ds_data = ood_data_dict[test_dataset]
    ds_test_all = splits[test_dataset]['test']
    
    filenames_test_ood, X_test_ood = extract_expanded_features(
        test_ds_data, ds_test_all, mode="test"
    )
    
    X_test = np.vstack([X_test_id, X_test_ood])
    y_test = np.hstack([np.zeros(len(X_test_id)), np.ones(len(X_test_ood))])
    filenames_test = filenames_test_id + filenames_test_ood
    
    y_probs = model.predict_proba(X_test)[:, 1]
    
    # Aggregate by filename
    df_avg_probs = aggregate_predictions_by_filename(
        y_probs, y_test, filenames_test, filename_to_dataset
    )
    
    y_probs_avg = df_avg_probs["probability"].values
    y_test_avg = df_avg_probs["label"].values
    
    auroc, fpr95 = compute_metrics(y_test_avg, y_probs_avg)

    # If there are test-only datasets, also evaluate them with this LODO model
    if test_only_datasets:
        per_dataset_metrics = {}
        for to_name in test_only_datasets:
            if to_name not in ood_data_dict:
                continue
            to_test_all = splits[to_name]['test']
            fnames_to, X_test_to = extract_expanded_features(
                ood_data_dict[to_name], to_test_all, mode="test"
            )
            X_test_combined = np.vstack([X_test_id, X_test_to])
            y_test_combined = np.hstack([np.zeros(len(X_test_id)), np.ones(len(X_test_to))])
            filenames_combined = filenames_test_id + fnames_to

            y_probs_to = model.predict_proba(X_test_combined)[:, 1]
            df_to = aggregate_predictions_by_filename(
                y_probs_to, y_test_combined, filenames_combined, filename_to_dataset
            )
            auroc_to, fpr95_to = compute_metrics(df_to["label"].values, df_to["probability"].values)
            per_dataset_metrics[to_name] = {'auroc': auroc_to, 'fpr95': fpr95_to}

        return auroc, fpr95, per_dataset_metrics
    
    return auroc, fpr95

# ============================================
# TEST-ONLY (UNSEEN) DATASET EVALUATION
# ============================================

def evaluate_test_only_datasets(
    models,
    src_data,
    src_test_filenames,
    test_only_data_dict,
    filename_to_dataset,
    id_dataset_name="lrad",
    aggregation='avg',
):
    """
    Evaluate already-trained classifier(s) on unseen test-only datasets.

    Args:
        models: Either a single trained model (for unified/lodo) or a dict
                mapping OOD dataset name to a trained model (for separate/ensemble).
        src_data: ID data dictionary.
        src_test_filenames: ID test filenames (from the same split used in training).
        test_only_data_dict: Dict mapping test-only dataset name to data dict.
        filename_to_dataset: Global filename -> dataset mapping.
        id_dataset_name: Name of the ID dataset.
        aggregation: How to combine predictions when models is a dict.
            'avg' = average across all per-dataset classifiers (ensemble style).
            'each' = evaluate each classifier separately and pick the max OOD prob
                     per test-only dataset (separate style — report per model).

    Returns:
        Dict mapping test-only dataset name to {'auroc': float, 'fpr95': float}.
    """
    if not test_only_data_dict:
        return {}

    # Extract ID test features (shared across all test-only evaluations)
    filenames_test_id, X_test_id = extract_expanded_features(
        src_data, src_test_filenames, mode="test"
    )

    results = {}

    for to_name, to_data in test_only_data_dict.items():
        to_filenames = list(to_data.keys())

        filenames_test_ood, X_test_ood = extract_expanded_features(
            to_data, to_filenames, mode="test"
        )

        X_test = np.vstack([X_test_id, X_test_ood])
        y_test = np.hstack([np.zeros(len(X_test_id)), np.ones(len(X_test_ood))])
        filenames_test = filenames_test_id + filenames_test_ood

        # Get predictions from model(s)
        if isinstance(models, dict):
            # Multiple per-dataset classifiers — ensemble their predictions
            all_preds = []
            for model in models.values():
                all_preds.append(model.predict_proba(X_test)[:, 1])
            all_preds = np.array(all_preds)
            if aggregation == 'avg':
                y_probs = np.mean(all_preds, axis=0)
            elif aggregation == 'max':
                y_probs = np.max(all_preds, axis=0)
            else:
                y_probs = np.mean(all_preds, axis=0)
        else:
            # Single model (unified, lodo)
            y_probs = models.predict_proba(X_test)[:, 1]

        df_avg = aggregate_predictions_by_filename(
            y_probs, y_test, filenames_test, filename_to_dataset
        )

        auroc, fpr95 = compute_metrics(
            df_avg["label"].values, df_avg["probability"].values
        )
        results[to_name] = {'auroc': auroc, 'fpr95': fpr95}

    return results


# ============================================
# PARALLEL RUNNERS
# ============================================

def run_single_iteration_unified(run_idx, base_seed, src_data, ood_data_dict, 
                                 src_filenames, ood_filenames_dict, train_size, 
                                 trainer, filename_to_dataset,
                                 test_only_data_dict=None, rf_params=None):
    """Run single iteration of unified ensemble, with optional test-only eval."""
    seed = base_seed + run_idx
    splits = create_splits_for_all_methods(
        src_filenames, ood_data_dict, ood_filenames_dict, seed
    )

    if test_only_data_dict:
        auroc, fpr95, per_ds, model, src_test_fnames = unified_ensemble(
            seed, src_data, ood_data_dict, splits, train_size, trainer,
            filename_to_dataset, return_models=True, rf_params=rf_params,
        )
        to_results = evaluate_test_only_datasets(
            model, src_data, src_test_fnames,
            test_only_data_dict, filename_to_dataset,
        )
        per_ds.update(to_results)
        return auroc, fpr95, per_ds

    return unified_ensemble(
        seed, src_data, ood_data_dict, splits, train_size, trainer, filename_to_dataset,
        rf_params=rf_params,
    )


def run_single_iteration_separate(run_idx, base_seed, src_data, ood_data_dict,
                                  src_filenames, ood_filenames_dict, train_size,
                                  trainer, filename_to_dataset,
                                  test_only_data_dict=None, rf_params=None):
    """Run single iteration of separate classifiers, with optional test-only eval."""
    seed = base_seed + run_idx
    splits = create_splits_for_all_methods(
        src_filenames, ood_data_dict, ood_filenames_dict, seed
    )

    if test_only_data_dict:
        std_results, trained_models, src_test_fnames = separate_classifiers(
            seed, src_data, ood_data_dict, splits, train_size, trainer,
            filename_to_dataset, return_models=True, rf_params=rf_params,
        )
        to_results = evaluate_test_only_datasets(
            trained_models, src_data, src_test_fnames,
            test_only_data_dict, filename_to_dataset,
            aggregation='avg',
        )
        std_results.update(to_results)
        return std_results

    return separate_classifiers(
        seed, src_data, ood_data_dict, splits, train_size, trainer, filename_to_dataset,
        rf_params=rf_params,
    )


def run_single_iteration_ensemble(run_idx, base_seed, src_data, ood_data_dict,
                                  src_filenames, ood_filenames_dict, train_size,
                                  trainer, filename_to_dataset, aggregation,
                                  test_only_data_dict=None, rf_params=None):
    """Run single iteration of ensemble, with optional test-only eval."""
    seed = base_seed + run_idx
    splits = create_splits_for_all_methods(
        src_filenames, ood_data_dict, ood_filenames_dict, seed
    )

    if test_only_data_dict:
        auroc, fpr95, per_ds, trained_models, src_test_fnames = ensemble_average(
            seed, src_data, ood_data_dict, splits, train_size, trainer,
            filename_to_dataset, "lrad", aggregation, return_models=True,
            rf_params=rf_params,
        )
        to_results = evaluate_test_only_datasets(
            trained_models, src_data, src_test_fnames,
            test_only_data_dict, filename_to_dataset,
            aggregation=aggregation,
        )
        per_ds.update(to_results)
        return auroc, fpr95, per_ds

    return ensemble_average(
        seed, src_data, ood_data_dict, splits, train_size, trainer,
        filename_to_dataset, "lrad", aggregation, rf_params=rf_params
    )


def run_single_iteration_lodo(run_idx, base_seed, src_data, ood_data_dict,
                              src_filenames, ood_filenames_dict, train_size,
                              trainer, filename_to_dataset, test_dataset,
                              test_only_data_dict=None, rf_params=None):
    """Run single iteration of LODO, with optional test-only eval."""
    seed = base_seed + run_idx
    splits = create_splits_for_all_methods(
        src_filenames, ood_data_dict, ood_filenames_dict, seed
    )
    auroc, fpr95 = lodo(
        seed, src_data, ood_data_dict, splits, train_size, test_dataset,
        trainer, filename_to_dataset, rf_params=rf_params
    )

    if not test_only_data_dict:
        return auroc, fpr95

    # Rebuild the LODO model to get the object for test-only eval
    src_train_all = splits['id']['train']
    src_test_all = splits['id']['test']
    src_train_sampled = safe_sample(src_train_all, train_size, seed=seed)
    _, src_train_feat = extract_expanded_features(
        src_data, src_train_sampled, mode="train", n_crops=_TRAIN_N_CROPS, seed=seed
    )

    ood_train_features = []
    for ds_name, ds_data in ood_data_dict.items():
        if ds_name == test_dataset:
            continue
        ds_train_all = splits[ds_name]['train']
        ds_sample_seed = seed + hash(ds_name) % 1000
        ds_train_sampled = safe_sample(ds_train_all, train_size, seed=ds_sample_seed)
        _, ds_train_feat = extract_expanded_features(
            ds_data, ds_train_sampled, mode="train", n_crops=_TRAIN_N_CROPS, seed=ds_sample_seed
        )
        ood_train_features.append(ds_train_feat)

    ood_train_combined = np.vstack(ood_train_features)
    X_train = np.vstack([src_train_feat, ood_train_combined])
    y_train = np.hstack([np.zeros(len(src_train_feat)), np.ones(len(ood_train_combined))])
    model = train_model(X_train, y_train, trainer=trainer, seed=seed, rf_params=rf_params)

    to_results = evaluate_test_only_datasets(
        model, src_data, src_test_all,
        test_only_data_dict, filename_to_dataset,
    )

    return auroc, fpr95, to_results


def run_method_parallel(method, num_runs, base_seed, src_data, ood_data_dict,
                       src_filenames, ood_filenames_dict, train_size, trainer,
                       filename_to_dataset, ood_dataset_names, n_jobs,
                       test_only_data_dict=None, **kwargs):
    """Generic parallel runner for any method."""
    if test_only_data_dict is None:
        test_only_data_dict = {}

    print(f"\nRunning {num_runs} iterations of {method}...")
    rf_params = kwargs.get('rf_params')
    
    if method == 'unified':
        def runner(run_idx, base_seed, src_data, ood_data_dict, src_filenames,
                   ood_filenames_dict, train_size, trainer, filename_to_dataset):
            return run_single_iteration_unified(
                run_idx, base_seed, src_data, ood_data_dict, src_filenames,
                ood_filenames_dict, train_size, trainer, filename_to_dataset,
                test_only_data_dict=test_only_data_dict,
                rf_params=rf_params,
            )
        has_per_dataset = True
    elif method == 'separate':
        def runner(run_idx, base_seed, src_data, ood_data_dict, src_filenames,
                   ood_filenames_dict, train_size, trainer, filename_to_dataset):
            return run_single_iteration_separate(
                run_idx, base_seed, src_data, ood_data_dict, src_filenames,
                ood_filenames_dict, train_size, trainer, filename_to_dataset,
                test_only_data_dict=test_only_data_dict,
                rf_params=rf_params,
            )
        has_per_dataset = True
    elif method == 'ensemble':
        def runner(run_idx, base_seed, src_data, ood_data_dict, src_filenames,
                   ood_filenames_dict, train_size, trainer, filename_to_dataset):
            return run_single_iteration_ensemble(
                run_idx, base_seed, src_data, ood_data_dict, src_filenames,
                ood_filenames_dict, train_size, trainer, filename_to_dataset,
                aggregation='avg',
                test_only_data_dict=test_only_data_dict,
                rf_params=rf_params,
            )
        has_per_dataset = True
    elif method == 'lodo':
        has_per_dataset = True
        return run_lodo_parallel_special(
            num_runs, base_seed, src_data, ood_data_dict, src_filenames,
            ood_filenames_dict, train_size, trainer, filename_to_dataset,
            ood_dataset_names, n_jobs,
            test_only_data_dict=test_only_data_dict,
            rf_params=rf_params,
        )
    else:
        raise ValueError(f"Unknown method: {method}")
    
    # Run iterations
    if n_jobs == 1:
        results_list = []
        for run_idx in range(num_runs):
            if (run_idx + 1) % 10 == 0:
                print(f"Completed {run_idx + 1}/{num_runs} runs...")
            result = runner(
                run_idx, base_seed, src_data, ood_data_dict, src_filenames,
                ood_filenames_dict, train_size, trainer, filename_to_dataset
            )
            results_list.append(result)
    else:
        results_list = Parallel(n_jobs=n_jobs, verbose=10)(
            delayed(runner)(
                run_idx, base_seed, src_data, ood_data_dict, src_filenames,
                ood_filenames_dict, train_size, trainer, filename_to_dataset
            ) for run_idx in range(num_runs)
        )
    
    # Parse results
    if method == 'separate':
        return parse_separate_results(results_list, ood_dataset_names)
    else:
        return parse_global_and_per_dataset_results(results_list, ood_dataset_names)


def run_lodo_parallel_special(num_runs, base_seed, src_data, ood_data_dict,
                              src_filenames, ood_filenames_dict, train_size,
                              trainer, filename_to_dataset, ood_dataset_names, n_jobs,
                              test_only_data_dict=None, rf_params=None):
    """Special parallel runner for LODO (tests each dataset separately)."""
    if test_only_data_dict is None:
        test_only_data_dict = {}

    all_report_names = list(ood_dataset_names)  # copy — includes test-only names from caller
    dataset_results = {ds: {'aurocs': [], 'fpr95s': []} for ds in all_report_names}

    # Standard OOD datasets that participate in LODO rotation
    standard_ood_names = [ds for ds in ood_dataset_names if ds not in test_only_data_dict]

    for test_ds in standard_ood_names:
        print(f"\n  Holding out: {test_ds.upper()}")
        
        if n_jobs == 1:
            results_list = []
            for run_idx in range(num_runs):
                if (run_idx + 1) % 20 == 0:
                    print(f"    Completed {run_idx + 1}/{num_runs} runs...")
                result = run_single_iteration_lodo(
                    run_idx, base_seed, src_data, ood_data_dict, src_filenames,
                    ood_filenames_dict, train_size, trainer, filename_to_dataset, test_ds,
                    test_only_data_dict=test_only_data_dict,
                    rf_params=rf_params,
                )
                results_list.append(result)
        else:
            results_list = Parallel(n_jobs=n_jobs, verbose=5)(
                delayed(run_single_iteration_lodo)(
                    run_idx, base_seed, src_data, ood_data_dict, src_filenames,
                    ood_filenames_dict, train_size, trainer, filename_to_dataset, test_ds,
                    test_only_data_dict=test_only_data_dict,
                    rf_params=rf_params,
                ) for run_idx in range(num_runs)
            )
        
        for result in results_list:
            if test_only_data_dict:
                auroc, fpr95, to_results = result
            else:
                auroc, fpr95 = result
                to_results = {}

            dataset_results[test_ds]['aurocs'].append(auroc)
            dataset_results[test_ds]['fpr95s'].append(fpr95)

            # Accumulate test-only results across LODO folds
            for to_name, to_metrics in to_results.items():
                dataset_results[to_name]['aurocs'].append(to_metrics['auroc'])
                dataset_results[to_name]['fpr95s'].append(to_metrics['fpr95'])
    
    return format_results_per_dataset_only(dataset_results, all_report_names)

# ============================================
# RESULT PARSING AND FORMATTING
# ============================================

def parse_global_and_per_dataset_results(results_list, ood_dataset_names):
    """Parse results that have both global and per-dataset metrics."""
    global_aurocs = []
    global_fpr95s = []
    dataset_results = {ds: {'aurocs': [], 'fpr95s': []} for ds in ood_dataset_names}
    
    for auroc, fpr95, per_ds in results_list:
        global_aurocs.append(auroc)
        global_fpr95s.append(fpr95)
        
        for ds, metrics in per_ds.items():
            dataset_results[ds]['aurocs'].append(metrics['auroc'])
            dataset_results[ds]['fpr95s'].append(metrics['fpr95'])
    
    return format_results_global_and_per_dataset(global_aurocs, global_fpr95s, dataset_results, ood_dataset_names)


def parse_separate_results(results_list, ood_dataset_names):
    """Parse results from separate classifiers (no global metrics).
    
    For test-only datasets, results arrive with keys like 'breast__via__rsna'.
    We collect all __via__ variants and report them alongside standard datasets.
    """
    # Discover all unique result keys across all runs
    all_keys = set()
    for result_dict in results_list:
        all_keys.update(result_dict.keys())

    # Build dataset_results for all keys (standard + __via__ keys)
    all_ds_names = list(ood_dataset_names)
    for key in sorted(all_keys):
        if key not in all_ds_names:
            all_ds_names.append(key)

    dataset_results = {ds: {'aurocs': [], 'fpr95s': []} for ds in all_ds_names}
    
    for result_dict in results_list:
        for ds, metrics in result_dict.items():
            if ds in dataset_results:
                dataset_results[ds]['aurocs'].append(metrics['auroc'])
                dataset_results[ds]['fpr95s'].append(metrics['fpr95'])
    
    return format_results_per_dataset_only(dataset_results, all_ds_names)


def format_results_global_and_per_dataset(global_aurocs, global_fpr95s, dataset_results, ood_dataset_names):
    """Format results with global and per-dataset metrics."""
    auroc_mean, auroc_std, auroc_lower, auroc_upper = compute_95ci_percentile(global_aurocs)
    fpr95_mean, fpr95_std, fpr95_lower, fpr95_upper = compute_95ci_percentile(global_fpr95s)
    
    results = {
        'global': {
            'auroc_mean': auroc_mean,
            'auroc_std': auroc_std,
            'auroc_ci_lower': auroc_lower,
            'auroc_ci_upper': auroc_upper,
            'fpr95_mean': fpr95_mean,
            'fpr95_std': fpr95_std,
            'fpr95_ci_lower': fpr95_lower,
            'fpr95_ci_upper': fpr95_upper,
            'all_aurocs': global_aurocs,
            'all_fpr95s': global_fpr95s
        },
        'per_dataset': {}
    }
    
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
            'all_aurocs': dataset_results[ds]['aurocs'],
            'all_fpr95s': dataset_results[ds]['fpr95s']
        }
    
    return results


def format_results_per_dataset_only(dataset_results, ood_dataset_names):
    """Format results for methods with only per-dataset metrics."""
    results = {'per_dataset': {}}
    
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
            'all_aurocs': dataset_results[ds]['aurocs'],
            'all_fpr95s': dataset_results[ds]['fpr95s']
        }
    
    return results


def print_results(results, method):
    """Print results in a nice format."""
    print("\n" + "=" * 80)
    print(f"RESULTS - {method.upper()}")
    print("=" * 80)
    
    if 'global' in results:
        g = results['global']
        print(f"\nGLOBAL (All OOD combined):")
        print(f"  AUROC: {g['auroc_mean']:.3f} ± {g['auroc_std']:.3f} | 95% CI: [{g['auroc_ci_lower']:.3f}, {g['auroc_ci_upper']:.3f}]")
        print(f"  FPR95: {g['fpr95_mean']:.3f} ± {g['fpr95_std']:.3f} | 95% CI: [{g['fpr95_ci_lower']:.3f}, {g['fpr95_ci_upper']:.3f}]")
    
    if 'per_dataset' in results and len(results['per_dataset']) > 0:
        print(f"\nPER-DATASET:")
        for ds, metrics in results['per_dataset'].items():
            print(f"\n{ds.upper()}:")
            print(f"  AUROC: {metrics['auroc_mean']:.3f} ± {metrics['auroc_std']:.3f} | 95% CI: [{metrics['auroc_ci_lower']:.3f}, {metrics['auroc_ci_upper']:.3f}]")
            print(f"  FPR95: {metrics['fpr95_mean']:.3f} ± {metrics['fpr95_std']:.3f} | 95% CI: [{metrics['fpr95_ci_lower']:.3f}, {metrics['fpr95_ci_upper']:.3f}]")


# ============================================
# MAIN EXPERIMENT RUNNER
# ============================================

def run_experiment(
    method='separate',
    model_name='mim',
    img_size=128,
    num_runs=100,
    base_seed=3108,
    train_size=-1,
    trainer='random_forest',
    remove_nan=True,
    n_jobs=-1,
    test_only_datasets=None,
    rf_params=None,
    output_tag=None,
    crop_mode='anchored',
    n_crops=4,
):
    """
    Run OOD detection experiment.

    Args:
        method: One of ['unified', 'separate', 'ensemble', 'lodo']
        test_only_datasets: Optional list of dataset names (e.g., ['breast', 'covid19a'])
            that are used ONLY for evaluation — never included in any training split.
            These must exist as keys in the feature vectors pickle file.
            Results for these datasets appear alongside the standard OOD results.
    """
    global _TRAIN_N_CROPS
    _TRAIN_N_CROPS = n_crops

    if test_only_datasets is None:
        test_only_datasets = []

    print(f"{'='*80}")
    print(f"OOD DETECTION EXPERIMENT - {method.upper()}")
    print(f"{'='*80}")
    print(f"Model: {model_name}, Image size: {img_size}, Train crops: {n_crops}")
    print(f"Runs: {num_runs}, Base seed: {base_seed}")
    print(f"Training size: {train_size} per class")
    print(f"Trainer: {trainer}")
    if rf_params:
        print(f"RF params override: {rf_params}")
    print(f"NaN handling: {'REMOVED' if remove_nan else 'KEPT'}")
    print(f"Parallelization: {'Enabled (n_jobs={})'.format(n_jobs) if n_jobs != 1 else 'Disabled'}")
    if test_only_datasets:
        print(f"Test-only (unseen) datasets: {test_only_datasets}")
    print(f"{'='*80}\n")
    
    # Load and prepare data
    print("Loading data...")
    nan_info = prepare_datasets(model_name, remove_nan=remove_nan)
    feature_vectors = load_feature_vectors(model_name, img_size, crop_mode=crop_mode)
    
    ood_dataset_names = ['rsna', 'covid19', 'kits23', 'pancreas']
    
    # Prepare ID data
    src_data = feature_vectors["lrad"]
    src_total = len(src_data)
    if remove_nan:
        src_data = filter_nan_files(src_data, nan_info['nan_files_lrad'])
        print(f"  ID dataset: {len(src_data)} / {src_total} scans kept (removed {src_total - len(src_data)})")
    
    # Prepare OOD data (standard — participates in train/test splits)
    ood_data_dict = {}
    ood_filenames_dict = {}
    
    for ds_name in ood_dataset_names:
        ds_data = feature_vectors[ds_name]
        ds_total = len(ds_data)
        if remove_nan:
            ds_data = filter_nan_files(ds_data, nan_info['nan_files_ood'])
            print(f"  {ds_name.upper()} dataset: {len(ds_data)} / {ds_total} scans kept (removed {ds_total - len(ds_data)})")
        
        ood_data_dict[ds_name] = ds_data
        ood_filenames_dict[ds_name] = list(ds_data.keys())
    
    # Prepare test-only datasets (never seen during training)
    # NaN filtering is NOT applied here — these datasets are evaluation-only
    # and we want to test on every available scan.
    test_only_data_dict = {}
    for ds_name in test_only_datasets:
        if ds_name not in feature_vectors:
            raise KeyError(
                f"Test-only dataset '{ds_name}' not found in feature vectors. "
                f"Available: {list(feature_vectors.keys())}"
            )
        ds_data = feature_vectors[ds_name]
        print(f"  {ds_name.upper()} (test-only, no NaN filter): {len(ds_data)} scans")
        test_only_data_dict[ds_name] = ds_data
    
    # Build filename to dataset mapping (include test-only datasets)
    all_ood_for_mapping = {**ood_data_dict, **test_only_data_dict}
    filename_to_dataset = build_filename_to_dataset_mapping(
        src_data, all_ood_for_mapping, id_name="lrad"
    )
    
    src_filenames = list(src_data.keys())
    
    # All dataset names for reporting (standard + test-only)
    all_ood_names = ood_dataset_names + test_only_datasets
    
    # Run experiment
    results = run_method_parallel(
        method, num_runs, base_seed, src_data, ood_data_dict,
        src_filenames, ood_filenames_dict, train_size, trainer,
        filename_to_dataset, all_ood_names, n_jobs,
        test_only_data_dict=test_only_data_dict,
        rf_params=rf_params,
    )
    
    # Print results
    print_results(results, method)
    
    # Add config
    results['config'] = {
        'method': method,
        'model_name': model_name,
        'img_size': img_size,
        'num_runs': num_runs,
        'base_seed': base_seed,
        'train_size': train_size,
        'trainer': trainer,
        'remove_nan': remove_nan,
        'n_jobs': n_jobs,
        'rf_params': rf_params or {},
    }
    
    # Save results
    def convert_to_serializable(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {key: convert_to_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_serializable(item) for item in obj]
        else:
            return obj
    
    results_serializable = convert_to_serializable(results)
    ensure_output_dirs()
    output_dir = ANALYSIS_RESULTS_ROOT / "ood_rfdeep"
    output_dir.mkdir(parents=True, exist_ok=True)
    suffix = f"_{output_tag}" if output_tag else ""
    output_path = output_dir / f"{method}_{model_name}_size{img_size}_runs{num_runs}_seed{base_seed}{suffix}.json"
    output_path.write_text(json.dumps(results_serializable, indent=2))
    print(f"\nSaved results to: {output_path}")
    
    return results


def build_parser():
    """Build the CLI parser for RF-Deep experiments."""
    parser = argparse.ArgumentParser(
        description="Run RF-Deep OOD detection experiments.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python ood_rfdeep.py --method lodo --model-name smit --img-size 128
  python ood_rfdeep.py --method unified separate --model-name smitmini --img-size 96
  python ood_rfdeep.py --method lodo --test-only-datasets breastc covid19a
""",
    )
    parser.add_argument(
        "--method",
        nargs="+",
        choices=["unified", "separate", "ensemble", "lodo"],
        default=["separate"],
        help="One or more RF-Deep evaluation methods to run.",
    )
    parser.add_argument("--model-name", default="smit", help="Feature model name.")
    parser.add_argument("--img-size", type=int, default=128, help="Feature image size.")
    parser.add_argument("--crop-mode", default="anchored", dest="crop_mode",
                        choices=["anchored", "spatial", "center"],
                        help="Which feature pkl to load (default: anchored).")
    parser.add_argument("--n-crops", type=int, default=4, dest="n_crops",
                        help="Number of crops per scan used during training (default: 4).")
    parser.add_argument("--num-runs", type=int, default=100, help="Number of repeated runs.")
    parser.add_argument("--base-seed", type=int, default=2109, help="Base random seed.")
    parser.add_argument(
        "--train-size",
        type=int,
        default=20,
        help="Training samples per class for each repeated run.",
    )
    parser.add_argument(
        "--trainer",
        choices=["random_forest", "mlp"],
        default="random_forest",
        help="Classifier used on top of the extracted features.",
    )
    parser.add_argument(
        "--remove-nan",
        dest="remove_nan",
        action="store_true",
        help="Filter scans with unusable radiomics rows before training.",
    )
    parser.add_argument(
        "--keep-nan",
        dest="remove_nan",
        action="store_false",
        help="Keep scans with unusable radiomics rows.",
    )
    parser.set_defaults(remove_nan=True)
    parser.add_argument("--n-jobs", type=int, default=-1, help="Parallel workers.")
    parser.add_argument(
        "--test-only-datasets",
        nargs="*",
        default=["breastc", "covid19a"],
        help="Datasets evaluated only at test time and never used for training.",
    )
    parser.add_argument(
        "--output-tag",
        default=None,
        help="Optional suffix appended to the saved results filename.",
    )
    return parser


def main():
    """CLI entrypoint for RF-Deep experiments."""
    args = build_parser().parse_args()

    for method in args.method:
        print(f"\n\n{'#'*80}")
        print(f"# RUNNING: {method.upper()}")
        print(f"{'#'*80}\n")

        run_experiment(
            method=method,
            model_name=args.model_name,
            img_size=args.img_size,
            num_runs=args.num_runs,
            base_seed=args.base_seed,
            train_size=args.train_size,
            trainer=args.trainer,
            remove_nan=args.remove_nan,
            n_jobs=args.n_jobs,
            test_only_datasets=args.test_only_datasets,
            output_tag=args.output_tag,
            crop_mode=args.crop_mode,
            n_crops=args.n_crops,
        )


if __name__ == '__main__':
    main()
