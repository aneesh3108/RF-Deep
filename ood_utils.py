"""
Utility functions for OOD detection experiments.
Supports multiple evaluation strategies.
"""

import os.path as osp
import pickle
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier

from project_paths import PICKLE_ROOT, RADIOMICS_FEATURES_ROOT


def _normalize_feature_vector_keys(feature_vectors):
    """Normalize per-dataset feature-vector keys to basenames."""
    normalized = {}
    for dataset_name, dataset_vectors in feature_vectors.items():
        normalized_dataset = {}
        for filename, feature_matrix in dataset_vectors.items():
            basename = osp.basename(filename)
            if basename in normalized_dataset and filename != basename:
                raise ValueError(
                    f"Basename collision in dataset '{dataset_name}': '{filename}' "
                    f"conflicts with an existing entry for '{basename}'."
                )
            normalized_dataset[basename] = feature_matrix
        normalized[dataset_name] = normalized_dataset
    return normalized


def load_feature_vectors(model_name, img_size, crop_mode="anchored"):
    """Load pre-extracted feature vectors from pickle file."""
    tag = "" if crop_mode == "anchored" else f"_{crop_mode}"
    feature_vectors_path = PICKLE_ROOT / f"{model_name}_size{img_size}_featvec{tag}.pkl"
    with open(feature_vectors_path, "rb") as f:
        feature_vectors = pickle.load(f)
    return _normalize_feature_vector_keys(feature_vectors)


def extract_expanded_features(data_dict, filenames, mode="train", n_crops=4, seed=None):
    """
    Extract features from data dictionary with multiple crops per scan.
    
    Args:
        data_dict: Dictionary mapping filenames to feature matrices
        filenames: List of filenames to extract
        mode: "train" (4 crops) or "test" (8 crops)
        n_crops: Number of crops for training mode
        seed: Random seed for reproducible crop selection in train mode
    
    Returns:
        Tuple of (filenames_list, features_array)
    """
    if mode == "train" and seed is not None:
        np.random.seed(seed)
    
    expanded_filenames = []
    expanded_features = []

    N = n_crops if mode == "train" else 8

    for filename in filenames:
        feature_matrix = data_dict[filename]
        num_examples = len(feature_matrix)

        if mode == "test":
            # Use all 8 crops for test
            selected_indices = np.arange(min(8, num_examples))
        else:
            # Randomly select N crops for train
            selected_indices = np.random.choice(
                num_examples, 
                min(N, num_examples), 
                replace=False
            )

        selected_features = feature_matrix[selected_indices]

        for feature_vector in selected_features:
            expanded_filenames.append(osp.basename(filename))
            expanded_features.append(feature_vector.numpy().flatten())

    return expanded_filenames, np.array(expanded_features)


def sample_features(data, filenames, train_size, nan_files=None, mode="train", seed=None):
    """Sample scans and expand them to crop-level features."""
    if mode == "train" and nan_files is not None:
        nan_set = set(nan_files)
        clean_filenames = [f for f in filenames if osp.basename(f) not in nan_set]
        if len(clean_filenames) < train_size:
            raise ValueError("Not enough valid samples after excluding NaNs.")
    else:
        clean_filenames = filenames

    if mode == "train" and seed is not None:
        np.random.seed(seed)

    sampled_filenames = np.random.choice(clean_filenames, train_size, replace=False)
    return extract_expanded_features(data, sampled_filenames, mode=mode, seed=seed)


def train_model(X_train, y_train, trainer='random_forest', seed=42, rf_params=None):
    """
    Train a classifier.
    
    Args:
        X_train: Training features
        y_train: Training labels (0=ID, 1=OOD)
        trainer: 'random_forest' or 'simple_mlp'
        seed: Random seed
    
    Returns:
        Trained model
    """
    if trainer == 'random_forest':
        rf_defaults = {
            'n_estimators': 1000,
            'max_depth': 20,
            'class_weight': 'balanced',
            'random_state': seed,
            'n_jobs': -1,
        }
        if rf_params:
            rf_defaults.update(rf_params)
        model = RandomForestClassifier(
            **rf_defaults
        )
    elif trainer == 'simple_mlp':
        model = MLPClassifier(
            hidden_layer_sizes=(1000,),
            batch_size=50,
            solver='adam',
            max_iter=1000,
            random_state=seed
        )
    else:
        raise ValueError(f"Unsupported trainer type: {trainer}")
    
    model.fit(X_train, y_train)
    return model


def _apply_nan_overrides(y_probs, y_test, filenames_test, nan_files_lrad=None, nan_files_ood=None):
    """Force confident predictions for samples known to have unusable radiomics rows."""
    if not (nan_files_lrad or nan_files_ood):
        return y_probs

    override_mask = np.zeros_like(y_probs, dtype=bool)
    filenames_test_base = [osp.basename(f) for f in filenames_test]

    if nan_files_lrad:
        lrad_set = set(nan_files_lrad)
        override_mask |= np.array(
            [fname in lrad_set and label == 0 for fname, label in zip(filenames_test_base, y_test)]
        )

    if nan_files_ood:
        ood_set = set(nan_files_ood)
        override_mask |= np.array(
            [fname in ood_set and label == 1 for fname, label in zip(filenames_test_base, y_test)]
        )

    y_probs = y_probs.copy()
    y_probs[override_mask] = y_test[override_mask]
    return y_probs


def aggregate_predictions_by_filename(y_probs, y_test, filenames_test, filename_to_dataset=None):
    """
    Aggregate predictions by filename (average over crops).
    
    Args:
        y_probs: Raw prediction probabilities for each crop
        y_test: Ground truth labels for each crop
        filenames_test: Filenames for each crop
        filename_to_dataset: Optional mapping from filename to dataset name
    
    Returns:
        DataFrame with aggregated predictions per scan
    """
    filenames_test_base = [osp.basename(f) for f in filenames_test]
    
    df_probs = pd.DataFrame({
        "filename": filenames_test,
        "filename_base": filenames_test_base,
        "probability": y_probs,
        "label": y_test,
    })
    
    if filename_to_dataset is not None:
        df_probs["dataset"] = df_probs["filename_base"].map(filename_to_dataset)
    else:
        df_probs["dataset"] = None
    
    # Average predictions across crops for each scan
    df_avg_probs = df_probs.groupby("filename", as_index=False).agg({
        "probability": "mean",
        "label": "first",
        "dataset": "first",
    })
    
    return df_avg_probs


def train_and_evaluate(
    seed,
    src_data,
    ood_data,
    src_train_all,
    ood_train_all,
    train_size,
    X_test,
    y_test,
    filenames_test,
    trainer="random_forest",
    nan_files_lrad=None,
    nan_files_ood=None,
    feature_slice=None,
):
    """Train on one ID/OOD pair and return global AUROC and FPR95."""
    src_train_fil, src_train_feat = sample_features(
        src_data, src_train_all, train_size, nan_files=nan_files_lrad, seed=seed
    )
    ood_train_fil, ood_train_feat = sample_features(
        ood_data, ood_train_all, train_size, nan_files=nan_files_ood, seed=seed
    )

    X_train = np.vstack((src_train_feat, ood_train_feat))
    y_train = np.hstack(([0] * len(src_train_feat), [1] * len(ood_train_feat)))
    X_test_eval = X_test

    if feature_slice is not None:
        X_train = X_train[:, feature_slice]
        X_test_eval = X_test[:, feature_slice]

    model = train_model(X_train, y_train, trainer=trainer, seed=seed)
    y_probs = model.predict_proba(X_test_eval)[:, 1]
    y_probs = _apply_nan_overrides(
        y_probs,
        y_test,
        filenames_test,
        nan_files_lrad=nan_files_lrad,
        nan_files_ood=nan_files_ood,
    )

    df_avg_probs = aggregate_predictions_by_filename(y_probs, y_test, filenames_test)
    return compute_metrics(df_avg_probs["label"].values, df_avg_probs["probability"].values)


def train_and_evaluate2(
    seed,
    src_data,
    ood_data,
    src_train_all,
    ood_train_all,
    train_size,
    X_test,
    y_test,
    filenames_test,
    trainer="random_forest",
    nan_files_lrad=None,
    nan_files_ood=None,
    feature_slice=None,
    filename_to_dataset=None,
    id_dataset_name="lrad",
):
    """Train on one ID/OOD pair and return global plus per-dataset metrics."""
    src_train_fil, src_train_feat = sample_features(
        src_data, src_train_all, train_size, nan_files=nan_files_lrad, seed=seed
    )
    ood_train_fil, ood_train_feat = sample_features(
        ood_data, ood_train_all, train_size, nan_files=nan_files_ood, seed=seed
    )

    X_train = np.vstack((src_train_feat, ood_train_feat))
    y_train = np.hstack(([0] * len(src_train_feat), [1] * len(ood_train_feat)))
    X_test_eval = X_test

    if feature_slice is not None:
        X_train = X_train[:, feature_slice]
        X_test_eval = X_test[:, feature_slice]

    model = train_model(X_train, y_train, trainer=trainer, seed=seed)
    y_probs = model.predict_proba(X_test_eval)[:, 1]
    y_probs = _apply_nan_overrides(
        y_probs,
        y_test,
        filenames_test,
        nan_files_lrad=nan_files_lrad,
        nan_files_ood=nan_files_ood,
    )

    df_avg_probs = aggregate_predictions_by_filename(
        y_probs,
        y_test,
        filenames_test,
        filename_to_dataset=filename_to_dataset,
    )
    global_metrics = compute_metrics(df_avg_probs["label"].values, df_avg_probs["probability"].values)
    per_dataset_metrics = compute_per_dataset_metrics(df_avg_probs, id_dataset_name=id_dataset_name)
    return (*global_metrics, per_dataset_metrics)


def compute_metrics(y_true, y_scores):
    """
    Compute AUROC and FPR95.
    
    Args:
        y_true: Ground truth labels (0=ID, 1=OOD)
        y_scores: Prediction scores (higher = more OOD)
    
    Returns:
        Tuple of (auroc, fpr95)
    """
    auroc = roc_auc_score(y_true, y_scores)
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    idx_95 = np.where(tpr >= 0.95)[0]
    fpr95 = fpr[idx_95[0]] if len(idx_95) > 0 else 1.0
    
    return auroc, fpr95


def compute_per_dataset_metrics(df_avg_probs, id_dataset_name="lrad"):
    """
    Compute metrics for each OOD dataset separately vs ID.
    
    Args:
        df_avg_probs: DataFrame with aggregated predictions
        id_dataset_name: Name of the ID dataset
    
    Returns:
        Dictionary mapping dataset name to metrics
    """
    per_dataset_metrics = {}
    unique_datasets = df_avg_probs["dataset"].dropna().unique()
    
    for ds in unique_datasets:
        if ds == id_dataset_name:
            continue  # Skip ID dataset
        
        # Subset: ID + this specific OOD dataset
        mask = (df_avg_probs["dataset"] == id_dataset_name) | (df_avg_probs["dataset"] == ds)
        subset = df_avg_probs[mask]
        
        # Need both classes for ROC
        if subset["label"].nunique() < 2:
            continue
        
        y_true_ds = subset["label"].values
        y_score_ds = subset["probability"].values
        
        auroc_ds, fpr95_ds = compute_metrics(y_true_ds, y_score_ds)
        
        per_dataset_metrics[ds] = {
            "auroc": auroc_ds,
            "fpr95": fpr95_ds,
        }
    
    return per_dataset_metrics


def compute_95ci_percentile(data):
    """Compute 95% confidence interval using percentiles (bootstrap method)."""
    mean = np.mean(data)
    std = np.std(data)
    ci_lower, ci_upper = np.percentile(data, [2.5, 97.5])
    return mean, std, ci_lower, ci_upper


def compute_confidence_interval(scores, confidence_level=0.95):
    """Compute a percentile confidence interval over repeated metric values."""
    scores = np.array(scores)
    mean = np.mean(scores)

    alpha = 1 - confidence_level
    lower_percentile = (alpha / 2) * 100
    upper_percentile = (1 - alpha / 2) * 100

    lower = np.percentile(scores, lower_percentile)
    upper = np.percentile(scores, upper_percentile)

    return mean, lower, upper


def prepare_datasets(model_name, remove_nan=True):
    """
    Load and prepare datasets, removing NaN files if requested.
    
    Args:
        model_name: Name of the model
        remove_nan: Whether to remove NaN files
    
    Returns:
        Dictionary with prepared data
    """
    # Load radiomics CSVs to identify NaN files
    df_lrad = pd.read_csv(RADIOMICS_FEATURES_ROOT / f"{model_name}_lrad_src.csv")
    
    ood_datasets = ['rsna', 'covid19', 'kits23', 'pancreas']
    dfs = []
    per_dataset_counts = {
        "lrad": {
            "total": len(df_lrad),
            "nan_tagged": int(df_lrad["numVoxelsOrig"].isna().sum()),
        }
    }
    
    for ood in ood_datasets:
        df = pd.read_csv(RADIOMICS_FEATURES_ROOT / f"{model_name}_{ood}_src.csv")
        df["ood_type"] = ood
        dfs.append(df)
        per_dataset_counts[ood] = {
            "total": len(df),
            "nan_tagged": int(df["numVoxelsOrig"].isna().sum()),
        }
    
    df_ood = pd.concat(dfs, ignore_index=True)
    
    # Get NaN files
    nan_files_lrad = set(df_lrad[df_lrad['numVoxelsOrig'].isna()]['id'])
    nan_files_ood = set(df_ood[df_ood['numVoxelsOrig'].isna()]['id'])
    
    print(f"NaN-tagged LRAD scans: {len(nan_files_lrad)} / {len(df_lrad)}")
    print(f"NaN-tagged OOD scans (unique filenames across OOD sets): {len(nan_files_ood)} / {len(df_ood)}")
    for dataset_name in ["lrad", *ood_datasets]:
        counts = per_dataset_counts[dataset_name]
        print(
            f"  {dataset_name.upper()}: {counts['total'] - counts['nan_tagged']} kept, "
            f"{counts['nan_tagged']} NaN-tagged, {counts['total']} total"
        )
    
    # Remove NaN files if requested
    if remove_nan:
        print("\nFiltering out NaN-tagged scans for the experiment...")
        return {
            'nan_files_lrad': nan_files_lrad,
            'nan_files_ood': nan_files_ood,
            'remove_nan': True,
            'per_dataset_counts': per_dataset_counts,
        }
    else:
        print("\nKeeping NaN-tagged scans in the experiment.")
        return {
            'nan_files_lrad': None,
            'nan_files_ood': None,
            'remove_nan': False,
            'per_dataset_counts': per_dataset_counts,
        }


def filter_nan_files(data_dict, nan_files):
    """
    Filter out NaN files from a data dictionary.
    
    Args:
        data_dict: Dictionary mapping filenames to features
        nan_files: Set of NaN filenames to remove
    
    Returns:
        Filtered dictionary
    """
    if nan_files is None:
        return data_dict
    
    return {k: v for k, v in data_dict.items() if osp.basename(k) not in nan_files}


def build_filename_to_dataset_mapping(src_data, ood_data_dict, id_name="lrad"):
    """
    Build mapping from filename to dataset name.
    
    Args:
        src_data: ID dataset dictionary
        ood_data_dict: Dictionary mapping OOD dataset names to data dictionaries
        id_name: Name of ID dataset
    
    Returns:
        Dictionary mapping filename to dataset name
    """
    filename_to_dataset = {}
    
    # Add ID filenames
    for fname in src_data.keys():
        filename_to_dataset[osp.basename(fname)] = id_name
    
    # Add OOD filenames
    for ds_name, ds_data in ood_data_dict.items():
        for fname in ds_data.keys():
            filename_to_dataset[osp.basename(fname)] = ds_name
    
    return filename_to_dataset

# Convert numpy arrays to lists for JSON serialization
def convert_to_serializable(obj):
    """Recursively convert numpy arrays to lists."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_to_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_serializable(item) for item in obj]
    else:
        return obj
    
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
    
    # Split each OOD dataset individually
    for ds_name in ood_data_dict.keys():
        ds_filenames = ood_filenames_dict[ds_name]
        ds_train, ds_test = train_test_split(
            ds_filenames,
            train_size=train_fraction,
            random_state=seed  # Same seed for consistency
        )
        splits[ds_name] = {'train': ds_train, 'test': ds_test}
    
    return splits
