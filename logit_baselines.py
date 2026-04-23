"""
logit_baselines.py — Spatial uncertainty analysis for OOD detection in medical image segmentation.

Unified module combining:
  - Core processing (logits → metrics → segmentation masks)
  - Global analysis  (dataset-level statistics, AUROC/FPR95, bootstrap CIs)
  - Local visualization (slice selection, contour overlays, uncertainty heatmaps)

Usage:
  # Global OOD analysis across datasets
  python logit_baselines.py global --metric maxlogit
  python logit_baselines.py global --metric maxsoftmax
  python logit_baselines.py global --metric energy

  # Local per-case visualization
  python logit_baselines.py local --case R01-114 --dataset lung --mode overlay
  python logit_baselines.py local --case R01-114 --dataset lung --mode permodel
  python logit_baselines.py local --case AMC-020 --dataset lung2 --mode heatmap
"""

# ============================================================
# Imports
# ============================================================

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import os
import os.path as osp
import numpy as np
import torch
import torch.nn.functional as F
import pandas as pd
import matplotlib
import sys as _sys
# Use Agg only on headless environments (CI, remote servers without a display).
# On macOS and Windows a display is always present.
# On Linux, require either a DISPLAY (X11) or WAYLAND_DISPLAY variable.
def _has_display():
    if _sys.platform in ('darwin', 'win32'):
        return True
    return bool(os.environ.get('DISPLAY') or os.environ.get('WAYLAND_DISPLAY'))

if not _has_display():
    matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy import ndimage
from scipy.ndimage import label as label_connected_components
from scipy.ndimage import generate_binary_structure
from scipy.stats import mannwhitneyu, wilcoxon
from sklearn.metrics import roc_auc_score, roc_curve
from skimage import measure
import nibabel as nii

from project_paths import (
    BREASTC_ROOT,
    COVID19A_ROOT,
    COVID19_ROOT,
    KITS23_ROOT,
    LOGITS_ROOT,
    LRAD_AMC_ROOT,
    LRAD_R01_ROOT,
    LOGIT_BASELINES_RESULTS_ROOT,
    PANCREAS_ROOT,
    RESULTS_ROOT,
    RSNA_ROOT,
    ensure_output_dirs,
)


# ============================================================
# SECTION 1: CORE
# Shared processing functions used by both global and local modes.
# ============================================================

def extract_boundary_mask(segmentation, width=1):
    """Extract boundary voxels by morphological erosion.

    Args:
        segmentation: (H, W, D) bool array
        width: erosion width in voxels

    Returns:
        boundary: (H, W, D) bool array — ring of voxels at segmentation edge
    """
    structure = ndimage.generate_binary_structure(3, 1)
    eroded = ndimage.binary_erosion(segmentation, structure=structure, iterations=width)
    return segmentation & ~eroded


def compute_entropy_map(logits):
    """Compute per-voxel entropy from class logits.

    Args:
        logits: (C, H, W, D) float tensor

    Returns:
        entropy: (H, W, D) float tensor
    """
    probs = F.softmax(logits, dim=0)
    log_probs = torch.log(probs + 1e-10)
    return -torch.sum(probs * log_probs, dim=0)


def compute_max_softmax(logits):
    """Compute maximum softmax probability (MSP) per voxel.

    Args:
        logits: (C, H, W, D) float tensor or numpy array

    Returns:
        max_softmax: (H, W, D) numpy array
    """
    if isinstance(logits, np.ndarray):
        logits = torch.from_numpy(logits).float()
    softmax = F.softmax(logits, dim=0)
    return torch.max(softmax, dim=0)[0].cpu().numpy()


def compute_energy_score(logits):
    """Compute free-energy OOD score: Energy(x) = -log(sum(exp(logits))).

    Lower energy → more ID-like. Higher energy → more OOD-like.

    Args:
        logits: (C, H, W, D) float tensor or numpy array

    Returns:
        energy: (H, W, D) numpy array
    """
    if isinstance(logits, np.ndarray):
        logits = torch.from_numpy(logits).float()
    return -torch.logsumexp(logits, dim=0).cpu().numpy()


def process_numpy_logits(numpy_path, min_component_size=10, verbose=True):
    """Load logits from .npy file and compute all uncertainty metrics.

    Returns all metrics needed by both global analysis (maxlogit, maxsoftmax,
    energy) and local visualization (entropy, boundary, interior).

    Args:
        numpy_path: path to .npy file of shape (C, H, W, D)
        min_component_size: minimum voxels for a connected component to be kept
        verbose: print progress

    Returns:
        dict with keys:
            logits         — raw (C, H, W, D) numpy array
            maxlogit       — (H, W, D) max class logit per voxel
            maxsoftmax     — (H, W, D) max softmax probability per voxel
            energy         — (H, W, D) free-energy score per voxel
            entropy        — (H, W, D) softmax entropy per voxel
            segmap_binary  — (H, W, D) bool, filtered binary segmentation
            boundary       — (H, W, D) bool, boundary voxels
            interior       — (H, W, D) bool, interior voxels
    """
    numpy_path = Path(numpy_path)
    if verbose:
        print(f"  Loading: {numpy_path.name}")

    logits_np = np.load(numpy_path)
    logits = torch.from_numpy(logits_np).float()

    if verbose:
        print(f"  Shape: {logits.shape}")

    # --- Segmentation map and scalar metrics ---
    maxlogit_vals, segmap = torch.max(logits, dim=0)
    maxlogit_np   = maxlogit_vals.cpu().numpy()
    maxsoftmax_np = compute_max_softmax(logits)
    energy_np     = compute_energy_score(logits)
    entropy_np    = compute_entropy_map(logits).cpu().numpy()

    # --- Connected-component filtering ---
    segmap_np = segmap.cpu().numpy().astype(np.int16)
    segmap_cc, num_features = label_connected_components(
        segmap_np,
        structure=generate_binary_structure(3, 3),
        output=np.int16
    )
    if verbose:
        print(f"  Components found: {num_features}")

    component_counts  = np.bincount(segmap_cc.flatten())
    valid_components  = np.where(component_counts > min_component_size)[0]
    segmap_filtered   = np.where(np.isin(segmap_cc, valid_components), segmap_cc, 0)
    segmap_binary     = (segmap_filtered > 0).astype(bool)

    # --- Boundary / interior decomposition ---
    boundary = extract_boundary_mask(segmap_binary, width=1)
    interior = segmap_binary & ~boundary

    return {
        'logits':        logits_np,
        'maxlogit':      maxlogit_np,
        'maxsoftmax':    maxsoftmax_np,
        'energy':        energy_np,
        'entropy':       entropy_np,
        'segmap_binary': segmap_binary,
        'boundary':      boundary,
        'interior':      interior,
    }


def find_best_slice(processed_data, min_seg_voxels=50, axis=2):
    """Score all slices along `axis` and return the most informative one.

    Scoring combines: segmentation size, entropy variance, boundary-interior
    entropy contrast. Use axis=2 for standard lung (H,W,D) orientation and
    axis=0 for KiTS-style (D,H,W) orientation.

    Args:
        processed_data: dict from process_numpy_logits
        min_seg_voxels: minimum segmentation voxels to consider a slice
        axis: depth axis to slice along (2 = lung/default, 0 = KiTS)

    Returns:
        best_slice: int index, or None if no suitable slice found
    """
    seg      = processed_data['segmap_binary']
    entropy  = processed_data['entropy']
    boundary = processed_data['boundary']
    interior = processed_data['interior']

    def _slice(arr, idx):
        if axis == 0:
            return arr[idx, :, :]
        elif axis == 1:
            return arr[:, idx, :]
        else:
            return arr[:, :, idx]

    n_slices = seg.shape[axis]
    slice_scores = []

    for z in range(n_slices):
        seg_sl  = _slice(seg, z)
        seg_cnt = seg_sl.sum()
        if seg_cnt < min_seg_voxels:
            continue

        bnd_sl  = _slice(boundary, z)
        int_sl  = _slice(interior, z)
        ent_sl  = _slice(entropy, z)

        size_score = seg_cnt
        variance_score = ent_sl[seg_sl].std() if seg_cnt > 0 else 0

        if bnd_sl.sum() > 10 and int_sl.sum() > 10:
            contrast_score = abs(ent_sl[bnd_sl].mean() - ent_sl[int_sl].mean())
        else:
            contrast_score = 0

        total = size_score * 0.4 + variance_score * 100 * 0.3 + contrast_score * 100 * 0.3
        slice_scores.append((z, total, seg_cnt))

    if not slice_scores:
        print(f"  No slices with >{min_seg_voxels} segmentation voxels found.")
        return None

    slice_scores.sort(key=lambda x: x[1], reverse=True)
    best = slice_scores[0]
    print(f"  Best slice: {best[0]} (score={best[1]:.1f}, voxels={best[2]})")
    return best[0]


# ============================================================
# SECTION 2: GLOBAL ANALYSIS
# Dataset-level statistics, statistical testing, AUROC/FPR95.
# ============================================================

def _summarize_logits_file(job):
    dataset_name, fpath, min_component_size, pooling = job
    fname = osp.basename(fpath)
    try:
        x = process_numpy_logits(fpath, min_component_size=min_component_size, verbose=False)
        seg = x['segmap_binary']
        if seg.sum() == 0:
            return {"scan_name": fname, "status": "no_segmentation"}

        def _pool(values):
            seg_vals = values[seg]
            if pooling == 'mean':
                return float(seg_vals.mean())
            if pooling == 'max':
                return float(seg_vals.max())
            if pooling == 'p95':
                return float(np.percentile(seg_vals, 95))
            if pooling == 'median':
                return float(np.median(seg_vals))
            raise ValueError(f"Unknown pooling: {pooling}")

        return {
            "scan_name": fname,
            "status": "ok",
            "row": {
                'scan_name': fname,
                'dataset': dataset_name,
                'maxlogit': _pool(x['maxlogit']),
                'maxsoftmax': _pool(x['maxsoftmax']),
                'energy': _pool(x['energy']),
            },
        }
    except Exception as e:
        return {"scan_name": fname, "status": "error", "error": str(e)}


def compute_dataset_statistics(logits_dir, dataset_name, min_component_size=300, n_jobs=4, pooling='mean'):
    """Process all .npy files in a directory and return per-scan metric DataFrame.

    Args:
        logits_dir: path or list of paths containing .npy logit files
        dataset_name: label for this dataset
        min_component_size: passed to process_numpy_logits

    Returns:
        pd.DataFrame with columns: scan_name, dataset, maxlogit, maxsoftmax, energy
    """
    if isinstance(logits_dir, (str, Path)):
        logits_dirs = [logits_dir]
    else:
        logits_dirs = logits_dir

    jobs = []
    for logits_path in logits_dirs:
        logits_path = str(logits_path)
        if not osp.isdir(logits_path):
            print(f"  WARNING: logits directory not found, skipping: {logits_path}")
            continue
        for fname in sorted(os.listdir(logits_path)):
            if not fname.endswith('.npy'):
                continue
            jobs.append((dataset_name, osp.join(logits_path, fname), min_component_size, pooling))

    rows = []
    n_processed = 0
    if jobs:
        with ProcessPoolExecutor(max_workers=n_jobs) as executor:
            for result in executor.map(_summarize_logits_file, jobs):
                fname = result["scan_name"]
                if result["status"] == "ok":
                    rows.append(result["row"])
                    n_processed += 1
                    print(f"  {fname}... ✓")
                elif result["status"] == "no_segmentation":
                    print(f"  {fname}... no segmentation")
                else:
                    print(f"  {fname}... ERROR: {result['error']}")

    print(f"\n  {dataset_name}: {n_processed} scans processed")
    return pd.DataFrame(rows)


def print_summary_statistics(df):
    """Print mean ± std for all three metrics."""
    if df.empty:
        print("\n  (no scans — skipping summary)")
        return
    name = df['dataset'].iloc[0]
    print(f"\n{name} (n={len(df)})")
    print(f"  MaxLogit:   {df['maxlogit'].mean():.3f} ± {df['maxlogit'].std():.3f}")
    print(f"  MaxSoftmax: {df['maxsoftmax'].mean():.3f} ± {df['maxsoftmax'].std():.3f}")
    print(f"  Energy:     {df['energy'].mean():.3f} ± {df['energy'].std():.3f}")


def perform_statistical_tests(id_df, ood_df_list):
    """Mann-Whitney U tests: ID vs each OOD dataset, for all three metrics."""
    print("\n" + "=" * 80)
    print("STATISTICAL TESTING  (Mann-Whitney U, two-sided)")
    print("=" * 80)

    if id_df.empty:
        print("  ID dataset is empty — skipping.")
        return

    metrics = ['maxlogit', 'maxsoftmax', 'energy']

    for i, ood_df in enumerate(ood_df_list):
        id_name  = id_df['dataset'].iloc[0]
        ood_name = ood_df['dataset'].iloc[0] if not ood_df.empty else f'(empty-{i})'
        print(f"\n{id_name}  vs  {ood_name}")
        print("-" * 60)

        if ood_df.empty:
            print("  OOD dataset is empty — skipping.")
            continue

        for metric in metrics:
            id_vals  = id_df[metric].dropna().values
            ood_vals = ood_df[metric].dropna().values

            if len(id_vals) == 0 or len(ood_vals) == 0:
                print(f"  {metric}: insufficient data")
                continue

            u, p = mannwhitneyu(id_vals, ood_vals, alternative='two-sided')
            n1, n2 = len(id_vals), len(ood_vals)
            r = 1 - (2 * u) / (n1 * n2)
            effect = ('negligible' if abs(r) < 0.1 else
                      'small'      if abs(r) < 0.3 else
                      'medium'     if abs(r) < 0.5 else 'large')
            sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'ns'

            print(f"  {metric}:")
            print(f"    ID  median={np.median(id_vals):.3f}  n={n1}")
            print(f"    OOD median={np.median(ood_vals):.3f}  n={n2}")
            print(f"    U={u:.1f}  p={p:.6f} {sig}  r={r:.3f} ({effect})")


def compute_ood_metrics(id_df, ood_df, metric_name='maxlogit'):
    """Compute AUROC and FPR95 for a single ID-vs-OOD pair.

    Score polarity convention:
        maxlogit, maxsoftmax: higher = more ID → negate for OOD scoring
        energy:               higher = more OOD → keep as-is

    Returns:
        dict with keys auroc, fpr95
    """
    id_vals  = id_df[metric_name].dropna().values
    ood_vals = ood_df[metric_name].dropna().values

    if len(id_vals) == 0 or len(ood_vals) == 0:
        return {'auroc': np.nan, 'fpr95': np.nan}

    scores = np.concatenate([id_vals, ood_vals])
    labels = np.concatenate([np.zeros(len(id_vals)), np.ones(len(ood_vals))])
    ood_scores = -scores if metric_name in ('maxlogit', 'maxsoftmax') else scores

    auroc = roc_auc_score(labels, ood_scores)
    fpr, tpr, _ = roc_curve(labels, ood_scores)
    idx95 = np.where(tpr >= 0.95)[0]
    fpr95 = fpr[idx95[0]] if len(idx95) > 0 else np.nan

    return {'auroc': auroc, 'fpr95': fpr95}


def compute_all_ood_metrics(id_df, ood_df_list, metric_name='maxlogit'):
    """Compute AUROC / FPR95 for all OOD datasets."""
    print("\n" + "=" * 80)
    print(f"OOD DETECTION  ({metric_name.upper()})  —  AUROC & FPR95")
    print("=" * 80)

    if id_df.empty:
        print("  ID dataset is empty — skipping.")
        return {}

    results = {}
    for ood_df in ood_df_list:
        name = ood_df['dataset'].iloc[0] if not ood_df.empty else f'(empty-{len(results)})'
        if ood_df.empty:
            print(f"  {name:<30}  skipped (empty)")
            results[name] = {'auroc': np.nan, 'fpr95': np.nan}
            continue
        m = compute_ood_metrics(id_df, ood_df, metric_name)
        results[name] = m
        print(f"  {name:<30}  AUROC={m['auroc']:.4f}  FPR95={m['fpr95']:.4f}")

    return results


def compute_ood_metrics_bootstrap(id_df, ood_df, metric_name='maxlogit',
                                   n_bootstrap=100, sample_fraction=1.0,
                                   random_seed=42):
    """Bootstrap AUROC and FPR95 with 95% confidence intervals.

    Args:
        id_df, ood_df: DataFrames for ID and OOD datasets
        metric_name: 'maxlogit', 'maxsoftmax', or 'energy'
        n_bootstrap: number of bootstrap resamples
        sample_fraction: fraction of each dataset to sample per iteration
        random_seed: for reproducibility

    Returns:
        dict with auroc_mean, auroc_std, auroc_ci, fpr95_mean, fpr95_std, fpr95_ci
        All values are np.nan if inputs are insufficient.
    """
    _nan_result = {
        'auroc_mean': np.nan, 'auroc_std': np.nan, 'auroc_ci': [np.nan, np.nan],
        'fpr95_mean': np.nan, 'fpr95_std': np.nan, 'fpr95_ci': [np.nan, np.nan],
    }

    if id_df.empty or ood_df.empty:
        print(f"  Skipping bootstrap: empty DataFrame(s)")
        return _nan_result

    np.random.seed(random_seed)

    id_raw  = id_df[metric_name].values.copy().astype(float)
    ood_raw = ood_df[metric_name].values.copy().astype(float)

    # Guard: all-NaN columns
    id_mean  = np.nanmean(id_raw)
    ood_mean = np.nanmean(ood_raw)
    if np.isnan(id_mean) or np.isnan(ood_mean):
        print(f"  Skipping bootstrap: all-NaN values in {metric_name}")
        return _nan_result

    # Replace NaN with dataset mean so sample sizes stay stable
    id_raw[np.isnan(id_raw)]   = id_mean
    ood_raw[np.isnan(ood_raw)] = ood_mean

    n_id  = max(1, int(len(id_raw)  * sample_fraction))
    n_ood = max(1, int(len(ood_raw) * sample_fraction))

    auroc_boot, fpr95_boot = [], []

    for _ in range(n_bootstrap):
        id_s  = id_raw[np.random.choice(len(id_raw),   size=n_id,  replace=True)]
        ood_s = ood_raw[np.random.choice(len(ood_raw), size=n_ood, replace=True)]

        scores = np.concatenate([id_s, ood_s])
        labels = np.concatenate([np.zeros(n_id), np.ones(n_ood)])
        ood_scores = -scores if metric_name in ('maxlogit', 'maxsoftmax') else scores

        # Skip degenerate bootstrap samples (single unique label)
        if len(np.unique(labels)) < 2:
            continue

        try:
            auroc = roc_auc_score(labels, ood_scores)
        except ValueError:
            continue

        fpr, tpr, _ = roc_curve(labels, ood_scores)
        idx95 = np.where(tpr >= 0.95)[0]
        fpr95 = fpr[idx95[0]] if len(idx95) > 0 else np.nan

        auroc_boot.append(auroc)
        if not np.isnan(fpr95):
            fpr95_boot.append(fpr95)

    if len(auroc_boot) == 0:
        print(f"  Bootstrap produced no valid samples for {metric_name}")
        return _nan_result

    auroc_boot = np.array(auroc_boot)
    fpr95_boot = np.array(fpr95_boot)

    return {
        'auroc_mean': auroc_boot.mean(),
        'auroc_std':  auroc_boot.std(),
        'auroc_ci':   np.percentile(auroc_boot, [2.5, 97.5]),
        'fpr95_mean': fpr95_boot.mean()  if len(fpr95_boot) else np.nan,
        'fpr95_std':  fpr95_boot.std()   if len(fpr95_boot) else np.nan,
        'fpr95_ci':   np.percentile(fpr95_boot, [2.5, 97.5]) if len(fpr95_boot) else [np.nan, np.nan],
    }


def compute_all_ood_metrics_bootstrap(id_df, ood_df_list, metric_name='maxlogit',
                                       n_bootstrap=100, sample_fraction=1.0,
                                       random_seed=42):
    """Bootstrap OOD metrics for all OOD datasets."""
    print("\n" + "=" * 80)
    print(f"OOD DETECTION (BOOTSTRAP)  —  {metric_name.upper()}")
    print(f"  seed={random_seed}  iters={n_bootstrap}  sample={sample_fraction*100:.0f}%")
    print("=" * 80)

    results = {}
    for ood_df in ood_df_list:
        # Use the dataset column if available; fall back to index-based label
        if not ood_df.empty:
            name = ood_df['dataset'].iloc[0]
        else:
            name = f'(empty-{len(results)})'
        m    = compute_ood_metrics_bootstrap(id_df, ood_df, metric_name,
                                              n_bootstrap, sample_fraction, random_seed)
        results[name] = m
        auroc_ci  = m['auroc_ci']
        fpr95_val = m['fpr95_mean']
        fpr95_ci  = m['fpr95_ci']
        print(f"  {name:<30}  "
              f"AUROC={m['auroc_mean']:.4f}±{m['auroc_std']:.4f} "
              f"[{auroc_ci[0]:.4f},{auroc_ci[1]:.4f}]  "
              f"FPR95={fpr95_val:.4f}±{m['fpr95_std']:.4f} "
              f"[{fpr95_ci[0]:.4f},{fpr95_ci[1]:.4f}]")

    return results


# ============================================================
# SECTION 3: LOCAL VISUALIZATION
# Single-case slice selection and plotting routines.
# ============================================================

# --- Dataset configs (edit paths here) ---
DATASET_CONFIGS = {
    'lung': {
        'image_dir':  str(LRAD_R01_ROOT / 'image'),
        'logits_dir': str(LOGITS_ROOT / 'smit_main' / 'lung_test_LRAD_srcnorm' / 'numpy'),
        'axis': 2,
        'orient': 'lung',
    },
    'lung2': {
        'image_dir':  str(LRAD_AMC_ROOT / 'image'),
        'logits_dir': str(LOGITS_ROOT / 'smit_main' / 'lung_test_LRAD_srcnorm' / 'numpy'),
        'axis': 2,
        'orient': 'lung2',
    },
    'rsna': {
        'image_dir':  str(RSNA_ROOT / 'train_nii'),
        'logits_dir': str(LOGITS_ROOT / 'smit_farood' / 'lung_test_rsna_srcnorm' / 'numpy'),
        'axis': 2,
        'orient': 'rsna',
    },
    'covid19': {
        'image_dir':  str(COVID19_ROOT),
        'logits_dir': str(LOGITS_ROOT / 'smit_farood' / 'lung_test_covid19_srcnorm' / 'numpy'),
        'axis': 2,
        'orient': 'covid19',
    },
    'kits': {
        'image_dir':  str(KITS23_ROOT / 'images'),
        'logits_dir': str(LOGITS_ROOT / 'smit_farood' / 'lung_test_kits23_srcnorm' / 'numpy'),
        'axis': 0,          # KiTS uses (D, H, W) orientation
        'orient': 'rsna',   # same rotation as RSNA
    },
    'pancreas': {
        'image_dir':  str(PANCREAS_ROOT),
        'logits_dir': str(LOGITS_ROOT / 'smit_farood' / 'lung_test_pancreas_srcnorm' / 'numpy'),
        'axis': 2,
        'orient': 'lung',
    },
    'breastc': {
        'image_dir':  str(BREASTC_ROOT / 'imgs'),
        'logits_dir': str(LOGITS_ROOT / 'smit_farood' / 'lung_test_breastc_srcnorm' / 'numpy'),
        'axis': 2,
        'orient': 'breastc',
    },
    'covid19a': {
        'image_dir':  str(COVID19A_ROOT),
        'logits_dir': str(LOGITS_ROOT / 'smit_farood' / 'lung_test_covid19a_srcnorm' / 'numpy'),
        'axis': 2,
        'orient': 'covid19',
    },
}

MODEL_DIRS = {
    'SMIT':          str(RESULTS_ROOT / 'smit_main' / 'lung_test_LRAD_srcnorm' / 'nii'),
    'SimMIM':        str(RESULTS_ROOT / 'mim_main' / 'lung_test_LRAD_srcnorm' / 'nii'),
    'iBOT':          str(RESULTS_ROOT / 'ibot_main' / 'lung_test_LRAD_srcnorm' / 'nii'),
    'SMIT-Lite':     str(RESULTS_ROOT / 'smitmini_main' / 'lung_test_LRAD_srcnorm' / 'nii'),
    'SwinUNETR-10k': str(RESULTS_ROOT / 'swinunetr_10k_main' / 'lung_test_LRAD_srcnorm' / 'nii'),
    'SwinUNETR':     str(RESULTS_ROOT / 'swinunetr_main' / 'lung_test_LRAD_srcnorm' / 'nii'),
}


def _orient_slice(arr2d, orient):
    """Apply dataset-specific 2D orientation correction."""
    if orient in ('lung', 'lung2'):
        return np.flipud(arr2d)
    elif orient == 'breastc':
        return np.flipud(np.fliplr(np.rot90(arr2d, k=1)))
    elif orient in ('rsna', 'kits'):
        return np.rot90(arr2d, k=-1)
    elif orient == 'covid19':
        return np.flipud(np.rot90(arr2d, k=-2))
    return arr2d


def _get_slice(volume, idx, axis):
    """Extract a 2D slice from a 3D volume along the given axis."""
    if axis == 0: return volume[idx, :, :]
    if axis == 1: return volume[:, idx, :]
    return volume[:, :, idx]


def _normalize_ct(image, hu_min=-400, hu_max=400):
    """Clip and normalize CT to uint8."""
    clipped = np.clip(image, hu_min, hu_max)
    return (((clipped - hu_min) / (hu_max - hu_min)) * 255).astype(np.uint8)


def visualize_heatmaps(processed_data, image, slice_idx, orient='lung',
                       axis=2, save_dir=None):
    """Produce entropy/maxlogit heatmaps + distribution plots for one slice.

    Generates six figures:
        image.png          — CT slice with boundary contour
        entropy.png        — entropy heatmap
        entropydist.png    — entropy distributions (boundary vs interior)
        maxlogits.png      — maxlogit heatmap
        maxlogitsdist.png  — maxlogit distributions (boundary vs interior)
        intensity.png      — CT intensity histogram within segmentation

    Args:
        processed_data: dict from process_numpy_logits
        image: (H, W, D) or (D, H, W) raw CT volume
        slice_idx: which slice to visualize
        orient: orientation tag for _orient_slice
        axis: depth axis (2 = default, 0 = KiTS)
        save_dir: output directory (show interactively if None)
    """
    if save_dir:
        out = Path(save_dir)
        out.mkdir(parents=True, exist_ok=True)

    def show_or_save(fig, fname):
        if save_dir:
            fig.savefig(out / fname, dpi=300, bbox_inches='tight')
            plt.close(fig)
        else:
            plt.show()
            plt.close(fig)

    image_u8   = _normalize_ct(image)
    img_sl     = _orient_slice(_get_slice(image_u8,                  slice_idx, axis), orient)
    ent_sl     = _orient_slice(_get_slice(processed_data['entropy'],  slice_idx, axis), orient)
    ml_sl      = _orient_slice(_get_slice(processed_data['maxlogit'], slice_idx, axis), orient)
    bnd_sl     = _orient_slice(_get_slice(processed_data['boundary'], slice_idx, axis), orient)
    seg        = processed_data['segmap_binary']
    boundary   = processed_data['boundary']
    interior   = processed_data['interior']

    # Figure 1 — CT + boundary contour
    fig1, ax1 = plt.subplots(figsize=(6, 6))
    ax1.imshow(img_sl.T, cmap='gray', origin='lower')
    for contour in measure.find_contours(bnd_sl, 0.5):
        ax1.plot(contour[:, 0], contour[:, 1], color='red', linewidth=2)
    ax1.set_xticks([]); ax1.set_yticks([])
    show_or_save(fig1, 'image.png')

    # Figure 2 — Entropy heatmap
    fig2, ax2 = plt.subplots(figsize=(7, 6))
    im2 = ax2.imshow(ent_sl.T, cmap='viridis', origin='lower', vmin=0, vmax=0.7)
    ax2.set_xticks([]); ax2.set_yticks([])
    plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04).ax.tick_params(labelsize=20)
    plt.tight_layout()
    show_or_save(fig2, 'entropy.png')

    # Figure 3 — Entropy distribution
    fig3, ax3 = plt.subplots(figsize=(7, 6))
    if boundary.sum() > 0:
        ax3.hist(processed_data['entropy'][boundary], bins=30, alpha=0.6,
                 label='Boundary', color='red', density=True, range=(0, 0.7))
    if interior.sum() > 0:
        ax3.hist(processed_data['entropy'][interior], bins=30, alpha=0.6,
                 label='Interior', color='blue', density=True, range=(0, 0.7))
    ax3.set_xlabel('Entropy', fontsize=20); ax3.set_ylabel('Density', fontsize=20)
    ax3.tick_params(labelsize=20); ax3.legend(fontsize=20)
    plt.tight_layout()
    show_or_save(fig3, 'entropydist.png')

    # Figure 4 — MaxLogit heatmap
    fig4, ax4 = plt.subplots(figsize=(7, 6))
    im4 = ax4.imshow(ml_sl.T, cmap='viridis', origin='lower', vmin=0, vmax=10)
    ax4.set_xticks([]); ax4.set_yticks([])
    plt.colorbar(im4, ax=ax4, fraction=0.046, pad=0.04).ax.tick_params(labelsize=20)
    plt.tight_layout()
    show_or_save(fig4, 'maxlogits.png')

    # Figure 5 — MaxLogit distribution
    fig5, ax5 = plt.subplots(figsize=(7, 6))
    if boundary.sum() > 0:
        ax5.hist(processed_data['maxlogit'][boundary], bins=30, alpha=0.6,
                 label='Boundary', color='red', density=True, range=(0, 10))
    if interior.sum() > 0:
        ax5.hist(processed_data['maxlogit'][interior], bins=30, alpha=0.6,
                 label='Interior', color='blue', density=True, range=(0, 10))
    ax5.set_xlabel('MaxLogit', fontsize=20); ax5.set_ylabel('Density', fontsize=20)
    ax5.tick_params(labelsize=20); ax5.legend(fontsize=20)
    plt.tight_layout()
    show_or_save(fig5, 'maxlogitsdist.png')

    # Figure 6 — CT intensity histogram
    fig6, ax6 = plt.subplots(figsize=(7, 6))
    if seg.sum() > 0:
        ax6.hist(image[seg], bins=50, alpha=0.7, color='#1f77b4', range=(-400, 400))
    ax6.set_xlabel('CT Intensity (HU)', fontsize=20)
    ax6.set_ylabel('Count', fontsize=20)
    ax6.tick_params(labelsize=20)
    plt.tight_layout()
    show_or_save(fig6, 'intensity.png')


def visualize_contours_overlay(file_of_i, image_slice, best_slice,
                                axis=2, orient='lung',
                                model_dirs=None, save_path=None):
    """Draw all models' contours on one image (overlay mode).

    Args:
        file_of_i: filename like 'R01-114.nii.gz'
        image_slice: (H, W) 2D CT slice (already extracted, oriented, normalized)
        best_slice: slice index used to extract segmentation slices
        axis: depth axis of the segmentation volumes (matches dataset config)
        orient: orientation tag applied to each segmentation slice
        model_dirs: dict {model_name: nii_dir}; defaults to MODULE-LEVEL MODEL_DIRS
        save_path: if set, save PNG here; else plt.show()
    """
    if model_dirs is None:
        model_dirs = MODEL_DIRS

    colors = sns.color_palette("colorblind", n_colors=len(model_dirs))

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(image_slice.T, cmap='gray', origin='lower')

    model_items = list(model_dirs.items())
    for idx, (model_name, model_dir) in reversed(list(enumerate(model_items))):
        seg_path = osp.join(model_dir, file_of_i)
        if not osp.exists(seg_path):
            print(f"  Skipping {model_name}: {seg_path} not found")
            continue
        seg_vol = nii.load(seg_path).get_fdata()
        seg_sl  = _orient_slice(_get_slice(seg_vol, best_slice, axis), orient)
        if seg_sl.sum() > 0:
            for c in measure.find_contours(seg_sl.T, 0.5):
                ax.plot(c[:, 1], c[:, 0], linewidth=4, color=colors[idx], label=model_name)
        else:
            print(f"  {model_name}: no segmentation in slice {best_slice}")

    ax.axis('off')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=600, bbox_inches='tight')
        print(f"  Saved: {save_path}")
    else:
        plt.show()
    plt.close()


def visualize_contours_permodel(file_of_i, image_slice, best_slice,
                                 axis=2, orient='lung',
                                 model_dirs=None, save_dir=None):
    """Save one PNG per model, each showing its contour on the same slice.

    Args:
        file_of_i: filename like 'R01-114.nii.gz'
        image_slice: (H, W) 2D CT slice (already extracted, oriented, normalized)
        best_slice: slice index
        axis: depth axis of the segmentation volumes (matches dataset config)
        orient: orientation tag applied to each segmentation slice
        model_dirs: dict {model_name: nii_dir}; defaults to MODEL_DIRS
        save_dir: directory to write PNGs to; show interactively if None
    """
    if model_dirs is None:
        model_dirs = MODEL_DIRS

    colors = sns.color_palette("colorblind", n_colors=len(model_dirs))
    stem   = file_of_i.replace('.nii.gz', '')

    for idx, (model_name, model_dir) in enumerate(model_dirs.items()):
        seg_path = osp.join(model_dir, file_of_i)
        if not osp.exists(seg_path):
            print(f"  Skipping {model_name}: {seg_path} not found")
            continue

        seg_vol = nii.load(seg_path).get_fdata()
        seg_sl  = _orient_slice(_get_slice(seg_vol, best_slice, axis), orient)

        fig, ax = plt.subplots(figsize=(10, 10))
        ax.imshow(image_slice.T, cmap='gray', origin='lower')

        if seg_sl.sum() > 0:
            for c in measure.find_contours(seg_sl.T, 0.5):
                ax.plot(c[:, 1], c[:, 0], linewidth=4,
                        color=colors[idx], label=model_name)
        else:
            print(f"  {model_name}: no segmentation in slice {best_slice}")

        ax.axis('off')
        plt.tight_layout()

        if save_dir:
            fname = f"contour_{model_name.replace(' ', '_')}_{stem}_slice{best_slice}.png"
            out   = osp.join(save_dir, fname)
            plt.savefig(out, dpi=600, bbox_inches='tight')
            print(f"  Saved: {out}")
        else:
            plt.show()
        plt.close()


# ============================================================
# SECTION 4: CLI ENTRYPOINTS
# ============================================================

# --- Dataset configs for global mode ---
GLOBAL_DATASET_CONFIGS = {
    'ID (Radiogenomics)': str(LOGITS_ROOT / 'smit_main' / 'lung_test_LRAD_srcnorm' / 'numpy'),
    'RSNA PE':            str(LOGITS_ROOT / 'smit_farood' / 'lung_test_rsna_srcnorm' / 'numpy'),
    'MIDRC C19':          str(LOGITS_ROOT / 'smit_farood' / 'lung_test_covid19_srcnorm' / 'numpy'),
    'KiTS':               str(LOGITS_ROOT / 'smit_farood' / 'lung_test_kits23_srcnorm' / 'numpy'),
    'Pancreas':           str(LOGITS_ROOT / 'smit_farood' / 'lung_test_pancreas_srcnorm' / 'numpy'),
    'Breast Cancer CT':   str(LOGITS_ROOT / 'smit_farood' / 'lung_test_breastc_srcnorm' / 'numpy'),
    'MIDRC C19+':         str(LOGITS_ROOT / 'smit_farood' / 'lung_test_covid19a_srcnorm' / 'numpy'),
}


def _to_jsonable(value):
    if isinstance(value, dict):
        return {k: _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    return value


def summarize_global_dataset(df):
    if df.empty:
        return {
            "n_scans": 0,
            "maxlogit_mean": np.nan,
            "maxlogit_std": np.nan,
            "maxsoftmax_mean": np.nan,
            "maxsoftmax_std": np.nan,
            "energy_mean": np.nan,
            "energy_std": np.nan,
        }
    return {
        "n_scans": int(len(df)),
        "maxlogit_mean": float(df["maxlogit"].mean()),
        "maxlogit_std": float(df["maxlogit"].std()),
        "maxsoftmax_mean": float(df["maxsoftmax"].mean()),
        "maxsoftmax_std": float(df["maxsoftmax"].std()),
        "energy_mean": float(df["energy"].mean()),
        "energy_std": float(df["energy"].std()),
    }


def write_global_results_json(output_path, payload):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(_to_jsonable(payload), f, indent=2)
    print(f"\nSaved global results JSON to: {output_path}")


def run_global(args):
    """Global analysis entrypoint: dataset stats → statistical tests → OOD metrics."""
    metric = args.metric
    pooling = args.pooling

    selected_configs = GLOBAL_DATASET_CONFIGS
    if args.datasets:
        requested = set(args.datasets)
        selected_configs = {
            name: path for name, path in GLOBAL_DATASET_CONFIGS.items()
            if name == 'ID (Radiogenomics)' or name in requested
        }
        missing = sorted(requested - set(selected_configs.keys()))
        if missing:
            print(f"  WARNING: unknown dataset names ignored: {', '.join(missing)}")

    all_dfs = []
    dataset_summaries = {}
    print(f"\nUsing pooling strategy: {pooling}")
    for dataset_name, logits_dir in selected_configs.items():
        print(f"\n{'='*80}\nProcessing: {dataset_name}\n{'='*80}")
        df = compute_dataset_statistics(logits_dir, dataset_name,
                                        min_component_size=args.min_component_size,
                                        n_jobs=args.n_jobs,
                                        pooling=pooling)
        all_dfs.append(df)
        dataset_summaries[dataset_name] = summarize_global_dataset(df)
        print_summary_statistics(df)

    id_df   = all_dfs[0]
    ood_dfs = all_dfs[1:]

    perform_statistical_tests(id_df, ood_dfs)
    ood_results = compute_all_ood_metrics(id_df, ood_dfs, metric_name=metric)

    print(f"\nRunning bootstrap (n={args.n_bootstrap}, seed={args.seed}) ...")
    bootstrap_results = compute_all_ood_metrics_bootstrap(
        id_df, ood_dfs,
        metric_name=metric,
        n_bootstrap=args.n_bootstrap,
        sample_fraction=args.sample_fraction,
        random_seed=args.seed,
    )

    if args.save_json:
        dataset_part = "all" if not args.datasets else "_".join(
            name.lower().replace(" ", "_").replace("+", "plus").replace("-", "minus")
            for name in args.datasets
        )
        json_path = Path(args.json_path) if args.json_path else (
            LOGIT_BASELINES_RESULTS_ROOT / "global" /
            f"global_{metric}_{pooling}_{dataset_part}.json"
        )
        write_global_results_json(
            json_path,
            {
                "metric": metric,
                "pooling": pooling,
                "selected_datasets": [name for name in selected_configs.keys()],
                "n_bootstrap": int(args.n_bootstrap),
                "sample_fraction": float(args.sample_fraction),
                "seed": int(args.seed),
                "min_component_size": int(args.min_component_size),
                "n_jobs": int(args.n_jobs),
                "dataset_summaries": dataset_summaries,
                "ood_metrics": ood_results,
                "ood_metrics_bootstrap": bootstrap_results,
            },
        )


def run_local(args):
    """Local visualization entrypoint: slice selection + contour / heatmap output."""
    cfg = DATASET_CONFIGS.get(args.dataset)
    if cfg is None:
        raise ValueError(f"Unknown dataset '{args.dataset}'. "
                         f"Choose from: {list(DATASET_CONFIGS.keys())}")

    # Warn early if figures will be silently dropped
    if not args.save and not _has_display():
        print("WARNING: No display detected and --save not set. "
              "Figures will not be shown. Re-run with --save to write PNGs to disk.")
        return

    file_of_i   = args.case if args.case.endswith('.nii.gz') else args.case + '.nii.gz'
    logits_path = osp.join(cfg['logits_dir'], file_of_i.replace('.nii.gz', '.npy'))
    image_path  = osp.join(cfg['image_dir'],  file_of_i)
    axis        = cfg['axis']
    orient      = cfg['orient']
    min_cs      = args.min_component_size

    missing = [p for p in (logits_path, image_path) if not osp.isfile(p)]
    if missing:
        for p in missing:
            print(f'ERROR: file not found: {p}')
        return

    print(f"Processing: {file_of_i}  (min_component_size={min_cs})")
    x = process_numpy_logits(logits_path, min_component_size=min_cs)
    best_slice = find_best_slice(x, min_seg_voxels=50, axis=axis)
    if best_slice is None:
        print("No suitable slice found — exiting.")
        return

    image_vol = nii.load(image_path).get_fdata()
    raw_slice = _get_slice(image_vol, best_slice, axis)
    img_slice = _orient_slice(_normalize_ct(raw_slice), orient)

    stem     = file_of_i.replace('.nii.gz', '')
    if args.save:
        ensure_output_dirs()
        save_dir = str(LOGIT_BASELINES_RESULTS_ROOT / args.dataset / stem)
    else:
        save_dir = None

    mode = args.mode
    if mode == 'heatmap':
        visualize_heatmaps(x, image_vol, best_slice,
                           orient=orient, axis=axis, save_dir=save_dir)
    elif mode == 'overlay':
        out = osp.join(save_dir, f'overlay_{stem}_slice{best_slice}.png') if save_dir else None
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
        visualize_contours_overlay(file_of_i, img_slice, best_slice,
                                   axis=axis, orient=orient, save_path=out)
    elif mode == 'permodel':
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
        visualize_contours_permodel(file_of_i, img_slice, best_slice,
                                    axis=axis, orient=orient, save_dir=save_dir)
    else:
        raise ValueError(f"Unknown mode '{mode}'. Choose: heatmap | overlay | permodel")


def run_local_batch(args):
    """Batch local heatmap mode: iterate all cases in a dataset directory."""
    cfg = DATASET_CONFIGS.get(args.dataset)
    if cfg is None:
        raise ValueError(f"Unknown dataset '{args.dataset}'.")

    if not _has_display():
        print("NOTE: No display detected — figures will be saved to disk.")

    image_dir = cfg['image_dir']
    if not osp.isdir(image_dir):
        print(f"ERROR: image directory not found: {image_dir}")
        return

    min_cs = args.min_component_size

    for fname in sorted(os.listdir(image_dir)):
        if not fname.endswith('.nii.gz'):
            continue
        logits_path = osp.join(cfg['logits_dir'], fname.replace('.nii.gz', '.npy'))
        if not osp.exists(logits_path):
            print(f"  No logits for {fname}, skipping.")
            continue

        print(f"\n--- {fname} ---")
        try:
            x = process_numpy_logits(logits_path, min_component_size=min_cs)
            best_slice = find_best_slice(x, min_seg_voxels=50, axis=cfg['axis'])
            if best_slice is None:
                continue
            image_vol = nii.load(osp.join(image_dir, fname)).get_fdata()
            stem      = fname.replace('.nii.gz', '')
            ensure_output_dirs()
            save_dir  = str(LOGIT_BASELINES_RESULTS_ROOT / args.dataset / stem)
            visualize_heatmaps(x, image_vol, best_slice,
                               orient=cfg['orient'], axis=cfg['axis'],
                               save_dir=save_dir)
            print(f"  ✓ Saved to {save_dir}/")
        except Exception as e:
            print(f"  ERROR: {e}")


# ============================================================
# SECTION 5: MAIN
# ============================================================

def build_parser():
    parser = argparse.ArgumentParser(
        description="Spatial uncertainty OOD analysis for medical image segmentation.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python logit_baselines.py global --metric maxlogit
  python logit_baselines.py global --metric energy --n-bootstrap 200 --seed 42

  python logit_baselines.py local --case R01-114 --dataset lung --mode overlay --save
  python logit_baselines.py local --case AMC-020  --dataset lung2 --mode permodel --save
  python logit_baselines.py local --case R01-114  --dataset lung  --mode heatmap

  python logit_baselines.py batch --dataset pancreas
""")

    sub = parser.add_subparsers(dest='command', required=True)

    # --- global ---
    p_global = sub.add_parser('global', help='Dataset-level OOD analysis')
    p_global.add_argument('--metric', default='maxlogit',
                          choices=['maxlogit', 'maxsoftmax', 'energy'],
                          help='OOD score metric (default: maxlogit)')
    p_global.add_argument('--n-bootstrap', type=int, default=100,
                          help='Bootstrap iterations (default: 100)')
    p_global.add_argument('--sample-fraction', type=float, default=1.0,
                          help='Fraction of data per bootstrap sample (default: 1.0)')
    p_global.add_argument('--seed', type=int, default=42,
                          help='Random seed (default: 42)')
    p_global.add_argument('--min-component-size', type=int, default=300,
                          dest='min_component_size',
                          help='Min voxels per connected component (default: 300)')
    p_global.add_argument('--n-jobs', type=int, default=4,
                          help='Worker processes for per-scan logit processing (default: 4)')
    p_global.add_argument('--datasets', nargs='+', default=None,
                          help='Optional OOD datasets to include; ID is always included.')
    p_global.add_argument('--pooling', default='mean',
                          choices=['mean', 'max', 'p95', 'median'],
                          help='Pooling strategy over foreground voxels (default: mean)')
    p_global.add_argument('--save-json', action='store_true',
                          help='Write global results to a JSON file.')
    p_global.add_argument('--json-path', default=None,
                          help='Optional explicit output path for the global results JSON.')

    # --- local ---
    p_local = sub.add_parser('local', help='Single-case visualization')
    p_local.add_argument('--case', required=True,
                         help='Case filename stem, e.g. R01-114 or R01-114.nii.gz')
    p_local.add_argument('--dataset', required=True,
                         choices=list(DATASET_CONFIGS.keys()),
                         help='Dataset name')
    p_local.add_argument('--mode', default='overlay',
                         choices=['overlay', 'permodel', 'heatmap'],
                         help='Visualization mode (default: overlay)')
    p_local.add_argument('--save', action='store_true',
                         help='Save outputs to results/logit_baselines/<dataset>/<case>/')
    p_local.add_argument('--min-component-size', type=int, default=10,
                         dest='min_component_size',
                         help='Min voxels per connected component (default: 10)')

    # --- batch ---
    p_batch = sub.add_parser('batch', help='Batch heatmap generation for a whole dataset')
    p_batch.add_argument('--dataset', required=True,
                         choices=list(DATASET_CONFIGS.keys()),
                         help='Dataset name')
    p_batch.add_argument('--min-component-size', type=int, default=10,
                         dest='min_component_size',
                         help='Min voxels per connected component (default: 10)')

    return parser


if __name__ == '__main__':
    parser = build_parser()
    args   = parser.parse_args()

    if args.command == 'global':
        run_global(args)
    elif args.command == 'local':
        run_local(args)
    elif args.command == 'batch':
        run_local_batch(args)
