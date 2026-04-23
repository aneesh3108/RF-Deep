"""
extract_features.py — Extract per-scan deep feature vectors for OOD analysis.

Each registered model is validated at startup before any GPU/IO work begins:
  - checkpoint file exists
  - all required dataset JSON files exist
  - forward_debug() returns a list of tensors with expected channel count

Usage:
  python extract_features.py --model smit
  python extract_features.py --model ibot
  python extract_features.py --model mim
  python extract_features.py --model smitmini
  python extract_features.py --model swinunetr_10k
  python extract_features.py --model swinunetr
  python extract_features.py --model smit --json-dir path/to/jsons  # override JSON dir

Crop modes (--crop-mode):
    anchored  : tumor-positive crops only (pos=1, neg=0)  [DEFAULT, original method]
    spatial   : random spatial crops, no label guidance
    center    : single center crop, no label guidance (not sample-matched)

Output pickle is tagged with the crop mode (anchored keeps the original filename):
  smit_size128_featvec.pkl
  smit_size128_featvec_spatial.pkl
  smit_size128_featvec_center.pkl
"""

import argparse
import os
import os.path as osp
import pickle
import random
import sys
from pathlib import Path
import numpy as np
import torch
from monai import data, transforms
from tqdm import tqdm

import ml_collections
from models import smit, swinunetr
from project_paths import FINETUNED_WEIGHTS_ROOT, JSONS_ROOT, PICKLE_ROOT, load_manifest_entries


# ---------------------------------------------------------------------------
# Model registry
#
# Each entry: (backbone_family, run_folder, img_size, fallback_depth)
#
#   backbone_family : selects config + model class in build_model()
#   run_folder      : relative path under models/finetuned_weights/ containing model_final.pt
#   img_size        : (sx, sy, sz) spatial size for transforms + model
#   fallback_depth  : sz used by SafeRandCropByPosNegLabeld fallback
#
# JSON files are expected at {json_dir}/{model_name}_{dataset}_src.json.
# If a model's JSONs don't exist yet, validate_model() will report exactly
# which files are missing so you know what to create before running.
# ---------------------------------------------------------------------------
MODEL_REGISTRY = {
    "smit":          ("smit",      "lung_smit_dice_sq",      (128, 128, 128), 128),
    "ibot":          ("smit",      "lung_ibot_dice_sq",      (128, 128, 128), 128),
    "mim":           ("smit",      "lung_mim_dice_sq",       (128, 128, 128), 128),
    "smitmini":      ("smitmini",  "lung_smitmini_dice_sq",  (96,  96,  96),  128),
    "swinunetr_10k": ("swinunetr", "lung_swin10k_dice_sq",   (96,  96,  96),  96),
    "swinunetr":     ("swinunetr", "lung_swinunetr_dice_sq", (96,  96,  96),  96),
}

CROP_MODES = ["anchored", "spatial", "center"]

DEFAULT_DATASET_ORDER = ["lrad", "rsna", "covid19", "covid19a", "kits23", "pancreas", "breastc"]


def resolve_model_json_path(model_name, dataset, json_dir):
    """Resolve a dataset manifest path for one model and dataset."""
    return Path(json_dir) / f"{model_name}_{dataset}_src.json"


def discover_datasets(model_name, json_dir):
    """Discover available datasets for a model from manifest filenames."""
    json_dir_path = Path(json_dir)
    discovered = []
    suffix = "_src.json"
    prefix = f"{model_name}_"
    for json_path in json_dir_path.glob(f"{model_name}_*_src.json"):
        name = json_path.name
        if not (name.startswith(prefix) and name.endswith(suffix)):
            continue
        discovered.append(name[len(prefix) : -len(suffix)])

    preferred = [dataset for dataset in DEFAULT_DATASET_ORDER if dataset in discovered]
    extras = sorted(dataset for dataset in discovered if dataset not in DEFAULT_DATASET_ORDER)
    return preferred + extras


def resolve_datasets(model_name, json_dir, requested_datasets):
    """Resolve the dataset list for a run."""
    available = discover_datasets(model_name, json_dir)
    if requested_datasets:
        missing = [dataset for dataset in requested_datasets if dataset not in available]
        if missing:
            print("ERROR: requested dataset manifest(s) not found:")
            for dataset in missing:
                print(f"  {Path(json_dir) / f'{model_name}_{dataset}_src.json'}")
            sys.exit(1)
        return requested_datasets

    if not available:
        print(f"ERROR: no JSON datalist files found for model '{model_name}' under {json_dir}")
        sys.exit(1)

    return available


# ---------------------------------------------------------------------------
# Startup validation
# ---------------------------------------------------------------------------
def validate_model(model_name, json_dir, datasets):
    """Check all preconditions before doing any real work.

    Validates:
      1. Checkpoint file exists under models/finetuned_weights/{run_folder}/model_final.pt
      2. All dataset JSON files exist under {json_dir}/{model_name}_{dataset}_src.json
      3. forward_debug() returns a non-empty list of tensors on a dummy input,
         and reports the expected concatenated channel count.

    Args:
        model_name: key into MODEL_REGISTRY
        json_dir: directory that should contain the JSON datalist files

    Returns:
        expected_channels (int): total channels in concatenated feature stack,
                                 used later to sanity-check real outputs.

    Raises:
        SystemExit if any hard precondition fails (missing files, bad interface).
    """
    family, run_folder, img_size, _ = MODEL_REGISTRY[model_name]
    errors = []

    # --- 1. Checkpoint ---
    ckpt = FINETUNED_WEIGHTS_ROOT / run_folder / "model_final.pt"
    if not ckpt.is_file():
        errors.append(f"  Checkpoint not found:  {ckpt}")

    # --- 2. JSON files ---
    missing_jsons = []
    for dataset in datasets:
        json_path = resolve_model_json_path(model_name, dataset, json_dir)
        if not json_path.is_file():
            missing_jsons.append(str(json_path))
    if missing_jsons:
        errors.append("  Missing JSON datalist files:")
        for p in missing_jsons:
            errors.append(f"    {p}")

    if errors:
        print(f"\nERROR: cannot run --model {model_name}:")
        for e in errors:
            print(e)
        if missing_jsons:
            print(
                "\n  Tip: create the missing JSON files (same format as "
                f"{JSONS_ROOT}/smitmini_*_src.json) and re-run."
            )
        sys.exit(1)

    # --- 3. forward_debug() interface dry-run ---
    # Build model on CPU with a tiny dummy input to verify the interface
    # before committing to loading real data.
    print(f"  Validating forward_debug() interface for {model_name}...")
    try:
        model = build_model(model_name)
        dummy = torch.zeros(1, 1, *img_size)
        with torch.no_grad():
            feats = model.forward_debug(dummy)

        if not isinstance(feats, (list, tuple)) or len(feats) == 0:
            print(
                f"ERROR: {model_name}.forward_debug() must return a non-empty "
                f"list/tuple of tensors, got: {type(feats)}"
            )
            sys.exit(1)

        for i, f in enumerate(feats):
            if not isinstance(f, torch.Tensor):
                print(
                    f"ERROR: forward_debug() element {i} is {type(f)}, expected Tensor"
                )
                sys.exit(1)

        expected_channels = sum(f.shape[1] for f in feats)
        print(
            f"  OK — {len(feats)} feature levels, "
            f"{expected_channels} total channels after concat"
        )
        return expected_channels, model

    except AttributeError:
        print(
            f"ERROR: {model_name} does not expose forward_debug(). "
            "Add it to the model class before registering this model."
        )
        sys.exit(1)
    except Exception as exc:
        print(f"ERROR: forward_debug() dry-run failed for {model_name}: {exc}")
        sys.exit(1)


# ---------------------------------------------------------------------------
# Safe crop wrapper
# ---------------------------------------------------------------------------
class SafeRandCropByPosNegLabeld(transforms.RandomizableTransform):
    def __init__(self, rand_crop_params, fallback_crop_params):
        super().__init__()
        self.rand_crop     = transforms.RandCropByPosNegLabeld(**rand_crop_params)
        self.fallback_crop = transforms.RandCropByPosNegLabeld(**fallback_crop_params)

    def __call__(self, data_dict):
        try:
            return self.rand_crop(data_dict)
        except Exception as exc:
            print(f"RandCropByPosNegLabeld failed: {exc}")
            print("Applying fallback: Original Size")
            return self.fallback_crop(data_dict)


# ---------------------------------------------------------------------------
# Configs
# ---------------------------------------------------------------------------
def build_smit_large_config(img_size):
    """Full SMIT: depths=(2,2,12,2) — used by smit / ibot / mim."""
    config = ml_collections.ConfigDict()
    config.if_transskip   = True
    config.if_convskip    = True
    config.patch_size     = 2
    config.in_chans       = 1
    config.embed_dim      = 48
    config.depths         = (2, 2, 12, 2)
    config.num_heads      = (4, 4, 8, 16)
    config.window_size    = (4, 4, 4)
    config.mlp_ratio      = 4
    config.pat_merg_rf    = 4
    config.qkv_bias       = True
    config.drop_rate      = 0
    config.drop_path_rate = 0.3
    config.ape            = False
    config.spe            = False
    config.patch_norm     = True
    config.use_checkpoint = False
    config.out_indices    = (0, 1, 2, 3)
    config.reg_head_chan  = 16
    config.img_size       = img_size
    return config


def build_smit_small_config(img_size):
    """SMIT-Lite: depths=(2,2,2,2) — used by smitmini."""
    config = ml_collections.ConfigDict()
    config.if_transskip   = True
    config.if_convskip    = True
    config.patch_size     = 2
    config.in_chans       = 1
    config.embed_dim      = 48
    config.depths         = (2, 2, 2, 2)
    config.num_heads      = (3, 6, 12, 24)
    config.window_size    = (4, 4, 4)
    config.mlp_ratio      = 4
    config.pat_merg_rf    = 4
    config.qkv_bias       = True
    config.drop_rate      = 0
    config.drop_path_rate = 0.3
    config.ape            = False
    config.spe            = False
    config.patch_norm     = True
    config.use_checkpoint = False
    config.out_indices    = (0, 1, 2, 3)
    config.reg_head_chan  = 16
    config.img_size       = img_size
    return config


# ---------------------------------------------------------------------------
# Model builder
# ---------------------------------------------------------------------------
def build_model(model_name):
    """Instantiate and return the model for model_name (weights loaded, eval mode)."""
    family, run_folder, img_size, _ = MODEL_REGISTRY[model_name]
    checkpoint_path = FINETUNED_WEIGHTS_ROOT / run_folder / "model_final.pt"
    state_dict = torch.load(checkpoint_path, map_location="cpu")["state_dict"]

    if family == "smit":
        config = build_smit_large_config(img_size)
        model  = smit.SMIT_3D_Seg(config, out_channels=2)
        model.load_state_dict(state_dict)
    elif family == "smitmini":
        config = build_smit_small_config(img_size)
        model  = smit.SMIT_3D_Seg(config, out_channels=2)
        model.load_state_dict(state_dict)
    elif family == "swinunetr":
        model = swinunetr.SwinUNETR(
            img_size=img_size,
            in_channels=1,
            out_channels=2,
            feature_size=48,
            drop_rate=0.0,
            attn_drop_rate=0.0,
        )
        model.load_state_dict(state_dict, strict=True)
    else:
        raise ValueError(f"Unknown backbone family: {family}")

    model.eval()
    return model


# ---------------------------------------------------------------------------
# Transforms
# ---------------------------------------------------------------------------
def build_transforms(sx, sy, sz, fallback_depth, crop_mode="anchored"):
    if crop_mode == "center":
        # Center crop — no label needed
        common = [
            transforms.Spacingd(keys=["image"], pixdim=(1.0, 1.0, 1.0), mode="bilinear"),
            transforms.ScaleIntensityRanged(
                keys=["image"], a_min=-400, a_max=400, b_min=0, b_max=1, clip=True,
            ),
            transforms.CropForegroundd(keys=["image"], source_key="image"),
            transforms.SpatialPadd(keys=["image"], spatial_size=(sx, sy, sz)),
            transforms.CenterSpatialCropd(keys=["image"], roi_size=(sx, sy, sz)),
            transforms.ToTensord(keys=["image"]),
        ]
        val_transform = transforms.Compose(
            [
                transforms.LoadImaged(keys=["image"]),
                transforms.AddChanneld(keys=["image"]),
                transforms.Orientationd(keys=["image"], axcodes="RAS"),
                *common,
            ]
        )
        val_transform_rsna = transforms.Compose(
            [
                transforms.LoadImaged(keys=["image"]),
                transforms.AddChanneld(keys=["image"]),
                transforms.Orientationd(keys=["image"], axcodes="PLI"),
                *common,
            ]
        )
        return val_transform, val_transform_rsna

    if crop_mode == "spatial":
        # Random spatial crops — no label guidance, matched to num_samples=8
        common = [
            transforms.Spacingd(keys=["image"], pixdim=(1.0, 1.0, 1.0), mode="bilinear"),
            transforms.ScaleIntensityRanged(
                keys=["image"], a_min=-400, a_max=400, b_min=0, b_max=1, clip=True,
            ),
            transforms.CropForegroundd(keys=["image"], source_key="image"),
            transforms.SpatialPadd(keys=["image"], spatial_size=(sx, sy, sz)),
            transforms.RandSpatialCropSamplesd(
                keys=["image"],
                roi_size=(sx, sy, sz),
                num_samples=8,
                random_center=True,
                random_size=False,
            ),
            transforms.ToTensord(keys=["image"]),
        ]
        val_transform = transforms.Compose(
            [
                transforms.LoadImaged(keys=["image"]),
                transforms.AddChanneld(keys=["image"]),
                transforms.Orientationd(keys=["image"], axcodes="RAS"),
                *common,
            ]
        )
        val_transform_rsna = transforms.Compose(
            [
                transforms.LoadImaged(keys=["image"]),
                transforms.AddChanneld(keys=["image"]),
                transforms.Orientationd(keys=["image"], axcodes="PLI"),
                *common,
            ]
        )
        return val_transform, val_transform_rsna

    # Anchored: tumor-positive crops only (pos=1, neg=0)
    pos, neg = 1, 0

    rand_crop_params = {
        "keys": ["image", "label"],
        "label_key": "label",
        "spatial_size": (sx, sy, sz),
        "pos": pos,
        "neg": neg,
        "num_samples": 8,
        "image_key": "image",
        "image_threshold": 0,
    }
    fallback_crop_params = {
        "keys": ["image", "label"],
        "label_key": "label",
        "spatial_size": (sx, sy, fallback_depth),
        "pos": pos,
        "neg": neg,
        "num_samples": 8,
        "image_key": "image",
        "image_threshold": 0,
    }

    common = [
        transforms.Spacingd(
            keys=["image", "label"],
            pixdim=(1.0, 1.0, 1.0),
            mode=("bilinear", "nearest"),
        ),
        transforms.ScaleIntensityRanged(
            keys=["image"], a_min=-400, a_max=400, b_min=0, b_max=1, clip=True,
        ),
        transforms.CropForegroundd(keys=["image", "label"], source_key="image"),
        transforms.SpatialPadd(keys=["image", "label"], spatial_size=(sx, sy, sz)),
        SafeRandCropByPosNegLabeld(rand_crop_params, fallback_crop_params),
        transforms.ToTensord(keys=["image", "label"]),
    ]

    val_transform = transforms.Compose(
        [
            transforms.LoadImaged(keys=["image", "label"]),
            transforms.AddChanneld(keys=["image", "label"]),
            transforms.Orientationd(keys=["image", "label"], axcodes="RAS"),
            *common,
        ]
    )
    val_transform_rsna = transforms.Compose(
        [
            transforms.LoadImaged(keys=["image", "label"]),
            transforms.AddChanneld(keys=["image", "label"]),
            transforms.Orientationd(keys=["image", "label"], axcodes="PLI"),
            *common,
        ]
    )
    return val_transform, val_transform_rsna


# ---------------------------------------------------------------------------
# Output path — single place that defines naming convention
# ---------------------------------------------------------------------------
def get_output_path(model_name, img_size, crop_mode="anchored", output_dir=PICKLE_ROOT):
    sx = img_size[0]
    tag = "" if crop_mode == "anchored" else f"_{crop_mode}"
    return Path(output_dir) / f"{model_name}_size{sx}_featvec{tag}.pkl"


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------
def extract_feature_vectors(
    model, model_name, img_size, json_dir,
    transform_default, transform_rsna,
    expected_channels, datasets, crop_mode="anchored", overwrite_existing=False, output_dir=PICKLE_ROOT,
):
    """Run inference over all datasets and save feature vectors to disk.

    Args:
        model: instantiated model with forward_debug()
        model_name: registry key (used for JSON paths and output filename)
        img_size: (sx, sy, sz) tuple
        json_dir: directory containing {model_name}_{dataset}_src.json files
        transform_default / transform_rsna: MONAI transforms per orientation
        expected_channels: total channels validated at startup; each scan is
                           checked against this so shape drift is caught early
        output_dir: where to write the .pkl file
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    output_path = get_output_path(model_name, img_size, crop_mode, output_dir)

    if output_path.exists():
        with open(output_path, "rb") as file_obj:
            feature_vectors = pickle.load(file_obj)
        print(f"Loaded existing feature vectors from {output_path}")
    else:
        feature_vectors = {}

    for dataset in datasets:
        if dataset in feature_vectors and not overwrite_existing:
            print(f"\nDataset: {dataset} — already exists, skipping")
            continue

        print(f"\nDataset: {dataset}  [crop_mode={crop_mode}]")
        json_path = resolve_model_json_path(model_name, dataset, json_dir)
        datalist = load_manifest_entries(json_path, "validation")

        torch.manual_seed(42)
        np.random.seed(42)
        random.seed(42)

        dataset_transform = transform_rsna if dataset == "rsna" else transform_default
        eval_ds     = data.Dataset(data=datalist, transform=dataset_transform)
        eval_loader = data.DataLoader(eval_ds, batch_size=1, shuffle=False)

        dataset_vectors = {}
        with torch.no_grad():
            for batch_data in tqdm(eval_loader):
                data_x   = batch_data["image"]
                filename = osp.basename(batch_data["image_meta_dict"]["filename_or_obj"][0])

                feats         = model.forward_debug(data_x)
                feature_stack = torch.cat(feats, dim=1)

                # Per-scan shape check — catches silent regressions if
                # forward_debug() changes output structure mid-run
                actual_channels = feature_stack.shape[1]
                if actual_channels != expected_channels:
                    raise RuntimeError(
                        f"Channel mismatch for {osp.basename(filename)}: "
                        f"expected {expected_channels}, got {actual_channels}. "
                        "forward_debug() output structure may have changed."
                    )

                dataset_vectors[filename] = feature_stack.cpu()

        feature_vectors[dataset] = dataset_vectors
        print(f"  {len(dataset_vectors)} scans processed")

    with open(output_path, "wb") as file_obj:
        pickle.dump(feature_vectors, file_obj)

    print(f"\nFeature vectors saved to {output_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def build_parser():
    parser = argparse.ArgumentParser(
        description="Extract deep feature vectors for OOD analysis.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Models and their spatial sizes:
  smit, ibot, mim              128x128x128  (full SMIT, depths 2-2-12-2)
  smitmini                     96x96x96     (SMIT-Lite, depths 2-2-2-2)
  swinunetr_10k, swinunetr     96x96x96     (SwinUNETR)

JSON files required per model:
  {JSONS_ROOT}/{model}_{dataset}_src.json

If the output pickle already exists, selected datasets are refreshed in place
and all other dataset entries are preserved.
""",
    )
    parser.add_argument(
        "--model",
        choices=sorted(MODEL_REGISTRY.keys()),
        required=True,
        help="Model to run.",
    )
    parser.add_argument(
        "--crop-mode",
        choices=CROP_MODES,
        default="anchored",
        dest="crop_mode",
        help="Crop strategy for feature extraction (default: anchored).",
    )
    parser.add_argument(
        "--json-dir",
        default=str(JSONS_ROOT),
        dest="json_dir",
        help=f"Directory containing datalist JSON files (default: {JSONS_ROOT}/)",
    )
    parser.add_argument(
        "--output-dir",
        default=str(PICKLE_ROOT),
        dest="output_dir",
        help=f"Directory to write .pkl output (default: {PICKLE_ROOT}/)",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=None,
        help="Optional dataset list to extract/update. Example: --datasets covid19a breastc",
    )
    parser.add_argument(
        "--overwrite-existing",
        action="store_true",
        help="Overwrite dataset entries that already exist in the output pickle.",
    )
    return parser


def main():
    args = build_parser().parse_args()

    print(f"\n{'='*60}")
    print(f"Model     : {args.model}")
    print(f"Crop mode : {args.crop_mode}")
    print(f"{'='*60}")

    datasets = resolve_datasets(args.model, args.json_dir, args.datasets)
    print(f"Datasets: {', '.join(datasets)}")

    # Validate everything before touching real data
    expected_channels, model = validate_model(args.model, args.json_dir, datasets)

    _, _, img_size, fallback_depth = MODEL_REGISTRY[args.model]
    sx, sy, sz = img_size

    transform_default, transform_rsna = build_transforms(sx, sy, sz, fallback_depth, crop_mode=args.crop_mode)

    extract_feature_vectors(
        model=model,
        model_name=args.model,
        img_size=img_size,
        json_dir=args.json_dir,
        transform_default=transform_default,
        transform_rsna=transform_rsna,
        expected_channels=expected_channels,
        datasets=datasets,
        crop_mode=args.crop_mode,
        overwrite_existing=args.overwrite_existing,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
