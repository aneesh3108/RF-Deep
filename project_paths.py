"""
Central path defaults for datasets and project-local outputs.

Override any default with an environment variable of the same name.
Example:
  export LRAD_ROOT="/path/to/TCIA_R01"
"""

from __future__ import annotations

import os
import os.path as osp
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
DATA_ROOT = Path(os.environ.get("DATA_ROOT", str(PROJECT_ROOT / "data"))).expanduser()


def _env_path(name: str, default: Path) -> Path:
    return Path(os.environ.get(name, str(default))).expanduser()


LRAD_R01_ROOT = _env_path("LRAD_R01_ROOT", DATA_ROOT / "TCIA_R01")
LRAD_AMC_ROOT = _env_path("LRAD_AMC_ROOT", DATA_ROOT / "TCIA_AMC")
LRAD_ROOT = _env_path("LRAD_ROOT", DATA_ROOT)
RSNA_ROOT = _env_path("RSNA_ROOT", DATA_ROOT / "RSNA")
COVID19_ROOT = _env_path("COVID19_ROOT", DATA_ROOT / "COVID-19")
COVID19A_ROOT = _env_path("COVID19A_ROOT", DATA_ROOT / "COVID-19-a")
KITS23_ROOT = _env_path("KITS23_ROOT", DATA_ROOT / "KITS_23")
PANCREAS_ROOT = _env_path("PANCREAS_ROOT", DATA_ROOT / "PANCREAS_CT")
BREASTC_ROOT = _env_path("BREASTC_ROOT", DATA_ROOT / "Breast66")
NSCLC_ROOT = _env_path("NSCLC_ROOT", DATA_ROOT / "NSCLC_TCIA")

MODELS_ROOT = _env_path("MODELS_ROOT", PROJECT_ROOT / "models")
FINETUNED_WEIGHTS_ROOT = _env_path(
    "FINETUNED_WEIGHTS_ROOT",
    MODELS_ROOT / "finetuned_weights",
)

RESULTS_ROOT = _env_path("RESULTS_ROOT", PROJECT_ROOT / "results")
LOGITS_ROOT = _env_path("LOGITS_ROOT", RESULTS_ROOT)
ANALYSIS_RESULTS_ROOT = _env_path("ANALYSIS_RESULTS_ROOT", RESULTS_ROOT / "analysis")
LOGIT_BASELINES_RESULTS_ROOT = _env_path(
    "LOGIT_BASELINES_RESULTS_ROOT",
    _env_path("OBJECTIVE_RESULTS_ROOT", RESULTS_ROOT / "logit_baselines"),
)
OBJECTIVE_RESULTS_ROOT = LOGIT_BASELINES_RESULTS_ROOT
PICKLE_ROOT = _env_path("PICKLE_ROOT", PROJECT_ROOT / "pickle_data")
JSONS_ROOT = _env_path("JSONS_ROOT", PROJECT_ROOT / "jsons")
FIGURES_ROOT = _env_path("FIGURES_ROOT", PROJECT_ROOT / "figures_tmlr")
PAPER_FIGURES_ROOT = _env_path("PAPER_FIGURES_ROOT", PROJECT_ROOT / "paper_figures")
EXCEL_RECORDS_ROOT = _env_path("EXCEL_RECORDS_ROOT", PROJECT_ROOT / "excelrecords")
RADIOMICS_FEATURES_ROOT = _env_path("RADIOMICS_FEATURES_ROOT", PROJECT_ROOT / "radiomics_features")
METADATA_ROOT = _env_path("METADATA_ROOT", PROJECT_ROOT / "metadata_info")
RADIOMICS_MAPPING_PATH = _env_path(
    "RADIOMICS_MAPPING_PATH",
    METADATA_ROOT / "radiomics_mapping.csv",
)
PRETRAINED_WEIGHTS_ROOT = _env_path(
    "PRETRAINED_WEIGHTS_ROOT",
    MODELS_ROOT / "pretrained_weights",
)

DATASET_IMAGE_DIRS = {
    "lrad_r01_image": LRAD_R01_ROOT / "image",
    "lrad_r01_label": LRAD_R01_ROOT / "label",
    "lrad_amc_image": LRAD_AMC_ROOT / "image",
    "lrad_amc_label": LRAD_AMC_ROOT / "label",
    "rsna_image": RSNA_ROOT / "train_nii",
    "rsna_label": RSNA_ROOT / "seg_nii",
    "covid19_image": COVID19_ROOT,
    "covid19a_image": COVID19A_ROOT,
    "kits23_image": KITS23_ROOT / "images",
    "pancreas_image": PANCREAS_ROOT,
    "breastc_image": BREASTC_ROOT / "imgs",
    "nsclc_image": NSCLC_ROOT / "image",
}

MANIFEST_ROOTS = {
    "DATA_ROOT": DATA_ROOT,
    "RESULTS_ROOT": RESULTS_ROOT,
    "PROJECT_ROOT": PROJECT_ROOT,
}


def encode_manifest_path(path: Path) -> str:
    """Encode a path relative to a known project root when possible."""
    resolved = Path(path).expanduser()
    for root_name, root_path in MANIFEST_ROOTS.items():
        try:
            relative = resolved.relative_to(root_path)
            return f"{root_name}::{relative.as_posix()}"
        except ValueError:
            continue
    return str(resolved)


def resolve_manifest_path(path_str: str, default_root: Path | None = None) -> Path:
    """Resolve a manifest path supporting both legacy absolute and root-tagged values."""
    if "::" in path_str:
        root_name, relative_path = path_str.split("::", 1)
        root_path = MANIFEST_ROOTS.get(root_name)
        if root_path is None:
            raise ValueError(f"Unknown manifest root '{root_name}' in path '{path_str}'")
        return root_path / Path(relative_path)
    path = Path(path_str).expanduser()
    if path.is_absolute() or default_root is None:
        return path
    return default_root / path


def load_manifest_entries(json_path: Path, split: str) -> list[dict]:
    """Load a manifest split and resolve root-tagged image/label paths."""
    import json

    with Path(json_path).open("r") as handle:
        manifest = json.load(handle)

    entries = []
    for item in manifest[split]:
        resolved_item = dict(item)
        if "image" in resolved_item:
            resolved_item["image"] = str(resolve_manifest_path(resolved_item["image"], default_root=DATA_ROOT))
        if "label" in resolved_item:
            resolved_item["label"] = str(resolve_manifest_path(resolved_item["label"], default_root=RESULTS_ROOT))
        entries.append(resolved_item)
    return entries


def ensure_output_dirs() -> None:
    """Create the main project-local output roots when needed."""
    for path in (
        RESULTS_ROOT,
        ANALYSIS_RESULTS_ROOT,
        OBJECTIVE_RESULTS_ROOT,
        FIGURES_ROOT,
        PAPER_FIGURES_ROOT,
        EXCEL_RECORDS_ROOT,
        JSONS_ROOT,
        PICKLE_ROOT,
        RADIOMICS_FEATURES_ROOT,
    ):
        path.mkdir(parents=True, exist_ok=True)
