# Project Layout

Canonical directories in this repository:

- `models/`: model definitions and weights
- `paper_figures/`: figure-generation code for paper assets
- `scripts/`: operational and maintenance scripts
- `data/`: expected local dataset mount/symlink root (kept out of git)
- `results/`: analysis outputs, experiment JSON/CSV, and non-paper generated artifacts
- `results/logit_baselines/`: JSON outputs and visual assets for voxelwise logit baselines
- `figures_tmlr/`: rendered paper figures and qualitative figure assets
- `radiomics_features/`: canonical radiomics CSV store
- `jsons/`: dataset split/config JSON files
- `pickle_data/`: cached feature-vector pickles
- `metadata_info/`: shared scanner metadata plus radiomics-support metadata such as IBSI settings and feature-name mappings

Crop modes (`--crop-mode`): anchored (default), spatial, center.

- Supported **only** in `extract_features.py` (feature extraction) and `ood_rfdeep.py` (RF-Deep evaluation).
- `ood_maha.py` and all radiomics-based pipelines do not support `--crop-mode` — they always use the default anchored pkl.

Output policy:

- analysis scripts write to `results/`
- paper figure scripts write to `figures_tmlr/`

Dataset paths:

- local datasets are expected under `data/` by default
- `data/` is intentionally ignored because it is machine-specific and is a symlink on the local system
- dataset roots can also be overridden with environment variables defined in `project_paths.py`
