# AGENTS.md

Guidance for agentic AI tools (Claude Code, Codex, Cursor, Aider, etc.) working in this repository.

## What this repo is

Research code for **RF-Deep**: a post-hoc out-of-distribution detector for 3D CT lung tumor segmentation. It extracts hierarchical deep features from predicted tumor regions, feeds them to a random forest (with lightweight Mahalanobis and logit baselines for comparison), and evaluates near-OOD / far-OOD scan-level detection across ~2k CT scans.

Paper: [arXiv:2512.08216](https://arxiv.org/abs/2512.08216).

## Start here (read these first)

- [`README.md`](README.md) — user-facing overview, install, typical workflow
- [`PROJECT_LAYOUT.md`](PROJECT_LAYOUT.md) — canonical directory layout and output policy
- [`CODE_REFERENCE.md`](CODE_REFERENCE.md) — module-by-module, function-by-function map of every tracked Python file

`CODE_REFERENCE.md` is the single most useful file for an agent: every module and top-level function has a one-line description. Grep it instead of reading source when locating functionality.

## Mental model

```
data/ (local)                      │
    │                              │ 1. extract_features.py
    ▼                              │    → pickle_data/*.pkl
jsons/*.json  (manifests)          │
    │                              │ 2. ood_rfdeep.py / ood_maha.py / logit_baselines.py
    │                              │    → results/**/*.json
    │                              │
    │                              │ 3. paper_figures/*.py
    ▼                              │    → figures_tmlr/*
metadata_info/ (scanner tables)    │
```

Three-stage pipeline:

1. **Extract** deep-feature vectors from segmentation backbones (SMIT / SwinUNETR). Outputs one pickle per `{model, img_size}` to `pickle_data/`.
2. **Score** — train an OOD detector on ID feature pickles, evaluate on OOD pickles. Multiple methods share one feature cache.
3. **Render** paper figures from the scored JSON results.

Every stage reads/writes project-local directories resolvable via [`project_paths.py`](project_paths.py) (see "Paths" below).

## Entry points

| File | Purpose |
|---|---|
| [`extract_features.py`](extract_features.py) | Feature extraction from segmentation backbones → `pickle_data/` |
| [`ood_rfdeep.py`](ood_rfdeep.py) | Main RF-Deep experiments (unified / separate / ensemble / lodo) |
| [`ood_maha.py`](ood_maha.py) | Mahalanobis deep-feature baseline (+ optional ReAct/ASH) |
| [`ood_metadata_holdout.py`](ood_metadata_holdout.py) | Metadata-stratified holdout (manufacturer / kernel / contrast) |
| [`logit_baselines.py`](logit_baselines.py) | Voxelwise logit OOD baselines (MaxLogit / Energy / MSP / Entropy) |
| [`roi_logit_baselines.py`](roi_logit_baselines.py) | ROI-restricted variant using RF-Deep's crop protocol |
| [`segmentation_inference.py`](segmentation_inference.py) | Run a segmentation backbone over a datalist |
| [`scripts/`](scripts) | Operational helpers (`python -m scripts.<name>`) |
| [`paper_figures/`](paper_figures) | Figure rendering (`python -m paper_figures.<name>`) |

Smoke test: `python -m scripts.smoke_check` — checks paths and imports.

## Paths and configuration

All paths resolve through [`project_paths.py`](project_paths.py). **Do not hardcode paths.** Every root is overridable by an environment variable of the same name:

- `DATA_ROOT` (default `./data`) — datasets; per-dataset roots (`LRAD_ROOT`, `RSNA_ROOT`, `COVID19_ROOT`, `KITS23_ROOT`, `PANCREAS_ROOT`, `BREASTC_ROOT`, `NSCLC_ROOT`) each fall back under `DATA_ROOT` by default
- `RESULTS_ROOT` / `ANALYSIS_RESULTS_ROOT` / `LOGIT_BASELINES_RESULTS_ROOT` — scored outputs
- `PICKLE_ROOT` — feature caches
- `JSONS_ROOT` — dataset manifests
- `METADATA_ROOT` — scanner / radiomics metadata
- `FIGURES_ROOT` — rendered paper figures
- `EXCEL_RECORDS_ROOT`, `RADIOMICS_FEATURES_ROOT` — CSV exports
- `MODELS_ROOT` / `FINETUNED_WEIGHTS_ROOT` / `PRETRAINED_WEIGHTS_ROOT` — checkpoints

Manifests under `jsons/` use tagged paths like `"DATA_ROOT::dataset/images/foo.nii.gz"` resolved at load time by `resolve_manifest_path()`. Prefer this over absolute paths when editing manifests.

## Conventions

- **Python 3.9.** Dependencies pinned in [`requirements.txt`](requirements.txt); MONAI, pyCERR, and DeepMind surface-distance come from pinned Git commits.
- **Naming:** feature pickles are `{model}_size{n}_featvec.pkl`; radiomics CSVs are `{model}_{dataset}_src.csv`; manifest JSONs are `{model}_{dataset}_src.json`.
- **Model flag:** `--model-name` / `--model` accepts `smit`, `mim`, `ibot`, `smitmini`, `swinunetr`, `swinunetr_10k`.
- **Crop modes** (`--crop-mode`): `anchored` (default), `spatial`, `center`. Supported **only** in `extract_features.py` and `ood_rfdeep.py`; other scorers always use the default anchored pickle.
- **Concurrency:** most scorers use `joblib.Parallel` for repeated seeds and `concurrent.futures.ProcessPoolExecutor` for per-scan work.
- **Determinism:** repeated-run scripts take `--iterations` and seed internally from a base seed.

## What is and is NOT in the tracked tree

**Tracked:**
- All Python source, shared scripts, paper-figure code
- `metadata_info/` (scanner tables, pyCERR IBSI1 config, radiomics column mapping)
- All directory `README.md`s
- `docs/readme_assets/` (figure images for the top-level README)

**Intentionally ignored** (local-only or generated):
- `data/` (symlink or local mount)
- `pickle_data/`, `radiomics_features/`, `excelrecords/`, `results/`, `jsons/` (contents — the `README.md` in each is tracked)
- `figures_tmlr/`, `figures_tmlr_rebuttal/`
- `models/finetuned_weights/`, `models/pretrained_weights/`
- `*.pth`, `*.pt`, `__pycache__/`, `.DS_Store`

If an agent needs to add a generated artifact, it belongs under one of the ignored roots, not in tracked source.

## Common pitfalls

- **Don't hardcode absolute paths or usernames.** Use `project_paths.py` + environment variables.
- **Don't add new top-level Python files** unless they're genuinely reusable pipelines. One-off analyses belong in downstream local tooling, not in the shared tree.
- **Manifest paths use the `ROOT::relative` tagged form** — naive `os.path.join(DATA_ROOT, path)` will double-prefix. Always go through `resolve_manifest_path()` / `load_manifest_entries()`.
- **`pickle_data/*.pkl` files are large.** Never suggest committing them.
- **Backbones expose `forward_debug()`** — any new feature-extracting model needs this to plug into `extract_features.py`.

## When making changes

- Update [`CODE_REFERENCE.md`](CODE_REFERENCE.md) whenever you add, rename, or remove a top-level function.
- Keep per-directory `README.md` files in sync with their contents (list of producers/consumers, naming conventions).
- Prefer relative Markdown links (`[foo](path/to/foo.py)`) — absolute paths break on github.com.
- Run `python -m scripts.smoke_check` after structural changes.
