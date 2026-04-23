# Results

`results/` is the canonical output root for generated experiment artifacts.

Typical contents:

- `analysis/`: aggregated experiment summaries, JSON exports, and benchmark outputs
- `logit_baselines/`: dataset-level statistics and local visualization payloads for voxelwise logit experiments
- model-specific segmentation outputs under directories such as `*_main` or `*_farood`
- ablation summaries exported as JSON by figure-generation scripts

This directory is intentionally ignored because it is generated, can be large, and is often machine-specific.
