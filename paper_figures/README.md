# Paper Figures

`paper_figures/` contains figure-generation scripts used for manuscript, rebuttal, and presentation assets derived from experiment outputs.

Only reusable figure pipelines should live here.

Representative scripts:

- `viz_logit_pooling.py`: compare voxelwise logit pooling baselines
- `scanner_performance.py`: scanner/vendor/kernel performance plots
- `ood_detection_panels.py`: summary OOD panel figures from tabular outputs
- `viz_ablations.py`: ablation summary figures (renders final paper figures from frozen results)
- `viz_shap_deepfeatures_external.py`: SHAP analysis for external test-only cohorts

These scripts generally read from `results/`, `radiomics_features/`, `pickle_data/`, `excelrecords/`, or `metadata_info/`, and write rendered assets to `figures_tmlr/` or related figure output locations.
