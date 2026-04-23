# Code Reference

This file is a lightweight reference for the shared Python code in the repository.
It summarizes what each module, class, and top-level function is responsible for, so new readers can navigate the codebase quickly without reading every implementation first.
It is meant as a developer-facing map of the repo rather than a formal API contract.

## Path and Data Utilities

### `project_paths.py`

- `_env_path`: resolve a `Path` from an environment variable with a default fallback.
- `encode_manifest_path`: convert an absolute path into a tagged manifest path when it lives under a known project root.
- `resolve_manifest_path`: resolve tagged or relative manifest paths into concrete filesystem paths.
- `load_manifest_entries`: load one split from a JSON manifest and resolve its image/label paths.
- `ensure_output_dirs`: create the main project-local output directories when needed.

### `ood_utils.py`

- `_normalize_feature_vector_keys`: normalize feature-vector dictionary keys to stable basenames.
- `load_feature_vectors`: load cached deep-feature pickles for one model and image size.
- `extract_expanded_features`: expand per-scan feature matrices into crop-level feature rows.
- `sample_features`: sample scans and expand them into training or test feature matrices.
- `train_model`: train a random-forest or MLP classifier for OOD experiments.
- `_apply_nan_overrides`: force fallback predictions for scans with unusable radiomics rows.
- `aggregate_predictions_by_filename`: average crop-level predictions back to scan-level predictions.
- `train_and_evaluate`: train on one ID/OOD pairing and compute aggregate metrics.
- `train_and_evaluate2`: variant of the train/evaluate pipeline with per-dataset bookkeeping.
- `compute_metrics`: compute AUROC and FPR95 from binary labels and scores.
- `compute_per_dataset_metrics`: compute scan-level metrics broken out by OOD dataset.
- `compute_95ci_percentile`: compute a percentile bootstrap interval summary.
- `compute_confidence_interval`: compute percentile confidence intervals over repeated-run scores.
- `prepare_datasets`: assemble ID and OOD datasets from cached deep-feature pickles and radiomics CSVs.
- `filter_nan_files`: remove scans known to contain unusable radiomics entries.
- `build_filename_to_dataset_mapping`: map scan basenames back to dataset labels.
- `convert_to_serializable`: recursively convert NumPy-heavy objects into JSON-safe structures.
- `create_splits_for_all_methods`: generate aligned train/test splits reused across methods.

## Core Experiment Modules

### `ood_rfdeep.py`

- `create_splits_for_all_methods`: shared split creation for all RF-Deep evaluation modes.
- `safe_sample`: sample up to a requested number of filenames without exceeding availability.
- `unified_ensemble`: train one classifier on ID versus pooled OOD data.
- `separate_classifiers`: train one classifier per OOD dataset.
- `ensemble_average`: combine dataset-specific models by averaging their scores.
- `lodo`: leave-one-dataset-out training and evaluation.
- `evaluate_test_only_datasets`: score held-out evaluation-only cohorts using trained models.
- `run_single_iteration_unified`: one seeded run for unified training.
- `run_single_iteration_separate`: one seeded run for dataset-specific training.
- `run_single_iteration_ensemble`: one seeded run for ensemble averaging.
- `run_single_iteration_lodo`: one seeded run for LODO evaluation.
- `run_method_parallel`: execute repeated runs of one RF-Deep method in parallel.
- `run_lodo_parallel_special`: execute repeated LODO runs with the extra bookkeeping they require.
- `parse_global_and_per_dataset_results`: split repeated-run outputs into global and per-dataset summaries.
- `parse_separate_results`: summarize the dataset-specific classifier outputs.
- `format_results_global_and_per_dataset`: format repeated-run summaries into a serializable report.
- `format_results_per_dataset_only`: format per-dataset summaries when no global metric is needed.
- `print_results`: render a human-readable summary of experiment outputs.
- `run_experiment`: public experiment entrypoint for RF-Deep workflows.

### `ood_maha.py`

- `MahalanobisDetector`: ID-only OOD detector using squared Mahalanobis distance with optional feature transforms.
- `NoTransform`, `ReActTransform`, `ASHTransform`: feature transforms applied before Mahalanobis scoring.
- `NoNormalization`, `ZScoreNormalization`: optional feature normalization fitted on ID data.
- `make_transform`: factory for feature transforms (`none`, `react`, `ash`).
- `make_normalizer`: factory for normalizers (`none`, `zscore`).
- `create_splits`: produce train/test splits for one filename list.
- `safe_sample`: bounded filename sampler for repeated runs.
- `run_single_iteration`: one repeated-run evaluation (ID + all OOD, global + per-dataset).
- `parse_unified_results`: summarize repeated-run outputs.
- `format_results`: assemble a final serializable results payload.
- `print_results`: render a console summary.
- `run_experiment`: public experiment entrypoint.

### `ood_metadata_holdout.py`

- `_normalize_case_id`: normalize file identifiers into metadata keys.
- `_clean_kernel_label`: normalize reconstruction-kernel labels.
- `load_holdout_metadata`: load metadata annotations used for holdout grouping.
- `build_id_metadata_table`: join ID scan lists with metadata attributes.
- `build_group_specs`: define groupings for manufacturer, contrast, and kernel holdouts.
- `split_ood_datasets`: split OOD pools into train/test partitions for metadata experiments.
- `safe_sample`: bounded sampler used across repeated runs.
- `summarize_series`: summarize repeated metrics into mean, spread, and interval fields.
- `build_filename_mapping_for_holdout`: map filenames to dataset labels for grouped reporting.
- `false_ood_rate`: compute the false-OOD rate above a fixed threshold.
- `evaluate_binary_problem`: score one binary ID versus OOD test problem.
- `train_id_features`: build the ID-side training feature matrix.
- `run_single_iteration_dataset_specific`: one seeded metadata-holdout run for dataset-specific models.
- `run_single_iteration_ensemble`: one seeded metadata-holdout run for ensemble models.
- `run_single_iteration_lodo`: one seeded metadata-holdout run for LODO models.
- `summarize_dataset_results`: summarize repeated-run per-dataset metrics.
- `summarize_ensemble_results`: summarize repeated-run ensemble metrics.
- `run_method_for_group`: run one method across one metadata grouping.
- `convert_to_serializable`: convert nested experiment payloads into JSON-safe types.
- `run_experiment`: top-level metadata holdout experiment entrypoint.
- `parse_args`: CLI argument parser.
- `main`: CLI entrypoint.

## Feature Extraction and Metrics

### `extract_features.py`

- `resolve_model_json_path`: locate the manifest file for one model/dataset pair.
- `discover_datasets`: infer available datasets from manifest filenames.
- `resolve_datasets`: decide which dataset manifests to use for one run.
- `validate_model`: preflight-check checkpoints, manifests, and feature interfaces.
- `SafeRandCropByPosNegLabeld`: crop transform wrapper with a fallback path for difficult scans.
- `build_smit_large_config`: build the large SMIT config used by full-size models.
- `build_smit_small_config`: build the smaller SMIT config used by `smitmini`.
- `build_model`: instantiate and load the requested segmentation backbone.
- `build_transforms`: build the preprocessing and crop transforms used during feature extraction.
- `get_output_path`: compute the pickle output path for one feature-extraction run.
- `extract_feature_vectors`: run feature extraction and persist the resulting pickle.
- `build_parser`: CLI parser.
- `main`: CLI entrypoint.

### `scripts/segmentation_metrics_utils.py`

- `dice`: compute binary Dice overlap.
- `sum_dims`: helper for summing over selected axes.
- `detect_lesions`: match connected components between prediction and reference masks.
- `compute_segmentation_scores`: compute lesion-wise segmentation metrics, including overlap and distance-based scores.

## Logit and ROI Baselines

### `logit_baselines.py`

- `_has_display`: detect whether a display backend is available for Matplotlib.
- `extract_boundary_mask`: derive boundary voxels from a binary segmentation.
- `compute_entropy_map`: compute voxelwise predictive entropy from logits.
- `compute_max_softmax`: compute maximum softmax probability per voxel.
- `compute_energy_score`: compute free-energy OOD scores per voxel.
- `process_numpy_logits`: load one logits array and derive all scalar/structural summaries.
- `find_best_slice`: identify the most informative 2D slice for visualization.
- `_summarize_logits_file`: worker helper for summarizing one logits file.
- `compute_dataset_statistics`: summarize logits-derived metrics for one dataset.
- `print_summary_statistics`: print descriptive statistics for one summary dataframe.
- `perform_statistical_tests`: run pairwise statistical tests between ID and OOD summaries.
- `compute_ood_metrics`: compute AUROC/FPR95 for one ID-versus-OOD pairing.
- `compute_all_ood_metrics`: run the OOD metric computation across all OOD datasets.
- `compute_ood_metrics_bootstrap`: bootstrap one ID-versus-OOD metric estimate.
- `compute_all_ood_metrics_bootstrap`: bootstrap all OOD pairings.
- `_orient_slice`: orient a 2D slice for presentation.
- `_get_slice`: extract a 2D slice along one axis.
- `_normalize_ct`: clip and normalize CT intensities for display.
- `visualize_heatmaps`: render heatmap-style uncertainty visualizations.
- `visualize_contours_overlay`: render one overlay view combining CT, mask, and uncertainty.
- `visualize_contours_permodel`: render one multi-model qualitative comparison.
- `_to_jsonable`: convert nested values into JSON-safe types.
- `summarize_global_dataset`: summarize one dataset’s global metrics for export.
- `write_global_results_json`: write a global-results JSON payload.
- `run_global`: CLI path for dataset-level analysis.
- `run_local`: CLI path for one local qualitative visualization.
- `run_local_batch`: CLI path for batch local qualitative visualization.
- `build_parser`: CLI parser.

### `roi_logit_baselines.py`

- `_to_jsonable`: convert nested values into JSON-safe types.
- `result_root`: resolve the output root for one ROI experiment.
- `resolve_scan_entry`: assemble the image/mask/logits paths for one scan.
- `sample_roi_slices`: sample 3D ROIs centered on a segmentation mask.
- `crop_with_padding`: crop one ROI and pad it to the requested size.
- `resample_mask_and_logits`: align masks and logits into a common array space.
- `summarize_scan`: summarize ROI statistics for one scan.
- `compute_dataset_statistics_roi`: aggregate ROI summaries across one dataset.
- `write_results_json`: write the ROI results payload.
- `build_parser`: CLI parser.
- `main`: CLI entrypoint.

## Reusable Scripts

### `scripts/make_json.py`

- `build_result_dir`: resolve the segmentation-result directory used to build a manifest.
- `build_image_path`: map one dataset filename to its source image path.
- `main`: CLI entrypoint for manifest generation.

### `scripts/evaluate_segmentation.py`

- `main`: evaluate segmentation outputs against ground truth and write per-scan CSV metrics.

### `scripts/summary_metrics.py`

- `compute_wilcoxon_results`: run paired Wilcoxon tests over summary metrics.
- `main`: CLI entrypoint for summary statistics generation.

### `scripts/radiomics_analysis.py`

- `build_features`: compute radiomics feature rows from one manifest split.
- `main`: CLI entrypoint for radiomics CSV generation.

### `scripts/radiomics_map_shorthand.py`

- `build_mapping`: build a shorthand-name mapping table for radiomics columns.
- `main`: CLI entrypoint.

### `scripts/smoke_check.py`

- `main`: lightweight repository path/import smoke test.

### `scripts/build_anchor_summary.py`

- `prediction_root`: resolve the segmentation prediction directory for one model and dataset.
- `component_summary`: summarize connected-component counts and sizes for one mask.
- `summarize_scan`: summarize one predicted scan for anchor-style analysis.
- `_worker`: multiprocessing wrapper for scan summarization.
- `build_anchor_summary`: aggregate scan-level summaries over a dataset.
- `parse_args`: CLI parser.
- `main`: CLI entrypoint.

### `scripts/export_rfdeep_scan_scores.py`

- `stable_dataset_seed`: derive deterministic per-dataset seeds from a base seed.
- `prepare_feature_data`: load and prepare deep-feature data for score export.
- `annotate_df`: attach dataset and scan metadata to a prediction dataframe.
- `dataset_specific_run`: export scores for dataset-specific RF-Deep models.
- `ensemble_run`: export scores for ensemble RF-Deep models.
- `unified_run`: export scores for unified RF-Deep models.
- `lodo_run`: export scores for LODO RF-Deep models.
- `export_scores`: orchestrate the chosen score-export mode and write outputs.
- `parse_args`: CLI parser.
- `main`: CLI entrypoint.

### `scripts/analyze_anchor_failure_modes.py`

- `summarize_metric`: summarize one repeated metric into mean and interval fields.
- `require_unique_scan_rows`: validate that input rows are unique at scan level.
- `add_component_bin`: bucket connected-component counts into analysis bins.
- `add_tiny_flag`: mark scans containing tiny anchor components.
- `volume_quartile_summary`: summarize fallback behavior by predicted-volume quartile.
- `component_distribution_summary`: summarize component-count distributions.
- `component_auroc_summary`: compute AUROC by connected-component group.
- `no_prediction_fallback_summary`: summarize empty-prediction fallback behavior.
- `parse_args`: CLI parser.
- `main`: CLI entrypoint.

## Figure Scripts

### `paper_figures/viz_logit_pooling.py`

- `load_json_results`: load saved logit-pooling results from JSON artifacts.
- `build_metric_results`: convert saved JSON into plotting-friendly structures.
- `create_subplot`: render one metric subplot.
- `create_figure`: build the full figure for a cohort set.
- `main`: CLI entrypoint.

### `paper_figures/viz_shap_deepfeatures_external.py`

- `safe_sample`: bounded sampler for repeated SHAP analyses.
- `prepare_data`: load and prepare feature data for external-cohort SHAP analyses.
- `create_id_split`: split ID data into train/test partitions.
- `train_dataset_specific_models`: train one model per OOD dataset for SHAP analysis.
- `build_external_test_matrix`: assemble the external test feature matrix.
- `plot_dataset_specific_shap`: render dataset-specific SHAP bar plots.
- `plot_ensemble_shap`: render ensemble SHAP bar plots.
- `run_external_shap`: execute the full external-cohort SHAP pipeline.
- `parse_args`: CLI parser.
- `main`: CLI entrypoint.


### `paper_figures/scanner_performance.py`

- `_mean_se`: mean and standard-error summary helper.
- `_mean_std`: mean and standard-deviation summary helper.
- `_siemens_band`: map Siemens kernel labels into broad bands.
- `_ge_bucket`: map GE kernel labels into broad buckets.
- `_kernel_group`: derive a harmonized kernel grouping for one row.
- `load_scanner_metadata`: load scanner annotations from spreadsheet data.
- `load_model_scores`: load per-scan model performance CSVs.
- `prepare_scanner_analysis`: merge score tables with metadata annotations.
- `create_overall_figure`: build the main scanner-summary figure.
- `create_radar_plot`: build the radar-style overview figure.
- `create_supplemental_figure`: build supplemental scanner-performance panels.
- `create_scanner_performance_figures`: convenience wrapper returning all scanner figures.
- `main`: CLI entrypoint.

### `paper_figures/ood_detection_panels.py`

- `create_figure`: create the summary OOD detection panel figure.
- `main`: CLI entrypoint.

### `paper_figures/viz_ablations.py`

- `create_subplot`: render one ablation subplot.
- `create_ablation_figure`: assemble one named ablation figure.
- `save_figures`: save one or more ablation figures to disk.
- `main`: CLI entrypoint.

## Model Source

### `models/smit.py`

- `Mlp`, `WindowAttention`, `SwinTransformerBlock`, `PatchMerging`, `BasicLayer`, `PatchEmbed`, `SinPositionalEncoding3D`, `UnetResBlock_No_Downsampleing`, `UnetrBasicBlock_No_DownSampling`, `SwinTransformer_`, `SMIT_3D_Seg`: model-building blocks for the SMIT family.
- `window_partition`, `window_reverse`: tensor reshaping helpers for windowed attention.

### `models/swinunetr.py`

- `SwinUNETR`, `WindowAttention`, `SwinTransformerBlock`, `PatchMerging`, `BasicLayer`, `SwinTransformer`: model-building blocks for the SwinUNETR family.
- `window_partition`, `window_reverse`, `get_window_size`, `compute_mask`: tensor/window helpers for the Swin transformer implementation.
