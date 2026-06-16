# Metadata Info

`metadata_info/` contains metadata inputs for scanner-based analysis and metadata holdout experiments.

Contents include spreadsheet-style metadata tables for scanner manufacturer, reconstruction kernel, and contrast-related annotations. These files are joined with model outputs during downstream analysis.

Shared radiomics metadata:

- `ibsi1.json`: pyCERR radiomics settings for [`scripts/radiomics_analysis.py`](../scripts/radiomics_analysis.py)
- `radiomics_mapping.csv`: shorthand and display-name mapping for radiomics feature columns

Relevant code:

- [`paper_figures/scanner_performance.py`](../paper_figures/scanner_performance.py): scanner-stratified summaries
- [`ood_metadata_holdout.py`](../ood_metadata_holdout.py): metadata holdout experiments
- [`scripts/radiomics_map_shorthand.py`](../scripts/radiomics_map_shorthand.py): radiomics column-name mapping
