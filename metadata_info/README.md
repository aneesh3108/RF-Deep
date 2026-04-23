# Metadata Info

`metadata_info/` contains metadata inputs used for scanner-based analysis and metadata holdout experiments.

Current contents include spreadsheet-style metadata tables such as scanner manufacturer, reconstruction kernel, and contrast-related annotations that are merged with model outputs during downstream analysis.

It also stores shared radiomics-support metadata:

- `ibsi1.json`: pyCERR radiomics settings used by [`scripts/radiomics_analysis.py`](../scripts/radiomics_analysis.py)
- `radiomics_mapping.csv`: shorthand and display-name mapping for radiomics feature columns

Primary consumers:

- [`paper_figures/scanner_performance.py`](../paper_figures/scanner_performance.py)
- [`ood_metadata_holdout.py`](../ood_metadata_holdout.py)
- [`scripts/radiomics_map_shorthand.py`](../scripts/radiomics_map_shorthand.py)
