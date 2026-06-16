# Excel Records

`excelrecords/` stores generated CSV metric tables, especially segmentation evaluation outputs intended for summary plots and statistical testing.

Relevant code:

- [`scripts/evaluate_segmentation.py`](../scripts/evaluate_segmentation.py): writes segmentation metric CSVs
- [`scripts/summary_metrics.py`](../scripts/summary_metrics.py): summarizes metric tables
- [`paper_figures/scanner_performance.py`](../paper_figures/scanner_performance.py): renders scanner-performance figures

This directory is generated output and is intentionally kept out of git.
