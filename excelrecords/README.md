# Excel Records

`excelrecords/` stores generated CSV metric tables, especially segmentation evaluation outputs intended for summary plots and statistical testing.

Current workflow:

- producer: [`scripts/evaluate_segmentation.py`](../scripts/evaluate_segmentation.py)
- consumer: [`scripts/summary_metrics.py`](../scripts/summary_metrics.py)
- consumer: [`paper_figures/scanner_performance.py`](../paper_figures/scanner_performance.py)

This directory is generated output and is intentionally kept out of the shared git surface.
