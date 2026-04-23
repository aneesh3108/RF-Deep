# Radiomics Features

`radiomics_features/` stores generated CSV exports of radiomics features computed from manifest-defined image and label pairs.

Conventions:

- filenames typically follow `{model}_{dataset}_src.csv`
- rows are scan-level feature records
- shorthand/pretty-name mapping is handled through [`metadata_info/radiomics_mapping.csv`](../metadata_info/radiomics_mapping.csv)

Primary producers and consumers:

- producer: [`scripts/radiomics_analysis.py`](../scripts/radiomics_analysis.py)
- figure scripts under [`paper_figures/`](../paper_figures)
