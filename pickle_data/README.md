# Pickle Data

`pickle_data/` stores cached deep-feature vectors extracted from segmentation backbones.

Conventions:

- filenames typically follow `{model}_size{img_size}_featvec.pkl`
- contents are per-dataset feature dictionaries consumed by the RF-Deep, Mahalanobis, SHAP, and ablation workflows

Primary producer:

- [`extract_features.py`](../extract_features.py)

Primary consumers:

- [`ood_rfdeep.py`](../ood_rfdeep.py)
- [`ood_maha.py`](../ood_maha.py)
- [`ood_utils.py`](../ood_utils.py)
- figure scripts under [`paper_figures/`](../paper_figures)
