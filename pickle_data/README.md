# Pickle Data

`pickle_data/` stores cached deep-feature vectors extracted from segmentation backbones.

Conventions:

- filenames typically follow `{model}_size{img_size}_featvec.pkl`
- files contain per-dataset feature dictionaries for RF-Deep, Mahalanobis, SHAP, and ablation experiments

Relevant code:

- [`extract_features.py`](../extract_features.py): writes feature caches
- [`ood_rfdeep.py`](../ood_rfdeep.py): RF-Deep experiments
- [`ood_maha.py`](../ood_maha.py): Mahalanobis experiments
- [`ood_utils.py`](../ood_utils.py): shared feature loading helpers
- figure scripts under [`paper_figures/`](../paper_figures)
