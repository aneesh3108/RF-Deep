# Models

`models/` contains the architecture definitions used by RF-Deep and related segmentation experiments.

Contents:

- `smit.py`: SMIT-based 3D segmentation backbone and supporting transformer blocks
- `swinunetr.py`: SwinUNETR-based 3D segmentation backbone and supporting transformer blocks
- `configs_smit.py`: configuration helpers for common SMIT model variants

Ignored local subdirectories:

- `finetuned_weights/`: trained checkpoints and training logs
- `pretrained_weights/`: downloaded or locally cached pretrained weights

Implementation note:

- models intended for deep-feature extraction should expose a `forward_debug()` method that returns the intermediate feature tensors needed by [`extract_features.py`](../extract_features.py)

If model code changes, keep architectural edits in the Python source files here and keep large checkpoint artifacts out of git.
