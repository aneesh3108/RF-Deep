# JSON Manifests

`jsons/` stores MONAI-style dataset manifests used by feature extraction, benchmarking, and radiomics pipelines.

These manifests decouple the codebase from machine-specific file paths and provide the shared scan lists used by feature extraction, inference, and analysis scripts.

## Example Structure

```json
{
  "validation": [
    {
      "image": "DATA_ROOT::dataset/images/LUNG_001.nii.gz",
      "label": "DATA_ROOT::dataset/labels/LUNG_001.nii.gz"
    }
  ]
}
```

## Key Fields

- `validation`: the current shared manifests use a single split key listing evaluation samples
- `image` and `label`: image and segmentation-mask paths, usually stored with root tags such as `DATA_ROOT::...`
- identifiers and scanner metadata are not currently embedded in these manifest rows; metadata-aware analysis instead joins through filenames and auxiliary files in [`metadata_info/`](../metadata_info)

## Conventions

- filenames typically follow `{model}_{dataset}_src.json`
- current shared manifests primarily use the `validation` split key
- paths may be stored with root tags such as `DATA_ROOT::...` or `RESULTS_ROOT::...`, which are resolved through [`project_paths.py`](../project_paths.py)

Generated or refreshed manifests are usually created by [`scripts/make_json.py`](../scripts/make_json.py).

## Practical Notes

- `data/` may be a symlink on local machines, as long as the tagged manifest paths still resolve correctly through [`project_paths.py`](../project_paths.py)
- if images are preprocessed before feature extraction, document that preprocessing alongside the manifest so the feature caches and downstream metrics remain interpretable
