"""
Create short-name mappings for radiomics feature columns.
"""

from __future__ import annotations

import argparse

import pandas as pd

from project_paths import RADIOMICS_FEATURES_ROOT, RADIOMICS_MAPPING_PATH


def build_mapping(csv_name: str) -> pd.DataFrame:
    df = pd.read_csv(RADIOMICS_FEATURES_ROOT / csv_name)
    mapping_rows = []
    counters: dict[str, int] = {}

    for name in df.columns.tolist():
        name_lower = name.lower()
        if "shape" in name_lower:
            prefix = "shape_"
        elif "glcm" in name_lower:
            prefix = "glcm_"
        elif "firstorder" in name_lower or "first_order" in name_lower:
            prefix = "first_"
        elif "gldm" in name_lower:
            prefix = "gldm_"
        elif "glrlm" in name_lower:
            prefix = "glrlm_"
        elif "glszm" in name_lower:
            prefix = "glszm_"
        elif "ngtdm" in name_lower:
            prefix = "ngtdm_"
        elif "gtdm" in name_lower:
            prefix = "gtdm_"
        else:
            prefix = "feat_"

        count = counters.get(prefix, 1)
        short_name = f"{prefix}{count}"
        counters[prefix] = count + 1
        mapping_rows.append(
            {
                "Mapped Name": short_name,
                "Original Feature Name": name,
            }
        )

    return pd.DataFrame(mapping_rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build shorthand radiomics feature names.")
    parser.add_argument("--input-csv", default="smit_lrad_src.csv", help="Radiomics CSV under radiomics_features/.")
    args = parser.parse_args()

    mapping_df = build_mapping(args.input_csv)
    mapping_df.to_csv(RADIOMICS_MAPPING_PATH, index=False)
    print(f"Wrote radiomics mapping to {RADIOMICS_MAPPING_PATH}")


if __name__ == "__main__":
    main()
