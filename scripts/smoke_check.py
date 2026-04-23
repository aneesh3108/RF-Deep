"""
Lightweight import and path-existence checks for the repo.
"""

from __future__ import annotations

import importlib
import sys

from project_paths import (
    FIGURES_ROOT,
    JSONS_ROOT,
    MODELS_ROOT,
    PAPER_FIGURES_ROOT,
    PROJECT_ROOT,
    RADIOMICS_FEATURES_ROOT,
    RESULTS_ROOT,
    ensure_output_dirs,
)


REQUIRED_DIRS = {
    "project": PROJECT_ROOT,
    "models": MODELS_ROOT,
    "paper_figures": PAPER_FIGURES_ROOT,
    "scripts": PROJECT_ROOT / "scripts",
    "results": RESULTS_ROOT,
    "figures_tmlr": FIGURES_ROOT,
    "radiomics_features": RADIOMICS_FEATURES_ROOT,
    "jsons": JSONS_ROOT,
}

MODULES = [
    "project_paths",
    "ood_utils",
    "ood_maha",
    "ood_rfdeep",
    "logit_baselines",
    "paper_figures",
    "paper_figures.scanner_performance",
    "paper_figures.viz_ablations",
    "paper_figures.ood_detection_panels",
    "scripts",
    "scripts.make_json",
    "scripts.evaluate_segmentation",
    "scripts.summary_metrics",
    "scripts.radiomics_analysis",
    "scripts.radiomics_map_shorthand",
]


def main() -> int:
    ensure_output_dirs()
    failures: list[str] = []

    for label, path in REQUIRED_DIRS.items():
        if path.exists():
            print(f"[ok] path: {label} -> {path}")
        else:
            failures.append(f"missing path: {label} -> {path}")

    for module_name in MODULES:
        try:
            importlib.import_module(module_name)
            print(f"[ok] import: {module_name}")
        except Exception as exc:  # pragma: no cover
            failures.append(f"import failed: {module_name}: {exc}")

    if failures:
        print("\nSmoke check failures:")
        for failure in failures:
            print(f"- {failure}")
        return 1

    print("\nSmoke check passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
