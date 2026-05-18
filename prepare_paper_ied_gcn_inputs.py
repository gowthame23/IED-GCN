from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from train_hybrid_classifier import FACTOR_FILES, load_full_feature_stack


DATASET_CSV = Path("hybrid_outputs/hybrid_training_samples.csv")
OUTPUT_DIR = Path("paper_ied_gcn_inputs")
FEATURES_PATH = OUTPUT_DIR / "features.npy"
LABELS_PATH = OUTPUT_DIR / "labels.npy"
PROFILE_PATH = OUTPUT_DIR / "profile.json"
REPORT_PATH = OUTPUT_DIR / "prepare_report.json"


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if not DATASET_CSV.exists() and FEATURES_PATH.exists() and LABELS_PATH.exists():
        print("Legacy hybrid training CSV not found; reusing existing paper_ied_gcn_inputs.")
        report = {
            "source_training_csv": None,
            "feature_shape": [int(v) for v in np.load(FEATURES_PATH).shape],
            "label_shape": [int(v) for v in np.load(LABELS_PATH).shape],
            "labeled_pixels": int(np.sum(np.load(LABELS_PATH) >= 0)),
            "used_training_rows": None,
            "skipped_training_rows": None,
            "outputs": {
                "features_npy": str(FEATURES_PATH.resolve()),
                "labels_npy": str(LABELS_PATH.resolve()),
                "profile_json": str(PROFILE_PATH.resolve()) if PROFILE_PATH.exists() else None,
            },
            "note": "Existing paper_ied_gcn_inputs were reused because the legacy hybrid dataset is not present.",
        }
        with open(REPORT_PATH, "w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2)
        return

    print("Loading normalized raster stack...")
    stack, profile, _, valid_mask = load_full_feature_stack()
    df = pd.read_csv(DATASET_CSV)

    required = {"pixel_x", "pixel_y", "label"}.union(FACTOR_FILES)
    missing = required.difference(df.columns)
    if missing:
        missing_str = ", ".join(sorted(missing))
        raise ValueError(f"Training CSV is missing required columns: {missing_str}")

    labels = np.full(stack.shape[:2], -1, dtype=np.int16)
    used = 0
    skipped = 0

    for _, row in df.iterrows():
        col = int(row["pixel_x"])
        rr = int(row["pixel_y"])
        if 0 <= rr < labels.shape[0] and 0 <= col < labels.shape[1] and valid_mask[rr, col]:
            labels[rr, col] = int(row["label"])
            used += 1
        else:
            skipped += 1

    np.save(OUTPUT_DIR / "features.npy", stack.astype(np.float32))
    np.save(OUTPUT_DIR / "labels.npy", labels)

    profile_json = {
        "driver": str(profile.get("driver", "GTiff")),
        "height": int(profile["height"]),
        "width": int(profile["width"]),
        "count": int(profile.get("count", 1)),
        "dtype": str(profile.get("dtype", "float32")),
        "crs": str(profile.get("crs")) if profile.get("crs") is not None else None,
        "transform": list(profile["transform"]) if profile.get("transform") is not None else None,
    }
    with open(OUTPUT_DIR / "profile.json", "w", encoding="utf-8") as handle:
        json.dump(profile_json, handle, indent=2)

    report = {
        "source_training_csv": str(DATASET_CSV.resolve()),
        "feature_shape": [int(v) for v in stack.shape],
        "label_shape": [int(v) for v in labels.shape],
        "labeled_pixels": int(np.sum(labels >= 0)),
        "used_training_rows": int(used),
        "skipped_training_rows": int(skipped),
        "outputs": {
            "features_npy": str((OUTPUT_DIR / "features.npy").resolve()),
            "labels_npy": str((OUTPUT_DIR / "labels.npy").resolve()),
            "profile_json": str((OUTPUT_DIR / "profile.json").resolve()),
        },
        "note": "labels.npy stores sparse labels with -1 for unlabeled pixels, suitable for the paper-closer IED-GCN prototype.",
    }
    with open(OUTPUT_DIR / "prepare_report.json", "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)

    print(f"Prepared paper-oriented inputs in {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
