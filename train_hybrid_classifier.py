from __future__ import annotations

from pathlib import Path

import numpy as np
import rasterio


FACTOR_FILES = [
    "sx_aspect",
    "sx_elevation",
    "sx_fault",
    "sx_landuse",
    "sx_lithology",
    "sx_ndvi",
    "sx_plan",
    "sx_profile",
    "sx_rainfall",
    "sx_river",
    "sx_road",
    "sx_rough",
    "sx_slope",
    "sx_spi",
    "sx_terrain",
    "sx_twi",
]


def _resolve_factor_path(base_dir: Path, factor_name: str) -> Path:
    for suffix in (".tif", ".png"):
        candidate = base_dir / f"{factor_name}{suffix}"
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Missing raster for factor: {factor_name}")


def load_full_feature_stack() -> tuple[np.ndarray, dict[str, object], list[str], np.ndarray]:
    dataset_dir = Path("_extracted_dataset")
    arrays: list[np.ndarray] = []
    profile: dict[str, object] | None = None
    valid_mask: np.ndarray | None = None

    for factor_name in FACTOR_FILES:
        raster_path = _resolve_factor_path(dataset_dir, factor_name)
        with rasterio.open(raster_path) as src:
            band = src.read(1).astype(np.float32)
            arrays.append(band)
            if profile is None:
                profile = src.profile.copy()
            band_mask = np.isfinite(band)
            valid_mask = band_mask if valid_mask is None else (valid_mask & band_mask)

    stack = np.stack(arrays, axis=-1)
    if profile is None or valid_mask is None:
        raise RuntimeError("Unable to load raster stack from extracted dataset.")

    return stack, profile, FACTOR_FILES, valid_mask
