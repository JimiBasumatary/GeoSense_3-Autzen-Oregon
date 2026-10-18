
from pathlib import Path
import json

import laspy
import numpy as np
import pandas as pd
import rasterio
from PIL import Image
from pyproj import Transformer

ROOT = Path(r"D:\GeoSense_Phase3")

ORIGINAL_PATH = (
    ROOT / "data" / "lidar" / "processed"
    / "autzen_classified_hag.laz"
)
TEST_PATH = (
    ROOT / "data" / "lidar" / "processed"
    / "autzen_classified_hag_count8_50ft_test.laz"
)
IMAGE_PATH = (
    ROOT / "data" / "aerial"
    / "autzen_naip_georeferenced_fixed.tif"
)
MASK_PATH = (
    ROOT / "outputs" / "sam_georeferenced"
    / "individual_masks" / "mask_0013.png"
)
OUT_DIR = ROOT / "outputs" / "sam_georeferenced"

for path in [ORIGINAL_PATH, TEST_PATH, IMAGE_PATH, MASK_PATH]:
    if not path.exists():
        raise FileNotFoundError(f"Required file missing: {path}")

mask = np.asarray(Image.open(MASK_PATH).convert("L")) > 0

with rasterio.open(IMAGE_PATH) as src:
    transform = src.transform
    image_crs = src.crs
    width = src.width
    height = src.height
    bounds = src.bounds

if mask.shape != (height, width):
    raise ValueError(
        f"Mask dimensions {mask.shape} do not match image "
        f"dimensions {(height, width)}."
    )

def analyse_file(path, label):
    print(f"\nReading {label}: {path.name}")
    las = laspy.read(path)

    classification = np.asarray(las.classification)
    x = np.asarray(las.x)
    y = np.asarray(las.y)

    hag_name = next(
        (
            name for name in las.point_format.dimension_names
            if name.lower() == "heightaboveground"
        ),
        None,
    )
    if hag_name is None:
        raise ValueError(f"No HAG dimension in {path.name}")

    hag = np.asarray(getattr(las, hag_name), dtype=np.float64)

    source_crs = las.header.parse_crs()
    if source_crs is None:
        raise ValueError(f"CRS missing in {path.name}")

    transformer = Transformer.from_crs(
        source_crs, image_crs, always_xy=True
    )
    xx, yy = transformer.transform(x, y)

    inverse = ~transform
    cols = np.floor(
        inverse.a * xx + inverse.b * yy + inverse.c
    ).astype(np.int64)
    rows = np.floor(
        inverse.d * xx + inverse.e * yy + inverse.f
    ).astype(np.int64)

    inside = (
        (cols >= 0) & (cols < width)
        & (rows >= 0) & (rows < height)
        & (xx >= bounds.left) & (xx < bounds.right)
        & (yy >= bounds.bottom) & (yy < bounds.top)
    )

    indices = np.flatnonzero(inside)
    matches = mask[rows[inside], cols[inside]]
    selected = indices[matches]

    # Keep building-class points inside Mask 13.
    building_idx = selected[classification[selected] == 6]
    values = hag[building_idx]
    finite = np.isfinite(values)
    values = values[finite]

    if len(values) == 0:
        raise ValueError(
            f"No finite building HAG values found inside Mask 13 "
            f"for {label}. Registration and overlap need review."
        )

    result = {
        "file": label,
        "point_count_in_mask_all_classes": int(len(selected)),
        "building_points_in_mask": int(len(building_idx)),
        "finite_building_hag_count": int(len(values)),
        "negative_building_hag_count": int(np.count_nonzero(values < 0)),
        "building_hag_min_ft": float(np.min(values)),
        "building_hag_p10_ft": float(np.percentile(values, 10)),
        "building_hag_median_ft": float(np.median(values)),
        "building_hag_p90_ft": float(np.percentile(values, 90)),
        "building_hag_max_ft": float(np.max(values)),
        "image_registration_verified": False,
    }

    print(json.dumps(result, indent=2))
    return result

original = analyse_file(ORIGINAL_PATH, "ORIGINAL")
alternative = analyse_file(TEST_PATH, "COUNT8_50FT_TEST")

report = {
    "mask_id": 13,
    "image_crs": str(image_crs),
    "image_registration_verified": False,
    "original": original,
    "alternative": alternative,
    "interpretation": (
        "Spatial mask-based comparison of building HAG distributions. "
        "Different point order is handled by independently mapping each "
        "file's coordinates to the raster mask. This is diagnostic only: "
        "unverified image registration and lack of independent terrain "
        "reference prevent declaring either method more accurate."
    ),
}

OUT_DIR.mkdir(parents=True, exist_ok=True)
json_path = OUT_DIR / "mask13_hag_methods_comparison.json"
csv_path = OUT_DIR / "mask13_hag_methods_comparison.csv"

json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
pd.DataFrame([original, alternative]).to_csv(csv_path, index=False)

print("\nMASK 13 HAG COMPARISON SAVED")
print("JSON:", json_path)
print("CSV:", csv_path)
print("\nReminder: this comparison does not independently verify roof boundaries.")
