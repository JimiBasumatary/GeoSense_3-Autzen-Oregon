
from pathlib import Path
import json

import laspy
import numpy as np
import rasterio
from PIL import Image
from pyproj import Transformer

ROOT = Path(r"D:\GeoSense_Phase3")

LIDAR_PATH = (
    ROOT / "data" / "lidar" / "processed"
    / "autzen_classified_hag.laz"
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

for path in [LIDAR_PATH, IMAGE_PATH, MASK_PATH]:
    if not path.exists():
        raise FileNotFoundError(f"Missing input: {path}")

OUT_DIR.mkdir(parents=True, exist_ok=True)

print("Reading candidate mask...")
mask = np.asarray(Image.open(MASK_PATH).convert("L")) > 0

with rasterio.open(IMAGE_PATH) as src:
    transform = src.transform
    image_crs = src.crs
    width, height = src.width, src.height
    pixel_area_map_units = abs(transform.a * transform.e - transform.b * transform.d)
    raster_bounds = src.bounds

if mask.shape != (height, width):
    raise ValueError(
        f"Mask size {mask.shape} differs from image size {(height, width)}"
    )

mask_pixels = int(np.count_nonzero(mask))
mask_area_source_units2 = mask_pixels * pixel_area_map_units

# EPSG:2992 uses US survey feet, so convert square feet to square metres.
# This assumes the raster CRS uses feet as its horizontal map units.
mask_area_m2 = mask_area_source_units2 * (0.3048006096012192 ** 2)

print("Reading LiDAR...")
las = laspy.read(LIDAR_PATH)
classification = np.asarray(las.classification)
x = np.asarray(las.x)
y = np.asarray(las.y)
hag = np.asarray(las["HeightAboveGround"], dtype=np.float64)

source_crs = las.header.parse_crs()
if source_crs is None or image_crs is None:
    raise ValueError("Missing CRS in LiDAR or raster.")

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

inside_raster = (
    (cols >= 0) & (cols < width)
    & (rows >= 0) & (rows < height)
    & (xx >= raster_bounds.left) & (xx < raster_bounds.right)
    & (yy >= raster_bounds.bottom) & (yy < raster_bounds.top)
)

indices = np.flatnonzero(inside_raster)
mask_matches = mask[rows[inside_raster], cols[inside_raster]]
selected = indices[mask_matches]

building_indices = selected[classification[selected] == 6]
building_hag = hag[building_indices]
building_hag = building_hag[np.isfinite(building_hag)]

if len(building_hag) == 0:
    raise ValueError("No building-class HAG points found inside Mask 13.")

# A simple diagnostic: mask area times a representative height.
# This is NOT a validated building volume because the mask may include
# non-roof pixels and HAG does not by itself establish roof geometry.
median_hag_ft = float(np.median(building_hag))
p90_hag_ft = float(np.percentile(building_hag, 90))
median_hag_m = median_hag_ft * 0.3048006096012192
p90_hag_m = p90_hag_ft * 0.3048006096012192

report = {
    "candidate_mask_id": 13,
    "mask_file": str(MASK_PATH.relative_to(ROOT)),
    "image_file": str(IMAGE_PATH.relative_to(ROOT)),
    "lidar_file": str(LIDAR_PATH.relative_to(ROOT)),
    "image_crs": str(image_crs),
    "mask_pixels": mask_pixels,
    "pixel_area_source_units_squared": pixel_area_map_units,
    "mask_area_source_units_squared": mask_area_source_units2,
    "mask_area_m2": mask_area_m2,
    "building_points_inside_mask": int(len(building_indices)),
    "finite_building_hag_points": int(len(building_hag)),
    "building_hag_median_ft": median_hag_ft,
    "building_hag_p90_ft": p90_hag_ft,
    "building_hag_median_m": median_hag_m,
    "building_hag_p90_m": p90_hag_m,
    "area_times_median_height_m3_diagnostic_only": mask_area_m2 * median_hag_m,
    "area_times_p90_height_m3_diagnostic_only": mask_area_m2 * p90_hag_m,
    "registration_verified": False,
    "volume_is_validated": False,
    "limitations": [
        "The aerial image georeferencing has not been independently validated.",
        "SAM Mask 13 may include pixels outside the actual roof footprint.",
        "Multiplying mask area by a representative HAG is a diagnostic proxy, not a validated building volume.",
        "HAG accuracy requires independent terrain and roof-height validation.",
        "The mask area calculation assumes the raster CRS horizontal units are feet."
    ]
}

json_path = OUT_DIR / "mask13_rooftop_volume_diagnostic.json"
csv_path = OUT_DIR / "mask13_rooftop_volume_diagnostic.csv"

json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

import pandas as pd
pd.DataFrame([report]).to_csv(csv_path, index=False)

print("\nMASK 13 ROOFTOP DIAGNOSTIC")
print(f"Mask pixels: {mask_pixels:,}")
print(f"Candidate mask area: {mask_area_m2:,.2f} m²")
print(f"Building points: {len(building_indices):,}")
print(f"Median HAG: {median_hag_ft:.2f} ft ({median_hag_m:.2f} m)")
print(f"90th percentile HAG: {p90_hag_ft:.2f} ft ({p90_hag_m:.2f} m)")
print(
    "Area × median height (diagnostic only): "
    f"{report['area_times_median_height_m3_diagnostic_only']:,.2f} m³"
)
print(
    "Area × p90 height (diagnostic only): "
    f"{report['area_times_p90_height_m3_diagnostic_only']:,.2f} m³"
)
print("\nSaved:", json_path)
print("Saved:", csv_path)
print("\nWARNING: These are diagnostic proxies, not validated building volumes.")
