
from pathlib import Path

import laspy
import numpy as np
import pandas as pd
from PIL import Image

ROOT = Path(r"D:\GeoSense_Phase3")

LIDAR_PATH = (
    ROOT / "data" / "lidar" / "processed"
    / "autzen_classified_hag.laz"
)
MASK_PATH = (
    ROOT / "outputs" / "sam_georeferenced"
    / "individual_masks" / "mask_0013.png"
)
OUT_PATH = (
    ROOT / "outputs" / "sam_georeferenced"
    / "mask13_height_diagnostic.csv"
)

# Read the SAM mask.
mask = np.asarray(Image.open(MASK_PATH).convert("L")) > 0
mask_rows, mask_cols = np.where(mask)

if len(mask_rows) == 0:
    raise ValueError("Mask 13 is empty.")

# Read the classified LiDAR points and HAG.
print("Reading LiDAR...")
las = laspy.read(LIDAR_PATH)

classification = np.asarray(las.classification)
x = np.asarray(las.x)
y = np.asarray(las.y)
z = np.asarray(las.z)

dimensions = las.point_format.dimension_names
hag_name = next(
    (
        name for name in dimensions
        if name.lower() == "heightaboveground"
    ),
    None,
)
if hag_name is None:
    raise ValueError("No height-above-ground dimension found.")

hag = np.asarray(getattr(las, hag_name))

# Convert LiDAR coordinates to the raster's pixel grid.
import rasterio

IMAGE_PATH = (
    ROOT / "data" / "aerial"
    / "autzen_naip_georeferenced_fixed.tif"
)

with rasterio.open(IMAGE_PATH) as src:
    transform = src.transform
    image_crs = src.crs
    width = src.width
    height = src.height
    bounds = src.bounds

from pyproj import Transformer

source_crs = las.header.parse_crs()
transformer = Transformer.from_crs(
    source_crs, image_crs, always_xy=True
)
xx, yy = transformer.transform(x, y)

inverse = ~transform
cols = np.floor(
    inverse.a * xx + inverse.b * yy + inverse.c
).astype(int)
rows = np.floor(
    inverse.d * xx + inverse.e * yy + inverse.f
).astype(int)

inside = (
    (cols >= 0) & (cols < width)
    & (rows >= 0) & (rows < height)
    & (xx >= bounds.left) & (xx < bounds.right)
    & (yy >= bounds.bottom) & (yy < bounds.top)
)

# Restrict the analysis to points inside Mask 13.
inside_indices = np.flatnonzero(inside)
mask_match = mask[rows[inside], cols[inside]]
selected_indices = inside_indices[mask_match]

# Report class counts inside the mask.
selected_classes = classification[selected_indices]
class_counts = pd.Series(selected_classes).value_counts().sort_index()

print("\nCLASS COUNTS INSIDE MASK 13")
print(class_counts.to_string())

# Summarise HAG separately by class.
records = []

for class_id in sorted(np.unique(selected_classes)):
    idx = selected_indices[selected_classes == class_id]
    values = hag[idx]
    values = values[np.isfinite(values)]

    if len(values) == 0:
        continue

    records.append({
        "mask_id": 13,
        "classification": int(class_id),
        "point_count": int(len(values)),
        "hag_min_ft": float(np.min(values)),
        "hag_p10_ft": float(np.percentile(values, 10)),
        "hag_median_ft": float(np.median(values)),
        "hag_p90_ft": float(np.percentile(values, 90)),
        "hag_max_ft": float(np.max(values)),
    })

df = pd.DataFrame(records)
df.to_csv(OUT_PATH, index=False)

print("\nHAG SUMMARY BY CLASS")
print(df.to_string(index=False))
print("\nSaved:", OUT_PATH)
print(
    "\nCaution: Mask overlap and HAG statistics do not by "
    "themselves verify roof boundaries or image registration."
)
