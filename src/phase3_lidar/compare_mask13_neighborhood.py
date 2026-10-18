
from pathlib import Path

import laspy
import numpy as np
import pandas as pd
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
OUT_PATH = (
    ROOT / "outputs" / "sam_georeferenced"
    / "mask13_neighborhood_comparison.csv"
)

print("Reading LiDAR...")
las = laspy.read(LIDAR_PATH)

classification = np.asarray(las.classification)
building = classification == 6

x = np.asarray(las.x)[building]
y = np.asarray(las.y)[building]
hag = np.asarray(las.HeightAboveGround)[building]

print("Building-class points:", len(x))

with rasterio.open(IMAGE_PATH) as src:
    transform = src.transform
    image_crs = src.crs
    width = src.width
    height = src.height
    bounds = src.bounds

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

inside_image = (
    (cols >= 0) & (cols < width)
    & (rows >= 0) & (rows < height)
    & (xx >= bounds.left) & (xx < bounds.right)
    & (yy >= bounds.bottom) & (yy < bounds.top)
)

x = x[inside_image]
y = y[inside_image]
hag = hag[inside_image]
cols = cols[inside_image]
rows = rows[inside_image]

mask = np.asarray(Image.open(MASK_PATH).convert("L")) > 0

# A 20-pixel expansion around the mask defines the local comparison area.
# At the reconstructed pixel size, 20 pixels is about 65 feet.
expanded = np.zeros_like(mask)
r0 = max(0, int(rows.min())) if len(rows) else 0
c0 = max(0, int(cols.min())) if len(cols) else 0

mask_rows, mask_cols = np.where(mask)
if len(mask_rows) == 0:
    raise ValueError("Mask 13 is empty.")

r_min = max(0, int(mask_rows.min()) - 20)
r_max = min(height, int(mask_rows.max()) + 21)
c_min = max(0, int(mask_cols.min()) - 20)
c_max = min(width, int(mask_cols.max()) + 21)

expanded[r_min:r_max, c_min:c_max] = True
nearby_not_mask = expanded & ~mask

in_mask = mask[rows, cols]
near_mask = nearby_not_mask[rows, cols]

records = []

for label, selected in [
    ("Inside mask", in_mask),
    ("Nearby outside mask", near_mask),
]:
    values = hag[selected]
    values = values[np.isfinite(values)]

    records.append({
        "region": label,
        "building_point_count": int(len(values)),
        "hag_median_ft": (
            float(np.median(values)) if len(values) else None
        ),
        "hag_p10_ft": (
            float(np.percentile(values, 10)) if len(values) else None
        ),
        "hag_p90_ft": (
            float(np.percentile(values, 90)) if len(values) else None
        ),
        "hag_min_ft": (
            float(np.min(values)) if len(values) else None
        ),
        "hag_max_ft": (
            float(np.max(values)) if len(values) else None
        ),
    })

df = pd.DataFrame(records)
df.to_csv(OUT_PATH, index=False)

print("\nMASK 13 NEIGHBORHOOD COMPARISON")
print(df.to_string(index=False))
print("\nSaved:", OUT_PATH)
print(
    "\nCaution: This comparison is diagnostic only. Nearby points "
    "may belong to other buildings, and image registration remains "
    "unverified."
)
