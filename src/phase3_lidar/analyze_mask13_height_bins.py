
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
    / "mask13_spatial_height_bins.csv"
)

print("Reading Mask 13...")
mask = np.asarray(Image.open(MASK_PATH).convert("L")) > 0
mask_rows, mask_cols = np.where(mask)

if len(mask_rows) == 0:
    raise ValueError("Mask 13 is empty.")

r_min, r_max = int(mask_rows.min()), int(mask_rows.max())
c_min, c_max = int(mask_cols.min()), int(mask_cols.max())

print("Reading image metadata...")
with rasterio.open(IMAGE_PATH) as src:
    transform = src.transform
    image_crs = src.crs
    width, height = src.width, src.height

print("Reading LiDAR...")
las = laspy.read(LIDAR_PATH)

classification = np.asarray(las.classification)
building = classification == 6

x = np.asarray(las.x)[building]
y = np.asarray(las.y)[building]
hag = np.asarray(las.HeightAboveGround)[building]

transformer = Transformer.from_crs(
    las.header.parse_crs(), image_crs, always_xy=True
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
)

cols = cols[inside_image]
rows = rows[inside_image]
hag = hag[inside_image]

inside_mask = mask[rows, cols]
cols = cols[inside_mask]
rows = rows[inside_mask]
hag = hag[inside_mask]

print("Class-6 points inside mask:", len(hag))

# Divide the mask's bounding rectangle into a 4-by-4 grid.
records = []
n_bins = 4

for row_bin in range(n_bins):
    for col_bin in range(n_bins):
        rb0 = r_min + (r_max - r_min + 1) * row_bin // n_bins
        rb1 = r_min + (r_max - r_min + 1) * (row_bin + 1) // n_bins
        cb0 = c_min + (c_max - c_min + 1) * col_bin // n_bins
        cb1 = c_min + (c_max - c_min + 1) * (col_bin + 1) // n_bins

        selected = (
            (rows >= rb0) & (rows < rb1)
            & (cols >= cb0) & (cols < cb1)
        )

        values = hag[selected]
        values = values[np.isfinite(values)]

        records.append({
            "row_bin": row_bin + 1,
            "column_bin": col_bin + 1,
            "point_count": int(len(values)),
            "hag_median_ft": (
                float(np.median(values)) if len(values) else None
            ),
            "hag_p10_ft": (
                float(np.percentile(values, 10)) if len(values) else None
            ),
            "hag_p90_ft": (
                float(np.percentile(values, 90)) if len(values) else None
            ),
        })

df = pd.DataFrame(records)
df.to_csv(OUT_PATH, index=False)

print("\nMASK 13: 4 x 4 SPATIAL HAG SUMMARY")
print(df.to_string(index=False))
print("\nSaved:", OUT_PATH)
print(
    "\nCaution: Empty or low-count cells may occur. This grid uses "
    "the mask bounding rectangle, so cells can include pixels outside "
    "the actual roof-shaped mask. Registration remains unverified."
)
