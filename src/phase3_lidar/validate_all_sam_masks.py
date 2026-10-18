
from pathlib import Path
import json

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
MASK_DIR = (
    ROOT / "outputs" / "sam_georeferenced"
    / "individual_masks"
)
OUT_DIR = ROOT / "outputs" / "sam_georeferenced"

OUT_CSV = OUT_DIR / "all_sam_masks_lidar_overlap.csv"
OUT_JSON = OUT_DIR / "all_sam_masks_lidar_overlap.json"

for path in [LIDAR_PATH, IMAGE_PATH, MASK_DIR]:
    if not path.exists():
        raise FileNotFoundError(f"Missing input: {path}")

print("Reading aerial image metadata...")
with rasterio.open(IMAGE_PATH) as src:
    image_crs = src.crs
    transform = src.transform
    width = src.width
    height = src.height
    bounds = src.bounds

if image_crs is None:
    raise ValueError("The aerial image has no CRS.")

print("Reading LiDAR...")
las = laspy.read(LIDAR_PATH)

classification = np.asarray(las.classification)
building = classification == 6

x = np.asarray(las.x)[building]
y = np.asarray(las.y)[building]

hag = None
for dimension in [
    "height_above_ground",
    "HeightAboveGround",
]:
    if dimension in las.point_format.dimension_names:
        hag = np.asarray(getattr(las, dimension))[building]
        break

source_crs = las.header.parse_crs()
if source_crs is None:
    raise ValueError("Could not identify the LiDAR CRS.")

transformer = Transformer.from_crs(
    source_crs, image_crs, always_xy=True
)
x_img, y_img = transformer.transform(x, y)

inverse = ~transform
cols = np.floor(
    inverse.a * x_img + inverse.b * y_img + inverse.c
).astype(int)
rows = np.floor(
    inverse.d * x_img + inverse.e * y_img + inverse.f
).astype(int)

inside = (
    (cols >= 0) & (cols < width)
    & (rows >= 0) & (rows < height)
    & (x_img >= bounds.left) & (x_img < bounds.right)
    & (y_img >= bounds.bottom) & (y_img < bounds.top)
)

rows = rows[inside]
cols = cols[inside]

if hag is not None:
    hag = hag[inside]

print("Building-class points:", len(x))
print("Points inside image bounds:", len(rows))
print("Scanning SAM masks...")

records = []

mask_paths = sorted(MASK_DIR.glob("mask_*.png"))

for mask_path in mask_paths:
    try:
        mask_id = int(mask_path.stem.split("_")[-1])
    except ValueError:
        continue

    mask = np.asarray(
        Image.open(mask_path).convert("L")
    ) > 0

    if mask.shape != (height, width):
        print("Skipping unexpected mask dimensions:", mask_path.name)
        continue

    matched = mask[rows, cols]
    count = int(matched.sum())
    area = int(mask.sum())

    record = {
        "mask_id": mask_id,
        "mask_area_pixels": area,
        "building_points_inside_mask": count,
        "image_crs": str(image_crs),
        "registration_verified": False,
    }

    if count > 0:
        record["points_per_mask_pixel"] = round(
            count / area, 4
        ) if area else None

        if hag is not None:
            values = hag[matched]
            values = values[np.isfinite(values)]
            record["valid_hag_points"] = int(len(values))

            if len(values):
                record["hag_median_source_units"] = float(
                    np.median(values)
                )
                record["hag_p90_source_units"] = float(
                    np.percentile(values, 90)
                )

    records.append(record)

if not records:
    raise RuntimeError(
        "No masks were analysed. Check the mask directory."
    )

df = pd.DataFrame(records)
df = df.sort_values(
    "building_points_inside_mask", ascending=False
)

OUT_DIR.mkdir(parents=True, exist_ok=True)
df.to_csv(OUT_CSV, index=False)

OUT_JSON.write_text(
    json.dumps(
        df.replace({np.nan: None}).to_dict(orient="records"),
        indent=2,
    ),
    encoding="utf-8",
)

print("\nALL-MASK OVERLAP SUMMARY")
print("Masks analysed:", len(df))
print("Masks containing building points:", int(
    (df["building_points_inside_mask"] > 0).sum()
))
print("\nTop 15 masks by building-point count:")
print(df.head(15).to_string(index=False))

print("\nCSV:", OUT_CSV)
print("JSON:", OUT_JSON)
print(
    "\nCAUTION: Overlap is preliminary, not proof of rooftop identity. "
    "The image georeferencing has not been independently verified. "
    "HAG values are in the source LiDAR vertical units, which are feet."
)
