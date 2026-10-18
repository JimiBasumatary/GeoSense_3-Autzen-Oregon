
from pathlib import Path
import json

import laspy
import numpy as np
import pandas as pd
import rasterio
from pyproj import Transformer
from PIL import Image

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
OUT_CSV = OUT_DIR / "rooftop_candidate_lidar_validation.csv"
OUT_JSON = OUT_DIR / "rooftop_candidate_lidar_validation.json"

CANDIDATE_IDS = [36, 32]

for path in [LIDAR_PATH, IMAGE_PATH]:
    if not path.exists():
        raise FileNotFoundError(f"Missing input: {path}")

# Read the raster's actual affine transform.
with rasterio.open(IMAGE_PATH) as src:
    image_crs = src.crs
    transform = src.transform
    width = src.width
    height = src.height
    bounds = src.bounds

if image_crs is None:
    raise ValueError("The aerial raster has no CRS.")

# Read LiDAR points and HAG values.
print("Reading LiDAR with HAG...")
las = laspy.read(LIDAR_PATH)

classification = np.asarray(las.classification)
building = classification == 6

x = np.asarray(las.x)[building]
y = np.asarray(las.y)[building]
z = np.asarray(las.z)[building]

hag = None
if "height_above_ground" in las.point_format.dimension_names:
    hag = np.asarray(las.height_above_ground)[building]
elif "HeightAboveGround" in las.point_format.dimension_names:
    hag = np.asarray(las.HeightAboveGround)[building]

print("Total points:", len(classification))
print("Building-class points:", len(x))
print("Image CRS:", image_crs)
print("Image bounds:", bounds)

# Transform source horizontal coordinates into raster CRS if needed.
source_crs = las.header.parse_crs()
if source_crs is None:
    raise ValueError("The LiDAR file has no parseable CRS.")

transformer = Transformer.from_crs(
    source_crs, image_crs, always_xy=True
)
x_img, y_img = transformer.transform(x, y)

# Convert map coordinates to raster pixel indices.
cols, rows = rasterio.transform.rowcol(
    transform, x_img, y_img
)
cols = np.asarray(cols)
rows = np.asarray(rows)

inside = (
    (x_img >= bounds.left) & (x_img < bounds.right)
    & (y_img >= bounds.bottom) & (y_img < bounds.top)
    & (cols >= 0) & (cols < width)
    & (rows >= 0) & (rows < height)
)

print("Building points within raster bounds:", int(inside.sum()))

x_img = np.asarray(x_img)[inside]
y_img = np.asarray(y_img)[inside]
cols = cols[inside]
rows = rows[inside]
z = z[inside]

if hag is not None:
    hag = hag[inside]

records = []

for mask_id in CANDIDATE_IDS:
    mask_path = MASK_DIR / f"mask_{mask_id:04d}.png"
    if not mask_path.exists():
        print("Missing mask:", mask_path)
        continue

    mask = np.asarray(Image.open(mask_path).convert("L")) > 0

    # Select building-class LiDAR points that fall within this mask.
    matched = mask[rows, cols]
    count = int(matched.sum())

    record = {
        "mask_id": mask_id,
        "mask_area_pixels": int(mask.sum()),
        "building_points_inside_mask": count,
        "image_crs": str(image_crs),
        "registration_verified": False,
    }

    if count > 0:
        record["z_min"] = float(np.min(z[matched]))
        record["z_median"] = float(np.median(z[matched]))
        record["z_max"] = float(np.max(z[matched]))

        if hag is not None:
            valid_hag = hag[matched & np.isfinite(hag)]
            record["valid_hag_points"] = int(len(valid_hag))
            if len(valid_hag) > 0:
                record["hag_median_source_units"] = float(
                    np.median(valid_hag)
                )
                record["hag_p90_source_units"] = float(
                    np.percentile(valid_hag, 90)
                )
    else:
        record["note"] = "No class 6 points fell inside this mask."

    records.append(record)

result = pd.DataFrame(records)
result.to_csv(OUT_CSV, index=False)
OUT_JSON.write_text(
    json.dumps(records, indent=2),
    encoding="utf-8",
)

print("\nPRELIMINARY CANDIDATE CHECK")
print(result.to_string(index=False))
print("\nCSV:", OUT_CSV)
print("JSON:", OUT_JSON)
print(
    "\nIMPORTANT: These overlap counts and height summaries are "
    "preliminary. They do not prove accurate image-to-LiDAR "
    "registration or establish a building's roof height."
)
