
from pathlib import Path
import csv
import json

import laspy
import numpy as np
import pandas as pd
from pyproj import Transformer
from PIL import Image

ROOT = Path(r"D:\GeoSense_Phase3")

LIDAR_PATH = ROOT / "data" / "lidar" / "raw" / "autzen-classified.copc.laz"
IMAGE_PATH = ROOT / "data" / "aerial" / "autzen_naip_preview.jpg"
METADATA_PATH = ROOT / "data" / "aerial" / "autzen_naip_preview_metadata.json"
MASK_DIR = ROOT / "outputs" / "sam_rooftop" / "individual_masks"
MASK_CSV = ROOT / "outputs" / "sam_rooftop" / "autzen_sam_masks.csv"

OUT_DIR = ROOT / "outputs" / "sam_rooftop"
OUTPUT_CSV = OUT_DIR / "sam_lidar_building_overlap.csv"
OUTPUT_JSON = OUT_DIR / "sam_lidar_overlap_metadata.json"

# Verify input files.
for path in [LIDAR_PATH, IMAGE_PATH, METADATA_PATH, MASK_CSV]:
    if not path.exists():
        raise FileNotFoundError(f"Missing input: {path}")

metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
width = int(metadata["width"])
height = int(metadata["height"])
extent = metadata["extent"]

xmin = float(extent["xmin"])
xmax = float(extent["xmax"])
ymin = float(extent["ymin"])
ymax = float(extent["ymax"])

# Transform source LiDAR coordinates into geographic coordinates.
transformer = Transformer.from_crs(
    "EPSG:2992", "EPSG:4326", always_xy=True
)

print("Reading LiDAR building points...")
with laspy.open(LIDAR_PATH) as reader:
    las = reader.read()

classification = np.asarray(las.classification)
building = classification == 6

x = np.asarray(las.x)[building]
y = np.asarray(las.y)[building]

print("Total LiDAR points:", len(classification))
print("Class 6 building points:", len(x))

longitude, latitude = transformer.transform(x, y)

# Convert geographic coordinates into image pixel coordinates.
# Image row 0 is at the northern edge.
col = np.floor((longitude - xmin) / (xmax - xmin) * width).astype(int)
row = np.floor((ymax - latitude) / (ymax - ymin) * height).astype(int)

inside = (
    (longitude >= xmin) & (longitude <= xmax)
    & (latitude >= ymin) & (latitude <= ymax)
    & (col >= 0) & (col < width)
    & (row >= 0) & (row < height)
)

col = col[inside]
row = row[inside]

print("Building points inside image extent:", len(col))

masks = pd.read_csv(MASK_CSV)
records = []

for item in masks.itertuples(index=False):
    mask_id = int(item.mask_id)
    mask_path = MASK_DIR / f"mask_{mask_id:04d}.png"

    if not mask_path.exists():
        print("Missing mask:", mask_path)
        continue

    mask = np.asarray(Image.open(mask_path).convert("L")) > 0

    # Count points whose projected image pixels lie inside this mask.
    point_count = int(mask[row, col].sum())

    records.append({
        "mask_id": mask_id,
        "mask_area_pixels": int(item.area_pixels),
        "predicted_iou": float(item.predicted_iou),
        "stability_score": float(item.stability_score),
        "building_points_in_mask": point_count,
    })

result = pd.DataFrame(records).sort_values(
    "building_points_in_mask", ascending=False
)
result.to_csv(OUTPUT_CSV, index=False)

summary = {
    "source_lidar": str(LIDAR_PATH),
    "source_crs_assumed": "EPSG:2992",
    "image_crs": metadata["crs"],
    "image_width": width,
    "image_height": height,
    "building_class_code": 6,
    "total_building_points": int(building.sum()),
    "building_points_inside_image_extent": int(len(col)),
    "number_of_masks_checked": len(records),
    "output_csv": str(OUTPUT_CSV),
    "warning": (
        "Preliminary overlap only. Assumes EPSG:2992 source coordinates "
        "and an axis-aligned image extent. The preview extent and "
        "LiDAR-to-image registration have not been independently "
        "validated. Counts do not prove that a mask is a rooftop."
    ),
}
OUTPUT_JSON.write_text(json.dumps(summary, indent=2), encoding="utf-8")

print("\nOverlap check complete.")
print(result.head(15).to_string(index=False))
print("\nCSV:", OUTPUT_CSV)
print("Metadata:", OUTPUT_JSON)
print("\nIMPORTANT: These are preliminary overlap counts, not roof labels.")
