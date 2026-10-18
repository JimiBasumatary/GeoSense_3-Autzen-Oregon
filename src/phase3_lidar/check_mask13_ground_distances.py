
from pathlib import Path

import laspy
import numpy as np
import pandas as pd
import rasterio
from PIL import Image
from pyproj import Transformer
from scipy.spatial import cKDTree

ROOT = Path(r"D:\GeoSense_Phase3")

LIDAR_PATH = (
    ROOT / "data" / "lidar" / "raw"
    / "autzen-classified.copc.laz"
)
IMAGE_PATH = (
    ROOT / "data" / "aerial"
    / "autzen_naip_georeferenced_fixed.tif"
)
MASK_PATH = (
    ROOT / "outputs" / "sam_georeferenced"
    / "individual_masks" / "mask_0013.png"
)
OUT_CSV = (
    ROOT / "outputs" / "reports"
    / "mask13_ground_distance_sample.csv"
)
OUT_JSON = (
    ROOT / "outputs" / "reports"
    / "mask13_ground_distance_summary.json"
)

print("Reading LiDAR...")
las = laspy.read(LIDAR_PATH)

classes = np.asarray(las.classification)
x = np.asarray(las.x)
y = np.asarray(las.y)

ground = classes == 2
building = classes == 6

ground_xy = np.column_stack((x[ground], y[ground]))
building_xy = np.column_stack((x[building], y[building]))

print("Building points:", len(building_xy))
print("Ground points:", len(ground_xy))

print("Mapping building points to aerial-image pixels...")
with rasterio.open(IMAGE_PATH) as src:
    transform = src.transform
    image_crs = src.crs
    width, height = src.width, src.height
    bounds = src.bounds

source_crs = las.header.parse_crs()
transformer = Transformer.from_crs(
    source_crs, image_crs, always_xy=True
)
xx, yy = transformer.transform(
    building_xy[:, 0], building_xy[:, 1]
)

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

mask = np.asarray(
    Image.open(MASK_PATH).convert("L")
) > 0

inside_indices = np.flatnonzero(inside_image)
mask_match = mask[rows[inside_image], cols[inside_image]]
inside_mask_indices = inside_indices[mask_match]

mask_rows, mask_cols = np.where(mask)
r_min = max(0, int(mask_rows.min()) - 20)
r_max = min(height, int(mask_rows.max()) + 21)
c_min = max(0, int(mask_cols.min()) - 20)
c_max = min(width, int(mask_cols.max()) + 21)

nearby = (
    inside_image
    & (rows >= r_min) & (rows < r_max)
    & (cols >= c_min) & (cols < c_max)
)
nearby_indices = np.flatnonzero(nearby)
nearby_mask = mask[rows[nearby], cols[nearby]]
outside_nearby_indices = nearby_indices[~nearby_mask]

print("Mask 13 building points:", len(inside_mask_indices))
print("Nearby outside-mask points:", len(outside_nearby_indices))

tree = cKDTree(ground_xy)
rng = np.random.default_rng(13)

# Sample up to 20,000 points per region.
groups = [
    ("Inside Mask 13", inside_mask_indices),
    ("Nearby outside Mask 13", outside_nearby_indices),
]
records = []
summary = {}

for label, indices in groups:
    n = min(20000, len(indices))
    if n == 0:
        print("No points for:", label)
        continue

    chosen = rng.choice(indices, size=n, replace=False)
    query_xy = building_xy[chosen]

    distances, _ = tree.query(query_xy, k=1)

    summary[label] = {
        "sample_size": int(n),
        "median_distance_ft": float(np.median(distances)),
        "p90_distance_ft": float(np.percentile(distances, 90)),
        "p95_distance_ft": float(np.percentile(distances, 95)),
        "p99_distance_ft": float(np.percentile(distances, 99)),
        "max_sampled_distance_ft": float(np.max(distances)),
        "over_25ft": int(np.sum(distances > 25)),
        "over_50ft": int(np.sum(distances > 50)),
    }

    for point_index, distance in zip(chosen, distances):
        records.append({
            "region": label,
            "point_index": int(point_index),
            "x_ft": float(building_xy[point_index, 0]),
            "y_ft": float(building_xy[point_index, 1]),
            "nearest_ground_distance_ft": float(distance),
        })

OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
pd.DataFrame(records).to_csv(OUT_CSV, index=False)
OUT_JSON.write_text(
    __import__("json").dumps(summary, indent=2),
    encoding="utf-8",
)

print("\nNEAREST-GROUND DISTANCE COMPARISON")
for label, values in summary.items():
    print(f"\n{label}")
    for key, value in values.items():
        print(f"  {key}: {value}")

print("\nCSV:", OUT_CSV)
print("JSON:", OUT_JSON)
print(
    "\nCaution: Distance is measured to the nearest class-2 point "
    "in horizontal source coordinates. It does not establish HAG "
    "accuracy or validate image registration."
)
