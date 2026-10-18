from pathlib import Path

import laspy
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

ROOT = Path(r"D:\GeoSense_Phase3")

LIDAR_PATH = (
    ROOT / "data" / "lidar" / "raw"
    / "autzen-classified.copc.laz"
)
OUT_CSV = (
    ROOT / "outputs" / "reports"
    / "building_nearest_ground_distances.csv"
)
OUT_JSON = (
    ROOT / "outputs" / "reports"
    / "building_nearest_ground_distance_summary.json"
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

print("Ground points:", len(ground_xy))
print("Building points:", len(building_xy))

if len(ground_xy) == 0 or len(building_xy) == 0:
    raise ValueError("Ground or building points are missing.")

# Sample up to 20,000 building points reproducibly.
rng = np.random.default_rng(42)
sample_count = min(20000, len(building_xy))
sample_indices = rng.choice(
    len(building_xy), size=sample_count, replace=False
)
sample_xy = building_xy[sample_indices]

print("Building sample:", sample_count)
print("Building nearest-ground search...")

tree = cKDTree(ground_xy)
distances, nearest_indices = tree.query(sample_xy, k=1)

# Coordinates use the source horizontal CRS, EPSG:2992, in feet.
summary = {
    "source_file": str(LIDAR_PATH.relative_to(ROOT)),
    "ground_class": 2,
    "building_class": 6,
    "building_sample_size": int(sample_count),
    "distance_units": "source horizontal coordinate units (feet)",
    "minimum_distance_ft": float(np.min(distances)),
    "median_distance_ft": float(np.median(distances)),
    "p90_distance_ft": float(np.percentile(distances, 90)),
    "p95_distance_ft": float(np.percentile(distances, 95)),
    "p99_distance_ft": float(np.percentile(distances, 99)),
    "maximum_distance_ft": float(np.max(distances)),
    "sampled_points_over_10ft": int(np.sum(distances > 10)),
    "sampled_points_over_25ft": int(np.sum(distances > 25)),
    "sampled_points_over_50ft": int(np.sum(distances > 50)),
}

records = pd.DataFrame({
    "sample_index": sample_indices,
    "x_ft": sample_xy[:, 0],
    "y_ft": sample_xy[:, 1],
    "nearest_ground_distance_ft": distances,
})

OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
records.to_csv(OUT_CSV, index=False)

import json
OUT_JSON.write_text(
    json.dumps(summary, indent=2),
    encoding="utf-8",
)

print("\nNEAREST-GROUND DISTANCE SUMMARY")
for key, value in summary.items():
    print(f"{key}: {value}")

print("\nCSV:", OUT_CSV)
print("JSON:", OUT_JSON)
print(
    "\nCaution: Horizontal distance alone does not establish HAG "
    "accuracy. This is a reproducible sample, not the full cloud."
)
