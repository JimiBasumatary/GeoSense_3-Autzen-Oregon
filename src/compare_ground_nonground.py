from pathlib import Path

import laspy
import numpy as np
import open3d as o3d

PROJECT_DIR = Path(__file__).resolve().parent.parent
INPUT_FILE = PROJECT_DIR / "data" / "lidar" / "raw" / "autzen-classified.copc.laz"
OUTPUT_DIR = PROJECT_DIR / "outputs" / "lidar_3d"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("Reading classified LiDAR data...")
las = laspy.read(INPUT_FILE)

xyz = np.column_stack((las.x, las.y, las.z)).astype(np.float64)
classification = np.asarray(las.classification)

# Class 2 represents ground points.
ground_mask = classification == 2
nonground_mask = ~ground_mask

rng = np.random.default_rng(42)
max_each = 250_000

def make_cloud(points, color):
    cloud = o3d.geometry.PointCloud()
    if len(points) > max_each:
        points = points[rng.choice(len(points), max_each, replace=False)]

    # Shift coordinates to a local origin.
    points = points - xyz.min(axis=0)

    cloud.points = o3d.utility.Vector3dVector(points)
    cloud.paint_uniform_color(color)
    return cloud

ground_cloud = make_cloud(xyz[ground_mask], [0.1, 0.8, 0.2])
nonground_cloud = make_cloud(xyz[nonground_mask], [0.9, 0.2, 0.2])

ground_path = OUTPUT_DIR / "autzen_ground_3d.ply"
nonground_path = OUTPUT_DIR / "autzen_nonground_3d.ply"

o3d.io.write_point_cloud(str(ground_path), ground_cloud)
o3d.io.write_point_cloud(str(nonground_path), nonground_cloud)

print(f"Total points: {len(xyz):,}")
print(f"Ground points (class 2): {ground_mask.sum():,}")
print(f"Non-ground points (other classes): {nonground_mask.sum():,}")
print(f"Saved: {ground_path}")
print(f"Saved: {nonground_path}")

print("Opening 3D comparison...")
o3d.visualization.draw_geometries(
    [ground_cloud, nonground_cloud],
    window_name="Autzen LiDAR: Ground (Green) and Non-ground (Red)",
    width=1200,
    height=800
)