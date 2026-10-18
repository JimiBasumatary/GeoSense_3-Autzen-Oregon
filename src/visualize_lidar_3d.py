
from pathlib import Path

import laspy
import numpy as np
import open3d as o3d

# Project paths
PROJECT_DIR = Path(__file__).resolve().parent.parent
INPUT_FILE = PROJECT_DIR / "data" / "lidar" / "raw" / "autzen-classified.copc.laz"
OUTPUT_DIR = PROJECT_DIR / "outputs" / "lidar_3d"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("Loading LiDAR point cloud...")
print(f"Input: {INPUT_FILE}")

# Read the classified LiDAR file
las = laspy.read(INPUT_FILE)

# Extract XYZ coordinates
xyz = np.column_stack((las.x, las.y, las.z)).astype(np.float64)

# Sample points for visualization to reduce memory usage
max_points = 500_000

if len(xyz) > max_points:
    rng = np.random.default_rng(42)
    indices = rng.choice(len(xyz), size=max_points, replace=False)
    xyz = xyz[indices]

# Shift coordinates to a local origin for stable 3D display
xyz_local = xyz - xyz.min(axis=0)

# Create Open3D point cloud
point_cloud = o3d.geometry.PointCloud()
point_cloud.points = o3d.utility.Vector3dVector(xyz_local)

# Colour points by elevation
z = xyz_local[:, 2]
z_min, z_max = z.min(), z.max()

if z_max > z_min:
    normalized_z = (z - z_min) / (z_max - z_min)
else:
    normalized_z = np.zeros_like(z)

colors = np.column_stack((
    normalized_z,
    0.35 * np.ones_like(normalized_z),
    1 - normalized_z
))
point_cloud.colors = o3d.utility.Vector3dVector(colors)

# Save the sampled point cloud
output_file = OUTPUT_DIR / "autzen_lidar_3d.ply"
o3d.io.write_point_cloud(str(output_file), point_cloud)

print(f"Total input points: {len(las.points):,}")
print(f"Points visualized: {len(xyz_local):,}")
print(f"Saved point cloud: {output_file}")

# Open interactive 3D viewer
print("Opening the 3D viewer...")
o3d.visualization.draw_geometries(
    [point_cloud],
    window_name="Autzen Stadium LiDAR - 3D Elevation View",
    width=1200,
    height=800
)
