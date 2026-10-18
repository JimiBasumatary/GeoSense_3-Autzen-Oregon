
from pathlib import Path

import laspy
import numpy as np
import open3d as o3d

PROJECT_DIR = Path(__file__).resolve().parent.parent
INPUT_FILE = PROJECT_DIR / "data" / "lidar" / "raw" / "autzen-classified.copc.laz"
OUTPUT_DIR = PROJECT_DIR / "outputs" / "lidar_3d"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

las = laspy.read(INPUT_FILE)
xyz = np.column_stack((las.x, las.y, las.z)).astype(np.float64)
classes = np.asarray(las.classification)

# Local coordinates improve visualization stability.
origin = xyz.min(axis=0)

class_settings = {
    5: ("vegetation", [0.1, 0.75, 0.1]),
    6: ("buildings", [0.9, 0.2, 0.15]),
    9: ("water", [0.1, 0.45, 0.95]),
}

rng = np.random.default_rng(42)
max_points = 250_000

for code, (name, color) in class_settings.items():
    points = xyz[classes == code]
    count = len(points)

    if count == 0:
        print(f"Class {code}: no points found")
        continue

    if count > max_points:
        indices = rng.choice(count, max_points, replace=False)
        points = points[indices]

    points = points - origin

    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(points)
    cloud.paint_uniform_color(color)

    output_file = OUTPUT_DIR / f"autzen_{name}_3d.ply"
    o3d.io.write_point_cloud(str(output_file), cloud)

    print(f"Class {code} ({name}): {count:,} original points")
    print(f"Saved: {output_file}")

    o3d.visualization.draw_geometries(
        [cloud],
        window_name=f"Autzen LiDAR - {name.title()}",
        width=1200,
        height=800,
    )

print("CLASS-SPECIFIC VISUALIZATION COMPLETE")
