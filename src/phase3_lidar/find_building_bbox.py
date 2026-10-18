
"""Find LiDAR bounding boxes containing building-class points."""

from pathlib import Path
import laspy
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
las_path = ROOT / "data/lidar/raw/autzen-classified.copc.laz"

# Work with a grid of 300 x 300 US-survey-foot cells.
cell_size = 300
counts = {}

with laspy.open(las_path) as reader:
    header = reader.header
    xmin, ymin = header.mins[0], header.mins[1]
    xmax, ymax = header.maxs[0], header.maxs[1]

    print("Scanning building-class points...")
    print(f"LiDAR X range: {xmin:.2f} to {xmax:.2f}")
    print(f"LiDAR Y range: {ymin:.2f} to {ymax:.2f}")

    for chunk in reader.chunk_iterator(500_000):
        building = np.asarray(chunk.classification) == 6
        if not building.any():
            continue

        x = np.asarray(chunk.x)[building]
        y = np.asarray(chunk.y)[building]

        ix = np.floor((x - xmin) / cell_size).astype(int)
        iy = np.floor((y - ymin) / cell_size).astype(int)

        for gx, gy in zip(ix, iy):
            counts[(gx, gy)] = counts.get((gx, gy), 0) + 1

print("\nTop 10 grid cells by building-class point count:")
for (gx, gy), count in sorted(
    counts.items(), key=lambda item: item[1], reverse=True
)[:10]:
    bxmin = xmin + gx * cell_size
    bymin = ymin + gy * cell_size
    print(
        f"Building points={count:6d} | "
        f"bounds=({bxmin:.2f}, {bymin:.2f}, "
        f"{bxmin + cell_size:.2f}, {bymin + cell_size:.2f})"
    )
