from pathlib import Path

import laspy
import numpy as np
import rasterio

ROOT = Path(__file__).resolve().parent
LAS_PATH = ROOT / "data/lidar/raw/autzen-classified.copc.laz"
DEM_PATH = ROOT / "data/lidar/processed/autzen_dem.tif"

XMIN, YMIN, XMAX, YMAX = (
    636477.79, 851882.15, 636777.79, 852182.15
)

building_x = []
building_y = []

print("Reading building-class LiDAR points...")

with laspy.open(LAS_PATH) as reader:
    for chunk in reader.chunk_iterator(500_000):
        classes = np.asarray(chunk.classification)
        x = np.asarray(chunk.x)
        y = np.asarray(chunk.y)

        selected = (
            (classes == 6)
            & (x >= XMIN) & (x <= XMAX)
            & (y >= YMIN) & (y <= YMAX)
        )

        if selected.any():
            building_x.extend(x[selected])
            building_y.extend(y[selected])

x = np.asarray(building_x)
y = np.asarray(building_y)

print(f"Building points selected: {x.size:,}")

print("\nChecking DEM coverage and nodata cells...")

with rasterio.open(DEM_PATH) as dem:
    rows, cols = rasterio.transform.rowcol(
        dem.transform, x, y
    )
    rows = np.asarray(rows)
    cols = np.asarray(cols)

    inside = (
        (rows >= 0) & (rows < dem.height)
        & (cols >= 0) & (cols < dem.width)
    )

    valid_mask = np.zeros(x.size, dtype=bool)
    valid_mask[inside] = (
        dem.read_masks(1)[rows[inside], cols[inside]] > 0
    )

    print(f"DEM dimensions: {dem.width} x {dem.height}")
    print(f"DEM nodata value: {dem.nodata}")
    print(f"Inside DEM extent: {inside.sum():,}")
    print(f"Outside DEM extent: {(~inside).sum():,}")
    print(f"Inside with valid raster mask: {valid_mask.sum():,}")
    print(
        "Inside on invalid raster mask: "
        f"{(inside & ~valid_mask).sum():,}"
    )

print("\nDiagnostic completed. No files were modified.")