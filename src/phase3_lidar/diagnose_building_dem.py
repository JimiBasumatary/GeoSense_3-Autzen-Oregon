
from pathlib import Path

import laspy
import numpy as np
import rasterio

ROOT = Path(__file__).resolve().parents[2]
LAS_PATH = ROOT / "data/lidar/raw/autzen-classified.copc.laz"
DEM_PATH = ROOT / "data/lidar/processed/autzen_dem.tif"

xmin, ymin, xmax, ymax = (
    636477.79, 851882.15, 636777.79, 852182.15
)

xs, ys, zs = [], [], []

with laspy.open(LAS_PATH) as reader:
    for chunk in reader.chunk_iterator(500_000):
        x = np.asarray(chunk.x)
        y = np.asarray(chunk.y)
        z = np.asarray(chunk.z)
        cls = np.asarray(chunk.classification)

        mask = (
            (cls == 6)
            & (x >= xmin) & (x <= xmax)
            & (y >= ymin) & (y <= ymax)
        )

        if mask.any():
            xs.extend(x[mask])
            ys.extend(y[mask])
            zs.extend(z[mask])

x = np.asarray(xs)
y = np.asarray(ys)

with rasterio.open(DEM_PATH) as dem:
    rows, cols = rasterio.transform.rowcol(dem.transform, x, y)
    rows = np.asarray(rows)
    cols = np.asarray(cols)

    inside = (
        (rows >= 0) & (rows < dem.height)
        & (cols >= 0) & (cols < dem.width)
    )

    print("Building-class points:", len(x))
    print("Points inside DEM raster:", int(inside.sum()))
    print("Points outside DEM raster:", int((~inside).sum()))

    sampled = np.full(len(x), np.nan)
    indices = np.flatnonzero(inside)

    band = dem.read(1)
    values = band[rows[inside], cols[inside]].astype(float)

    valid = np.isfinite(values)
    if dem.nodata is not None:
        valid &= values != dem.nodata

    sampled[indices[valid]] = values[valid]

    print("Points on valid DEM cells:", int(np.isfinite(sampled).sum()))
    print("Points on NoData cells:", int((inside & ~np.isfinite(sampled)).sum()))

    if np.isfinite(sampled).any():
        print(
            "Valid sampled DEM range (ft):",
            float(np.nanmin(sampled)),
            "to",
            float(np.nanmax(sampled))
        )
