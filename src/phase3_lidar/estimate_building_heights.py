
"""Estimate building-point height above the DEM for one bbox."""

from pathlib import Path
import json

import laspy
import numpy as np
import rasterio
from rasterio.windows import from_bounds

ROOT = Path(__file__).resolve().parents[2]

LAS_PATH = ROOT / "data/lidar/raw/autzen-classified.copc.laz"
DEM_PATH = ROOT / "data/lidar/processed/autzen_dem.tif"
OUT_PATH = ROOT / "outputs/reports/building_height_estimate.json"

# Building-rich area identified by the grid scan; EPSG:2992.
xmin, ymin, xmax, ymax = (
    636477.79, 851882.15, 636777.79, 852182.15
)

# Read building-class points inside the bounding box.
building_x = []
building_y = []
building_z = []

with laspy.open(LAS_PATH) as reader:
    for chunk in reader.chunk_iterator(500_000):
        cls = np.asarray(chunk.classification)
        x = np.asarray(chunk.x)
        y = np.asarray(chunk.y)
        z = np.asarray(chunk.z)

        mask = (
            (cls == 6)
            & (x >= xmin) & (x <= xmax)
            & (y >= ymin) & (y <= ymax)
        )

        if mask.any():
            building_x.extend(x[mask])
            building_y.extend(y[mask])
            building_z.extend(z[mask])

x = np.asarray(building_x)
y = np.asarray(building_y)
z = np.asarray(building_z)

if z.size == 0:
    raise RuntimeError("No building-class points found in the bbox.")

# Sample the DEM at each building point's location.
with rasterio.open(DEM_PATH) as dem:
    rows, cols = rasterio.transform.rowcol(dem.transform, x, y)
    rows = np.asarray(rows)
    cols = np.asarray(cols)

    inside = (
        (rows >= 0) & (rows < dem.height)
        & (cols >= 0) & (cols < dem.width)
    )

    ground = np.full(z.shape, np.nan, dtype=float)
    valid_indices = np.where(inside)[0]

    values = dem.read(1)[rows[inside], cols[inside]].astype(float)
    nodata = dem.nodata

    valid = np.isfinite(values)
    if nodata is not None:
        valid &= values != nodata

    ground[valid_indices[valid]] = values[valid]

valid = np.isfinite(ground)
height = z[valid] - ground[valid]

# Do not silently discard negative values; report their count.
result = {
    "study_area": "Autzen building-rich bbox",
    "coordinate_system": "EPSG:2992 horizontal; elevations in US survey feet",
    "bounding_box": {
        "xmin": xmin, "ymin": ymin, "xmax": xmax, "ymax": ymax
    },
    "method": "Building-class point Z minus DEM elevation at point location",
    "building_class_point_count": int(z.size),
    "points_with_valid_dem": int(valid.sum()),
    "points_without_valid_dem": int((~valid).sum()),
    "negative_height_estimate_count": int((height < 0).sum()),
    "height_estimate_ft": None,
    "limitations": [
        "DEM is an interpolated terrain model; sampling may not represent the exact ground under buildings.",
        "Point-wise estimates are not individual building footprints or validated building heights.",
        "Building points and DEM cells may have horizontal or classification errors.",
        "Inspect the distribution and validate representative buildings before interpreting results."
    ]
}

if height.size:
    result["height_estimate_ft"] = {
        "minimum": float(np.min(height)),
        "median": float(np.median(height)),
        "mean": float(np.mean(height)),
        "95th_percentile": float(np.percentile(height, 95)),
        "maximum": float(np.max(height))
    }

OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
OUT_PATH.write_text(json.dumps(result, indent=2), encoding="utf-8")

print(json.dumps(result, indent=2))
print(f"\nSaved: {OUT_PATH}")
