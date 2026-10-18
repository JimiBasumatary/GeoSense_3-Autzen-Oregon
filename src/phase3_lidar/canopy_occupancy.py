
"""Exploratory LiDAR vegetation-canopy occupancy indicator."""

from pathlib import Path
import json

import laspy
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
LIDAR_FILE = ROOT / "data/lidar/raw/autzen-classified.copc.laz"

# Existing 300 ft x 300 ft example area, EPSG:2992.
BOUNDS = (637000.0, 850500.0, 637300.0, 850800.0)
CELL_SIZE_FT = 3.0


def calculate_canopy_occupancy(bounds=BOUNDS, cell_size_ft=CELL_SIZE_FT):
    xmin, ymin, xmax, ymax = map(float, bounds)

    if not np.all(np.isfinite([xmin, ymin, xmax, ymax])):
        raise ValueError("Bounds must be finite numbers.")
    if xmin >= xmax or ymin >= ymax:
        raise ValueError("Invalid bounds.")
    if not np.isfinite(cell_size_ft) or cell_size_ft <= 0:
        raise ValueError("Cell size must be a positive finite number.")

    width = (xmax - xmin) / cell_size_ft
    height = (ymax - ymin) / cell_size_ft

    if not np.isclose(width, round(width)) or not np.isclose(
        height, round(height)
    ):
        raise ValueError("Bounds dimensions must be multiples of cell size.")

    cols = int(round(width))
    rows = int(round(height))
    observed = np.zeros((rows, cols), dtype=bool)
    vegetation = np.zeros((rows, cols), dtype=bool)

    if not LIDAR_FILE.exists():
        raise FileNotFoundError(f"LiDAR file not found: {LIDAR_FILE}")

    total_selected = 0
    vegetation_points = 0

    with laspy.open(LIDAR_FILE) as reader:
        for points in reader.chunk_iterator(500_000):
            x = np.asarray(points.x)
            y = np.asarray(points.y)
            classification = np.asarray(points.classification)

            inside = (
                (x >= xmin) & (x < xmax)
                & (y >= ymin) & (y < ymax)
            )
            if not np.any(inside):
                continue

            x = x[inside]
            y = y[inside]
            classes = classification[inside]
            total_selected += len(x)

            col = np.floor((x - xmin) / cell_size_ft).astype(int)
            row = np.floor((ymax - y) / cell_size_ft).astype(int)

            valid = (
                (row >= 0) & (row < rows)
                & (col >= 0) & (col < cols)
            )
            row, col, classes = row[valid], col[valid], classes[valid]

            observed[row, col] = True

            is_vegetation = classes == 5
            vegetation_points += int(is_vegetation.sum())
            vegetation[row[is_vegetation], col[is_vegetation]] = True

    observed_cells = int(observed.sum())
    vegetation_cells = int((vegetation & observed).sum())

    occupancy_percent = (
        100.0 * vegetation_cells / observed_cells
        if observed_cells else None
    )

    return {
        "study_area": "Autzen example bounding box",
        "coordinate_system": "EPSG:2992; horizontal units are feet",
        "bounds_ft": {
            "xmin": xmin, "ymin": ymin,
            "xmax": xmax, "ymax": ymax,
        },
        "cell_size_ft": cell_size_ft,
        "grid_dimensions": {"rows": rows, "columns": cols},
        "total_selected_lidar_points": total_selected,
        "class_5_high_vegetation_points": vegetation_points,
        "observed_cells": observed_cells,
        "cells_with_class_5_vegetation": vegetation_cells,
        "vegetation_cell_occupancy_percent": occupancy_percent,
        "interpretation": (
            "Percentage of observed grid cells containing at least one "
            "Class 5 return. This is an exploratory occupancy proxy, "
            "not validated canopy-cover percentage."
        ),
        "limitations": [
            "Unobserved cells are excluded from the denominator.",
            "Results depend on grid resolution and LiDAR sampling density.",
            "Class 5 returns may represent high vegetation, not only tree canopy.",
            "This metric does not measure canopy density or canopy area directly.",
            "Independent validation is required before reporting canopy cover."
        ],
    }


if __name__ == "__main__":
    result = calculate_canopy_occupancy()
    output_dir = ROOT / "outputs/reports"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "canopy_occupancy_example.json"
    output_file.write_text(json.dumps(result, indent=2), encoding="utf-8")

    print(json.dumps(result, indent=2))
    print(f"\nSaved to: {output_file}")
