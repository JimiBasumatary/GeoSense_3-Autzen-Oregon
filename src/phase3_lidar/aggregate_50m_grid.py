
"""Aggregate Autzen LiDAR building heights and vegetation occupancy to a 50 m grid.

Source CRS: EPSG:2992 (horizontal coordinates in feet).
Grid size: 50 metres, converted to US survey feet for grid calculations.
Original LiDAR and HAG files are read-only inputs.
"""

from pathlib import Path
import json

import laspy
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]

HAG_FILE = ROOT / "data/lidar/processed/autzen_classified_hag.laz"
OUTPUT_DIR = ROOT / "outputs/reports"
OUTPUT_CSV = OUTPUT_DIR / "autzen_50m_grid_metrics.csv"
OUTPUT_JSON = OUTPUT_DIR / "autzen_50m_grid_summary.json"

GRID_SIZE_M = 50.0
# EPSG:2992 uses US survey feet: 1 metre = 3937/1200 feet.
GRID_SIZE_FT = GRID_SIZE_M * (3937.0 / 1200.0)
FT_TO_M = 1200.0 / 3937.0
CHUNK_SIZE = 500_000


def main():
    if not HAG_FILE.exists():
        raise FileNotFoundError(f"Required HAG file not found: {HAG_FILE}")

    # First pass: determine the extent of valid XY points.
    xmin = ymin = float("inf")
    xmax = ymax = float("-inf")
    total_points = 0

    with laspy.open(HAG_FILE) as reader:
        for points in reader.chunk_iterator(CHUNK_SIZE):
            x = np.asarray(points.x)
            y = np.asarray(points.y)
            finite = np.isfinite(x) & np.isfinite(y)
            if not finite.any():
                continue

            xmin = min(xmin, float(x[finite].min()))
            ymin = min(ymin, float(y[finite].min()))
            xmax = max(xmax, float(x[finite].max()))
            ymax = max(ymax, float(y[finite].max()))
            total_points += int(finite.sum())

    if total_points == 0:
        raise RuntimeError("No finite LiDAR coordinates were found.")

    # Anchor the grid to the dataset's lower-left extent.
    # Cell indices are consistent throughout the second pass.
    ncols = int(np.floor((xmax - xmin) / GRID_SIZE_FT)) + 1
    nrows = int(np.floor((ymax - ymin) / GRID_SIZE_FT)) + 1

    records = {}

    # Second pass: aggregate per grid cell without loading all points at once.
    with laspy.open(HAG_FILE) as reader:
        has_hag = "HeightAboveGround" in reader.header.point_format.dimension_names
        if not has_hag:
            raise RuntimeError(
                "The input file does not contain HeightAboveGround. "
                "Check the HAG file's dimension names before proceeding."
            )

        for points in reader.chunk_iterator(CHUNK_SIZE):
            x = np.asarray(points.x)
            y = np.asarray(points.y)
            z = np.asarray(points.z)
            cls = np.asarray(points.classification)
            hag = np.asarray(points["HeightAboveGround"], dtype=float)

            finite = (
                np.isfinite(x) & np.isfinite(y)
                & np.isfinite(z) & np.isfinite(hag)
            )
            if not finite.any():
                continue

            x, y, cls, hag = x[finite], y[finite], cls[finite], hag[finite]

            col = np.floor((x - xmin) / GRID_SIZE_FT).astype(np.int64)
            row = np.floor((y - ymin) / GRID_SIZE_FT).astype(np.int64)

            valid = (
                (row >= 0) & (row < nrows)
                & (col >= 0) & (col < ncols)
            )
            row, col, cls, hag = row[valid], col[valid], cls[valid], hag[valid]

            if len(row) == 0:
                continue

            cell_ids = row * ncols + col

            for cell_id in np.unique(cell_ids):
                in_cell = cell_ids == cell_id
                cell_classes = cls[in_cell]
                cell_hag = hag[in_cell]

                item = records.setdefault(
                    int(cell_id),
                    {
                        "total_points": 0,
                        "ground_points": 0,
                        "vegetation_points": 0,
                        "building_points": 0,
                        "building_hag_m": [],
                        "vegetation_hag_m": [],
                    },
                )

                item["total_points"] += int(in_cell.sum())
                item["ground_points"] += int((cell_classes == 2).sum())
                item["vegetation_points"] += int((cell_classes == 5).sum())
                item["building_points"] += int((cell_classes == 6).sum())

                building_mask = (cell_classes == 6) & np.isfinite(cell_hag)
                if building_mask.any():
                    item["building_hag_m"].extend(
                        (cell_hag[building_mask] * FT_TO_M).tolist()
                    )

                vegetation_mask = (cell_classes == 5) & np.isfinite(cell_hag)
                if vegetation_mask.any():
                    item["vegetation_hag_m"].extend(
                        (cell_hag[vegetation_mask] * FT_TO_M).tolist()
                    )

    output_rows = []
    for cell_id, item in records.items():
        row = cell_id // ncols
        col = cell_id % ncols

        # Cell-centre coordinates are reported in the source CRS (feet).
        cell_xmin = xmin + col * GRID_SIZE_FT
        cell_ymin = ymin + row * GRID_SIZE_FT

        building_h = np.asarray(item.pop("building_hag_m"), dtype=float)
        vegetation_h = np.asarray(item.pop("vegetation_hag_m"), dtype=float)

        record = {
            "grid_row": int(row),
            "grid_col": int(col),
            "cell_id": int(cell_id),
            "cell_center_x_epsg2992_ft": cell_xmin + GRID_SIZE_FT / 2,
            "cell_center_y_epsg2992_ft": cell_ymin + GRID_SIZE_FT / 2,
            "grid_cell_size_m": GRID_SIZE_M,
            **item,
            "building_hag_median_m": (
                float(np.median(building_h)) if building_h.size else None
            ),
            "building_hag_p95_m": (
                float(np.percentile(building_h, 95)) if building_h.size else None
            ),
            "building_hag_max_m": (
                float(np.max(building_h)) if building_h.size else None
            ),
            "vegetation_hag_median_m": (
                float(np.median(vegetation_h)) if vegetation_h.size else None
            ),
            "vegetation_occupied": bool(item["vegetation_points"] > 0),
        }
        output_rows.append(record)

    df = pd.DataFrame(output_rows).sort_values(["grid_row", "grid_col"])
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_CSV, index=False)

    summary = {
        "study_area": "Autzen LiDAR dataset extent",
        "source_file": str(HAG_FILE),
        "horizontal_crs": "EPSG:2992",
        "horizontal_coordinate_units": "US survey feet",
        "grid_size_m": GRID_SIZE_M,
        "grid_size_ft": GRID_SIZE_FT,
        "grid_anchor": "Minimum X and Y of the input LiDAR extent",
        "input_finite_xy_point_count": total_points,
        "grid_rows": nrows,
        "grid_columns": ncols,
        "occupied_grid_cells": len(df),
        "cells_with_building_points": int((df["building_points"] > 0).sum()),
        "cells_with_vegetation_points": int(df["vegetation_occupied"].sum()),
        "building_height_unit": "metres",
        "vegetation_metric": (
            "Cell occupancy by Class 5 points, not validated canopy-cover area."
        ),
        "limitations": [
            "Grid origin is anchored to the input point-cloud extent.",
            "Only occupied cells are written to the CSV.",
            "Building HAG statistics depend on the quality of the existing HAG field.",
            "Negative HAG values are retained in the summary statistics.",
            "Class 5 occupancy is a vegetation-presence proxy, not canopy-cover percentage.",
            "No building footprints or independent canopy boundaries are used.",
        ],
        "csv_output": str(OUTPUT_CSV),
    }
    OUTPUT_JSON.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(json.dumps(summary, indent=2))
    print(f"\nCSV saved: {OUTPUT_CSV}")
    print(f"Summary saved: {OUTPUT_JSON}")


if __name__ == "__main__":
    main()
