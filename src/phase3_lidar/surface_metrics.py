
"""Exploratory surface indicators from classified LiDAR points."""

from pathlib import Path
import json

import laspy
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
LIDAR_FILE = ROOT / "data/lidar/raw/autzen-classified.copc.laz"


def calculate_surface_metrics(bounds):
    """Summarize classified point counts in EPSG:2992 feet.

    These are point-density indicators, not area-cover percentages.
    """
    if len(bounds) != 4:
        raise ValueError("Bounds must contain xmin, ymin, xmax, ymax.")

    xmin, ymin, xmax, ymax = map(float, bounds)

    if not np.all(np.isfinite([xmin, ymin, xmax, ymax])):
        raise ValueError("Bounds must be finite numbers.")

    if xmin >= xmax or ymin >= ymax:
        raise ValueError("Bounds must satisfy xmin < xmax and ymin < ymax.")

    if not LIDAR_FILE.exists():
        raise FileNotFoundError(f"LiDAR file not found: {LIDAR_FILE}")

    area_sq_ft = (xmax - xmin) * (ymax - ymin)
    counts = {2: 0, 5: 0, 6: 0}
    total_points = 0

    with laspy.open(LIDAR_FILE) as reader:
        for points in reader.chunk_iterator(500_000):
            x = np.asarray(points.x)
            y = np.asarray(points.y)

            selected = (
                (x >= xmin) & (x < xmax)
                & (y >= ymin) & (y < ymax)
            )

            if not np.any(selected):
                continue

            classes = np.asarray(points.classification)[selected]
            total_points += int(selected.sum())

            for class_code in counts:
                counts[class_code] += int(np.count_nonzero(
                    classes == class_code
                ))

        header = reader.header
        lidar_bounds = (
            float(header.mins[0]),
            float(header.mins[1]),
            float(header.maxs[0]),
            float(header.maxs[1]),
        )

    left, bottom, right, top = lidar_bounds
    intersects = (
        xmax > left and xmin < right
        and ymax > bottom and ymin < top
    )
    fully_covered = (
        xmin >= left and ymin >= bottom
        and xmax <= right and ymax <= top
    )

    return {
        "bounds_epsg_2992_ft": {
            "xmin": xmin, "ymin": ymin,
            "xmax": xmax, "ymax": ymax,
        },
        "area_sq_ft": area_sq_ft,
        "lidar_bounds_intersect": intersects,
        "lidar_bounds_fully_cover_box": fully_covered,
        "total_selected_points": total_points,
        "classified_point_counts": {
            "ground_class_2": counts[2],
            "high_vegetation_class_5": counts[5],
            "building_class_6": counts[6],
        },
        "point_density_per_sq_ft": {
            "ground_class_2": counts[2] / area_sq_ft,
            "high_vegetation_class_5": counts[5] / area_sq_ft,
            "building_class_6": counts[6] / area_sq_ft,
        },
        "interpretation": {
            "vegetation": (
                "Class 5 point density only; not canopy-cover percentage."
            ),
            "buildings": (
                "Class 6 point density only; not building footprint area."
            ),
            "ground": "Class 2 point density only.",
        },
        "limitations": [
            "LiDAR rectangular bounds do not guarantee uniform point coverage.",
            "Point density is affected by acquisition and classification.",
            "Point counts do not directly measure surface area.",
            "Canopy-cover percentage and impervious-surface percentage "
            "are not calculated by this module.",
        ],
    }


if __name__ == "__main__":
    # Use the existing 300 ft x 300 ft example area.
    bounds = (637000, 850500, 637300, 850800)
    result = calculate_surface_metrics(bounds)

    output_dir = ROOT / "outputs/reports"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "surface_metrics_example.json"
    output_file.write_text(json.dumps(result, indent=2), encoding="utf-8")

    print(json.dumps(result, indent=2))
    print(f"\nSaved to: {output_file}")
