
import json
from pathlib import Path

import numpy as np
import rasterio

ROOT = Path(__file__).resolve().parents[2]

SLOPE_PATH = ROOT / "data" / "lidar" / "processed" / "autzen_slope.tif"
OUTPUT_PATH = ROOT / "outputs" / "reports" / "slope_risk_indicator.json"


def calculate_slope_indicator(slope_path=SLOPE_PATH):
    """Summarize slope classes as an exploratory terrain-screening indicator."""

    slope_path = Path(slope_path)

    if not slope_path.exists():
        raise FileNotFoundError(f"Slope raster not found: {slope_path}")

    with rasterio.open(slope_path) as src:
        slope = src.read(1, masked=True).astype("float64")
        values = slope.compressed()

    if values.size == 0:
        raise ValueError("The slope raster contains no valid cells.")

    if np.any(~np.isfinite(values)):
        values = values[np.isfinite(values)]

    if values.size == 0:
        raise ValueError("The slope raster contains no finite values.")

    # Descriptive slope classes, not scientifically validated risk thresholds.
    classes = {
        "low_slope_0_to_5_degrees": (values >= 0) & (values < 5),
        "moderate_slope_5_to_15_degrees": (values >= 5) & (values < 15),
        "steep_slope_15_to_30_degrees": (values >= 15) & (values < 30),
        "very_steep_slope_30_degrees_or_more": values >= 30,
    }

    class_summary = {}

    for name, mask in classes.items():
        count = int(np.count_nonzero(mask))
        class_summary[name] = {
            "valid_cell_count": count,
            "percentage_of_valid_cells": round(
                count / values.size * 100, 3
            ),
        }

    result = {
        "study_area": "Autzen Stadium area, Eugene, Oregon, USA",
        "input_raster": str(slope_path.relative_to(ROOT)),
        "slope_unit": "degrees",
        "valid_cell_count": int(values.size),
        "slope_min_degrees": round(float(values.min()), 4),
        "slope_mean_degrees": round(float(values.mean()), 4),
        "slope_max_degrees": round(float(values.max()), 4),
        "slope_classes": class_summary,
        "interpretation": (
            "Exploratory terrain-screening indicator based on slope classes. "
            "The class thresholds are descriptive and are not validated "
            "hazard, landslide, accessibility, or safety thresholds."
        ),
        "limitations": [
            "Results depend on DEM resolution and slope-processing settings.",
            "Slope alone cannot establish landslide or other hazard risk.",
            "No field validation or hazard inventory was used.",
            "Percentages describe valid raster cells, not the entire study area."
        ],
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(result, indent=2), encoding="utf-8")

    return result


if __name__ == "__main__":
    result = calculate_slope_indicator()

    print("SLOPE SCREENING SUMMARY")
    print("-----------------------")
    print(f"Valid cells: {result['valid_cell_count']:,}")
    print(f"Minimum slope: {result['slope_min_degrees']} degrees")
    print(f"Mean slope: {result['slope_mean_degrees']} degrees")
    print(f"Maximum slope: {result['slope_max_degrees']} degrees")

    for name, stats in result["slope_classes"].items():
        print(
            f"{name}: {stats['valid_cell_count']:,} cells "
            f"({stats['percentage_of_valid_cells']}%)"
        )

    print(f"\nReport saved to: {OUTPUT_PATH}")
