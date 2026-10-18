
from pathlib import Path
import csv
import json
import statistics
from datetime import datetime, timezone

import laspy
import rasterio


ROOT = Path(__file__).resolve().parent.parent

LIDAR_FILE = ROOT / "data/lidar/raw/autzen-classified.copc.laz"
PROCESSED = ROOT / "data/lidar/processed"
SEGMENTATION_CSV = (
    ROOT / "outputs/street_imagery/segmentation_summary_fixed.csv"
)
OVERLAYS = ROOT / "outputs/street_imagery/overlays_cityscapes"
REPORT_DIR = ROOT / "outputs/reports"


def raster_summary(path):
    """Read basic metadata and valid-cell statistics from a raster."""
    if not path.exists():
        return {"status": "missing", "file": str(path.relative_to(ROOT))}

    with rasterio.open(path) as src:
        band = src.read(1, masked=True)
        valid = band.compressed()

        result = {
            "file": str(path.relative_to(ROOT)),
            "width_pixels": src.width,
            "height_pixels": src.height,
            "band_count": src.count,
            "crs": str(src.crs),
            "pixel_size": list(src.res),
            "bounds": {
                "left": src.bounds.left,
                "bottom": src.bounds.bottom,
                "right": src.bounds.right,
                "top": src.bounds.top,
            },
            "valid_cells": int(valid.size),
            "nodata": src.nodata,
        }

        if valid.size:
            result.update({
                "minimum": float(valid.min()),
                "maximum": float(valid.max()),
                "mean": float(valid.mean()),
                "standard_deviation": float(valid.std()),
                "valid_cell_percent": round(
                    100 * valid.size / (src.width * src.height), 2
                ),
            })

        return result


def lidar_summary(path):
    """Read point-cloud header metadata without loading all points."""
    if not path.exists():
        return {"status": "missing", "file": str(path.relative_to(ROOT))}

    with laspy.open(path) as reader:
        header = reader.header
        crs = header.parse_crs()

        return {
            "file": str(path.relative_to(ROOT)),
            "point_count": int(header.point_count),
            "point_format": str(header.point_format),
            "version": str(header.version),
            "crs": crs.to_string() if crs else None,
            "bounds": {
                "x_min": float(header.mins[0]),
                "y_min": float(header.mins[1]),
                "z_min": float(header.mins[2]),
                "x_max": float(header.maxs[0]),
                "y_max": float(header.maxs[1]),
                "z_max": float(header.maxs[2]),
            },
            "scale": [float(v) for v in header.scales],
            "offset": [float(v) for v in header.offsets],
        }


def segmentation_summary(path):
    """Summarize existing per-image segmentation predictions."""
    if not path.exists():
        return {"status": "missing", "file": str(path.relative_to(ROOT))}

    with path.open("r", newline="", encoding="utf-8-sig") as file:
        rows = list(csv.DictReader(file))

    if not rows:
        return {
            "file": str(path.relative_to(ROOT)),
            "image_count": 0,
            "mean_class_percentages": {},
        }

    excluded = {"image_id"}
    class_columns = [
        column for column in rows[0]
        if column not in excluded
    ]

    means = {}
    for column in class_columns:
        values = []
        for row in rows:
            try:
                value = float(row[column])
                if value == value:  # Exclude NaN values
                    values.append(value)
            except (ValueError, TypeError):
                continue

        if values:
            means[column] = round(statistics.mean(values), 4)

    return {
        "file": str(path.relative_to(ROOT)),
        "image_count": len(rows),
        "image_ids": [row.get("image_id") for row in rows],
        "mean_class_percentages": means,
        "interpretation": (
            "Model predictions, not manually verified ground truth."
        ),
    }


def main():
    required_files = [
        LIDAR_FILE,
        PROCESSED / "autzen_ground.laz",
        PROCESSED / "autzen_dem.tif",
        PROCESSED / "autzen_hillshade.tif",
        PROCESSED / "autzen_slope.tif",
        PROCESSED / "autzen_aspect.tif",
        SEGMENTATION_CSV,
    ]

    missing = [
        str(path.relative_to(ROOT))
        for path in required_files if not path.exists()
    ]

    if missing:
        raise FileNotFoundError(
            "Required input files are missing:\n- " + "\n- ".join(missing)
        )

    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    profile = {
        "project": "GeoSense Agent - Phase III",
        "study_area": "Autzen Stadium area, Eugene, Oregon, USA",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "data_notes": [
            "Terrain products were generated from classified LiDAR ground points.",
            "Street-image class percentages are model predictions.",
            "LiDAR and street imagery have not been spatially matched in this report.",
            "Terrain elevations and coordinates use the source dataset's units and CRS.",
        ],
        "lidar": {
            "original_point_cloud": lidar_summary(LIDAR_FILE),
            "ground_point_cloud": lidar_summary(
                PROCESSED / "autzen_ground.laz"
            ),
        },
        "terrain": {
            "dem": raster_summary(PROCESSED / "autzen_dem.tif"),
            "hillshade": raster_summary(
                PROCESSED / "autzen_hillshade.tif"
            ),
            "slope": raster_summary(PROCESSED / "autzen_slope.tif"),
            "aspect": raster_summary(PROCESSED / "autzen_aspect.tif"),
        },
        "street_imagery": {
            "segmentation": segmentation_summary(SEGMENTATION_CSV),
            "corrected_overlay_count": len(
                list(OVERLAYS.glob("*_overlay.png"))
            ) if OVERLAYS.exists() else 0,
        },
    }

    json_path = REPORT_DIR / "unified_site_profile.json"
    with json_path.open("w", encoding="utf-8") as file:
        json.dump(profile, file, indent=2, ensure_ascii=False)

    # A compact table of terrain raster measurements.
    csv_path = REPORT_DIR / "terrain_metrics.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "product", "width_pixels", "height_pixels",
                "pixel_size", "crs", "minimum", "maximum",
                "mean", "standard_deviation", "valid_cell_percent",
            ],
        )
        writer.writeheader()

        for name, details in profile["terrain"].items():
            writer.writerow({
                "product": name,
                "width_pixels": details.get("width_pixels"),
                "height_pixels": details.get("height_pixels"),
                "pixel_size": details.get("pixel_size"),
                "crs": details.get("crs"),
                "minimum": details.get("minimum"),
                "maximum": details.get("maximum"),
                "mean": details.get("mean"),
                "standard_deviation": details.get("standard_deviation"),
                "valid_cell_percent": details.get("valid_cell_percent"),
            })

    print("EXERCISE 5 - LIDAR ANALYZER")
    print("Status: SUCCESS")
    print(f"Original LiDAR points: {profile['lidar']['original_point_cloud']['point_count']:,}")
    print(f"Ground points: {profile['lidar']['ground_point_cloud']['point_count']:,}")
    print(f"Segmentation images: {profile['street_imagery']['segmentation']['image_count']}")
    print(f"Corrected overlays: {profile['street_imagery']['corrected_overlay_count']}")
    print(f"JSON profile: {json_path}")
    print(f"Terrain CSV: {csv_path}")


if __name__ == "__main__":
    main()
