
"""GeoSense Phase III: LiDAR bounding-box analyzer."""

from pathlib import Path
import json

import laspy
import numpy as np
import rasterio
from rasterio.windows import Window, from_bounds
from rasterio.errors import WindowError


ROOT = Path(__file__).resolve().parents[2]

LIDAR_FILE = ROOT / "data/lidar/raw/autzen-classified.copc.laz"
GROUND_FILE = ROOT / "data/lidar/processed/autzen_ground.laz"
DEM_FILE = ROOT / "data/lidar/processed/autzen_dem.tif"
SLOPE_FILE = ROOT / "data/lidar/processed/autzen_slope.tif"


def validate_bounds(bounds):
    """Validate projected bounds and return numeric coordinates."""
    try:
        if len(bounds) != 4:
            raise ValueError
        xmin, ymin, xmax, ymax = map(float, bounds)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "Bounds must contain four numeric values: "
            "(xmin, ymin, xmax, ymax)."
        ) from exc

    if not np.all(np.isfinite([xmin, ymin, xmax, ymax])):
        raise ValueError("Bounds must be finite numbers.")

    if not (xmin < xmax and ymin < ymax):
        raise ValueError(
            "Bounds must satisfy xmin < xmax and ymin < ymax."
        )

    return xmin, ymin, xmax, ymax


def get_coverage_status(
    xmin, ymin, xmax, ymax,
    left, bottom, right, top,
):
    """Compare a bounding box with rectangular dataset bounds."""
    intersects = (
        xmax > left and xmin < right
        and ymax > bottom and ymin < top
    )

    fully_covered = (
        xmin >= left and ymin >= bottom
        and xmax <= right and ymax <= top
    )

    if fully_covered:
        status = "full"
    elif intersects:
        status = "partial"
    else:
        status = "none"

    requested_area = (xmax - xmin) * (ymax - ymin)

    if intersects:
        overlap_width = max(
            0.0, min(xmax, right) - max(xmin, left)
        )
        overlap_height = max(
            0.0, min(ymax, top) - max(ymin, bottom)
        )
        overlap_area = overlap_width * overlap_height
    else:
        overlap_area = 0.0

    return {
        "intersects": bool(intersects),
        "fully_covered_bbox": bool(fully_covered),
        "status": status,
        "requested_area_sq_ft": float(requested_area),
        "overlap_area_sq_ft": float(overlap_area),
        "overlap_fraction": float(
            min(1.0, overlap_area / requested_area)
        ),
    }


def analyze_bbox(bounds):
    """Analyze classified LiDAR points and DEM/slope raster cells.

    Bounds must be in EPSG:2992 horizontal coordinates, in feet.
    """
    xmin, ymin, xmax, ymax = validate_bounds(bounds)

    result = {
        "bounds": {
            "xmin": xmin,
            "ymin": ymin,
            "xmax": xmax,
            "ymax": ymax,
        },
        "coordinate_system": (
            "EPSG:2992 horizontal; elevations in US survey feet"
        ),
        "area_sq_ft": float((xmax - xmin) * (ymax - ymin)),
        "coverage": {},
        "metrics": {},
        "limitations": [
            "Coverage describes rectangular dataset bounds, not actual "
            "point density or valid data at every location."
        ],
    }

    if not LIDAR_FILE.exists():
        raise FileNotFoundError(f"LiDAR file not found: {LIDAR_FILE}")

    with laspy.open(LIDAR_FILE) as reader:
        header = reader.header

        cov = get_coverage_status(
            xmin, ymin, xmax, ymax,
            float(header.mins[0]), float(header.mins[1]),
            float(header.maxs[0]), float(header.maxs[1]),
        )

        result["coverage"]["lidar_intersects"] = cov["intersects"]
        result["coverage"]["lidar_fully_covers_bbox"] = (
            cov["fully_covered_bbox"]
        )
        result["coverage"]["lidar_coverage_status"] = cov["status"]
        result["coverage"]["lidar_overlap_area_sq_ft"] = (
            cov["overlap_area_sq_ft"]
        )
        result["coverage"]["lidar_overlap_fraction"] = (
            cov["overlap_fraction"]
        )
        result["coverage"]["lidar_covered"] = cov["fully_covered_bbox"]

        counts = {}
        z_ground = []
        z_building = []
        z_vegetation = []

        if cov["intersects"]:
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
                z = np.asarray(points.z)[selected]

                unique_classes, class_counts = np.unique(
                    classes, return_counts=True
                )
                for cls, count in zip(unique_classes, class_counts):
                    key = str(int(cls))
                    counts[key] = counts.get(key, 0) + int(count)

                z_ground.extend(z[classes == 2].tolist())
                z_building.extend(z[classes == 6].tolist())
                z_vegetation.extend(z[classes == 5].tolist())

        result["metrics"]["classification_point_counts"] = counts
        result["metrics"]["total_selected_point_count"] = sum(counts.values())
        result["metrics"]["ground_point_count"] = len(z_ground)
        result["metrics"]["building_point_count"] = len(z_building)
        result["metrics"]["high_vegetation_point_count"] = len(
            z_vegetation
        )

        if z_ground:
            result["metrics"]["ground_elevation_ft"] = {
                "minimum": float(min(z_ground)),
                "maximum": float(max(z_ground)),
                "mean": float(np.mean(z_ground)),
            }
        else:
            result["metrics"]["ground_elevation_ft"] = None

        if z_building:
            result["metrics"]["building_point_elevation_range_ft"] = {
                "minimum": float(min(z_building)),
                "maximum": float(max(z_building)),
            }
        else:
            result["metrics"]["building_point_elevation_range_ft"] = None

        result["limitations"].append(
            "Building-point elevation range is not building height."
        )
        result["limitations"].append(
            "Canopy cover and impervious-surface area are not calculated "
            "from point counts alone."
        )

        if not cov["intersects"]:
            result["limitations"].append(
                "The requested box does not intersect the LiDAR bounds."
            )
        elif not counts:
            result["limitations"].append(
                "The box intersects the LiDAR bounds but contains no points."
            )

    # Summarize DEM and slope cells.
    for name, path in (("dem", DEM_FILE), ("slope", SLOPE_FILE)):
        if not path.exists():
            result["coverage"][f"{name}_available"] = False
            result["coverage"][f"{name}_intersects"] = False
            result["coverage"][f"{name}_fully_covers_bbox"] = False
            result["coverage"][f"{name}_coverage_status"] = "none"
            result["coverage"][f"{name}_has_valid_cells"] = False
            result["metrics"][name] = None
            continue

        result["coverage"][f"{name}_available"] = True

        with rasterio.open(path) as raster:
            cov = get_coverage_status(
                xmin, ymin, xmax, ymax,
                raster.bounds.left, raster.bounds.bottom,
                raster.bounds.right, raster.bounds.top,
            )

            result["coverage"][f"{name}_intersects"] = cov["intersects"]
            result["coverage"][f"{name}_fully_covers_bbox"] = (
                cov["fully_covered_bbox"]
            )
            result["coverage"][f"{name}_coverage_status"] = cov["status"]
            result["coverage"][f"{name}_overlap_fraction"] = (
                cov["overlap_fraction"]
            )

            if not cov["intersects"]:
                result["metrics"][name] = None
                result["coverage"][f"{name}_has_valid_cells"] = False
                continue

            window = from_bounds(
                xmin, ymin, xmax, ymax, transform=raster.transform
            ).round_offsets().round_lengths()

            try:
                window = window.intersection(
                    Window(0, 0, raster.width, raster.height)
                )
            except WindowError:
                window = Window(0, 0, 0, 0)

            if window.width <= 0 or window.height <= 0:
                values = np.array([])
            else:
                data = raster.read(1, window=window, masked=True)
                values = data.compressed()
                values = values[np.isfinite(values)]

            if values.size == 0:
                result["metrics"][name] = None
                result["coverage"][f"{name}_has_valid_cells"] = False
                continue

            result["coverage"][f"{name}_has_valid_cells"] = True
            result["metrics"][name] = {
                "valid_cell_count": int(values.size),
                "minimum": float(values.min()),
                "maximum": float(values.max()),
                "mean": float(values.mean()),
                "standard_deviation": float(values.std()),
            }

    return result


if __name__ == "__main__":
    example_bounds = (637000, 850500, 637300, 850800)
    profile = analyze_bbox(example_bounds)

    output_dir = ROOT / "outputs/reports"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "bbox_analysis_example.json"
    output_file.write_text(
        json.dumps(profile, indent=2), encoding="utf-8"
    )

    print("Bounding-box analysis completed.")
    print("LiDAR coverage:", profile["coverage"]["lidar_coverage_status"])
    print(json.dumps(profile, indent=2))
    print(f"\nSaved to: {output_file}")
