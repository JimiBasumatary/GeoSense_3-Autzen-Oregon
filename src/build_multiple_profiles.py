
from pathlib import Path
import json
import sys
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parent.parent

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.phase3_lidar.analyze_bbox import analyze_bbox


def build_profile(location_name, bounds):
    """Build and save a profile for one bounding box."""

    result = analyze_bbox(bounds)

    return {
        "location_name": location_name,
        "project": "GeoSense Agent - Phase III",
        "study_area": "Autzen Stadium area, Eugene, Oregon, USA",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "coordinate_system": "EPSG:2992",
        "bounding_box": result,
        "data_sources": {
            "lidar": "Autzen classified LiDAR",
            "terrain": "Autzen DEM and slope raster",
            "street_imagery": (
                "Not spatially matched to this bounding box"
            ),
            "phase_i": "Not supplied for Autzen",
            "phase_ii": "Not supplied for Autzen",
        },
        "limitations": [
            "Building-point elevation range is not building height.",
            "Canopy cover and impervious-surface area are not calculated.",
            "Slope statistics are descriptive, not a validated risk score.",
            "Street imagery has not been spatially matched to this box.",
        ],
    }


def main():
    output_dir = ROOT / "outputs" / "reports" / "site_profiles"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Two separate test areas within the dataset's approximate extent.
    locations = [
        {
            "name": "autzen_area_1",
            "bounds": (637000, 850500, 637300, 850800),
        },
        {
            "name": "autzen_area_2",
            "bounds": (637500, 851000, 637800, 851300),
        },
    ]

    index = {
        "project": "GeoSense Agent - Phase III",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "profiles": [],
    }

    for location in locations:
        profile = build_profile(location["name"], location["bounds"])

        filename = f"{location['name']}.json"
        path = output_dir / filename

        with path.open("w", encoding="utf-8") as file:
            json.dump(profile, file, indent=2, ensure_ascii=False)

        index["profiles"].append({
            "location_name": location["name"],
            "file": filename,
            "lidar_covered": profile["bounding_box"]["coverage"][
                "lidar_covered"
            ],
            "ground_point_count": profile["bounding_box"]["metrics"][
                "ground_point_count"
            ],
        })

        print(f"Saved: {path}")

    index_path = output_dir / "profiles_index.json"
    with index_path.open("w", encoding="utf-8") as file:
        json.dump(index, file, indent=2, ensure_ascii=False)

    print(f"\nProfile index saved: {index_path}")
    print(json.dumps(index, indent=2))


if __name__ == "__main__":
    main()
