
"""
Exercise 5: LiDAR analysis interface.

Input bounding box order:
(min_lat, min_lon, max_lat, max_lon), in WGS84 degrees.

Existing detailed analysis is delegated to analyze_bbox_wgs84.
Unsupported metrics remain None rather than being fabricated.
"""

from typing import Any


def analyze_bbox(
    bbox: tuple[float, float, float, float],
) -> dict[str, Any]:
    """
    Analyze a geographic bounding box.

    Parameters
    ----------
    bbox : tuple
        (min_lat, min_lon, max_lat, max_lon) in EPSG:4326.

    Returns
    -------
    dict
        Existing LiDAR analysis plus the four Exercise 5 fields.
    """
    if len(bbox) != 4:
        raise ValueError(
            "bbox must be (min_lat, min_lon, max_lat, max_lon)"
        )

    min_lat, min_lon, max_lat, max_lon = map(float, bbox)

    if not (
        -90 <= min_lat <= 90
        and -90 <= max_lat <= 90
        and -180 <= min_lon <= 180
        and -180 <= max_lon <= 180
    ):
        raise ValueError("Bounding-box coordinates are out of range")

    if min_lat >= max_lat or min_lon >= max_lon:
        raise ValueError("Bounding-box minimums must be below maximums")

    # The existing function expects longitude/latitude order.
    from src.phase3_lidar.analyze_bbox_wgs84 import (
        analyze_bbox_wgs84 as existing_analysis,
    )

    result = existing_analysis(
        min_lon, min_lat, max_lon, max_lat
    )

    # Do not present point counts or occupancy proxies as validated
    # surface percentages or building-height measurements.
    result["avg_building_height_m"] = None
    result["canopy_cover_pct"] = None
    result["impervious_surface_pct"] = None
    result["slope_risk_score"] = None

    result["exercise5_metric_status"] = {
        "avg_building_height_m": (
            "Not calculated for this specific bounding box by this interface"
        ),
        "canopy_cover_pct": (
            "Not calculated; occupancy is only an exploratory proxy"
        ),
        "impervious_surface_pct": (
            "Not calculated as an area-based percentage"
        ),
        "slope_risk_score": (
            "Not calculated as a validated single risk score"
        ),
    }

    result["metric_units"] = {
        "avg_building_height_m": "m",
        "canopy_cover_pct": "%",
        "impervious_surface_pct": "%",
        "slope_risk_score": "Not assigned",
    }

    return result
