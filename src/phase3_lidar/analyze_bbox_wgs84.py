
"""Run LiDAR bounding-box analysis from WGS84 coordinates."""

from src.phase3_lidar.coordinate_utils import wgs84_bbox_to_lidar
from src.phase3_lidar.analyze_bbox import analyze_bbox


def analyze_bbox_wgs84(
    xmin_lon,
    ymin_lat,
    xmax_lon,
    ymax_lat,
):
    """Convert a WGS84 bounding box to EPSG:2992 and analyse it."""

    projected_bounds = wgs84_bbox_to_lidar(
        xmin_lon,
        ymin_lat,
        xmax_lon,
        ymax_lat,
    )

    result = analyze_bbox(projected_bounds)

    result["input_coordinate_system"] = "EPSG:4326 (WGS84)"
    result["analysis_coordinate_system"] = "EPSG:2992"
    result["input_bounds"] = {
        "xmin_lon": float(xmin_lon),
        "ymin_lat": float(ymin_lat),
        "xmax_lon": float(xmax_lon),
        "ymax_lat": float(ymax_lat),
    }

    return result
