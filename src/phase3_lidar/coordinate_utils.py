
"""Coordinate conversion utilities for GeoSense Phase III."""

from pyproj import Transformer


def wgs84_bbox_to_lidar(
    xmin_lon,
    ymin_lat,
    xmax_lon,
    ymax_lat,
):
    """Convert a WGS84 bounding box to EPSG:2992.

    Input:
        Longitude and latitude in decimal degrees.

    Output:
        xmin, ymin, xmax, ymax in the LiDAR CRS (EPSG:2992).

    Note:
        This transforms the four corners. For large bounding boxes,
        edge densification may be needed for a more accurate envelope.
    """
    if xmin_lon >= xmax_lon or ymin_lat >= ymax_lat:
        raise ValueError(
            "Bounding-box minimums must be less than maximums."
        )

    if not (
        -180 <= xmin_lon <= 180
        and -180 <= xmax_lon <= 180
        and -90 <= ymin_lat <= 90
        and -90 <= ymax_lat <= 90
    ):
        raise ValueError("Coordinates are outside valid WGS84 ranges.")

    transformer = Transformer.from_crs(
        "EPSG:4326",
        "EPSG:2992",
        always_xy=True,
    )

    corners = [
        transformer.transform(xmin_lon, ymin_lat),
        transformer.transform(xmin_lon, ymax_lat),
        transformer.transform(xmax_lon, ymin_lat),
        transformer.transform(xmax_lon, ymax_lat),
    ]

    xs = [point[0] for point in corners]
    ys = [point[1] for point in corners]

    return min(xs), min(ys), max(xs), max(ys)
