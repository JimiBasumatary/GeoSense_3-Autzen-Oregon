
"""Estimate building-point heights using nearby LiDAR ground points.

Results are exploratory point-height estimates, not validated
individual building heights.
"""

from pathlib import Path
import json

import laspy
import numpy as np
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[2]
LAS_PATH = ROOT / "data/lidar/raw/autzen-classified.copc.laz"
OUT_PATH = ROOT / "outputs/reports/building_height_local_estimate.json"

# Building-rich bounding box in EPSG:2992, in feet.
BOUNDS = (636477.79, 851882.15, 636777.79, 852182.15)

# Require at least this many ground points within the search radius.
SEARCH_RADIUS_FT = 15.0
MIN_NEIGHBORS = 3
K_NEIGHBORS = 8
CHUNK_SIZE = 500_000


def load_points():
    xmin, ymin, xmax, ymax = BOUNDS
    ground_chunks = []
    building_chunks = []

    with laspy.open(LAS_PATH) as reader:
        for chunk in reader.chunk_iterator(CHUNK_SIZE):
            x = np.asarray(chunk.x)
            y = np.asarray(chunk.y)
            z = np.asarray(chunk.z)
            cls = np.asarray(chunk.classification)

            inside = (
                (x >= xmin) & (x <= xmax)
                & (y >= ymin) & (y <= ymax)
            )

            ground_mask = inside & (cls == 2)
            building_mask = inside & (cls == 6)

            if ground_mask.any():
                ground_chunks.append(
                    np.column_stack(
                        (x[ground_mask], y[ground_mask], z[ground_mask])
                    )
                )

            if building_mask.any():
                building_chunks.append(
                    np.column_stack(
                        (x[building_mask], y[building_mask], z[building_mask])
                    )
                )

    if not ground_chunks or not building_chunks:
        raise RuntimeError("Ground or building points were not found.")

    return np.vstack(ground_chunks), np.vstack(building_chunks)


def main():
    ground, buildings = load_points()

    # Build a spatial index from ground-point XY coordinates.
    tree = cKDTree(ground[:, :2])

    # Find up to K nearby ground points for each building point.
    distances, indices = tree.query(
        buildings[:, :2],
        k=K_NEIGHBORS,
        distance_upper_bound=SEARCH_RADIUS_FT,
        workers=-1,
    )

    if K_NEIGHBORS == 1:
        distances = distances[:, None]
        indices = indices[:, None]

    valid_neighbor = (
        np.isfinite(distances)
        & (indices < len(ground))
    )
    neighbor_count = valid_neighbor.sum(axis=1)
    supported = neighbor_count >= MIN_NEIGHBORS

    # Estimate local ground using the median of available neighbors.
    local_ground = np.full(len(buildings), np.nan)

    for i in np.flatnonzero(supported):
        local_ground[i] = np.median(
            ground[indices[i, valid_neighbor[i]], 2]
        )

    raw_height = buildings[:, 2] - local_ground

    # Negative values are retained for diagnosis, not silently clipped.
    supported_heights = raw_height[supported]
    nonnegative = supported_heights >= 0

    result = {
        "study_area": "Autzen building-rich bounding box",
        "coordinate_system": "EPSG:2992; horizontal and vertical units are feet",
        "bounds": {
            "xmin": BOUNDS[0],
            "ymin": BOUNDS[1],
            "xmax": BOUNDS[2],
            "ymax": BOUNDS[3],
        },
        "method": (
            "Building-point elevation minus median elevation of nearby "
            "Class 2 ground points in XY"
        ),
        "parameters": {
            "search_radius_ft": SEARCH_RADIUS_FT,
            "minimum_ground_neighbors": MIN_NEIGHBORS,
            "maximum_ground_neighbors": K_NEIGHBORS,
        },
        "counts": {
            "ground_points": int(len(ground)),
            "building_points": int(len(buildings)),
            "building_points_with_sufficient_neighbors": int(supported.sum()),
            "building_points_without_sufficient_neighbors": int((~supported).sum()),
            "supported_points_with_negative_height": int((supported_heights < 0).sum()),
            "supported_points_with_nonnegative_height": int(nonnegative.sum()),
        },
        "height_estimates_ft": None,
        "height_estimates_m": None,
        "limitations": [
            "These are point-wise exploratory estimates, not individual building heights.",
            "Nearby ground points may belong to another surface or elevation level.",
            "The fixed search radius may be unsuitable for some buildings.",
            "Building classification, ground classification, and spatial alignment require validation.",
            "Do not interpret these estimates as validated building-height measurements.",
        ],
    }

    if supported_heights.size:
        # Report all supported estimates and nonnegative estimates separately.
        result["height_estimates_ft"] = {
            "minimum": float(np.min(supported_heights)),
            "median": float(np.median(supported_heights)),
            "mean": float(np.mean(supported_heights)),
            "95th_percentile": float(np.percentile(supported_heights, 95)),
            "maximum": float(np.max(supported_heights)),
        }

        result["height_estimates_m"] = {
            key: value * 0.3048
            for key, value in result["height_estimates_ft"].items()
        }

        if nonnegative.any():
            result["nonnegative_height_estimates_ft"] = {
                "median": float(np.median(supported_heights[nonnegative])),
                "mean": float(np.mean(supported_heights[nonnegative])),
                "95th_percentile": float(
                    np.percentile(supported_heights[nonnegative], 95)
                ),
            }

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(
        json.dumps(result, indent=2),
        encoding="utf-8",
    )

    print(json.dumps(result, indent=2))
    print(f"\nSaved report: {OUT_PATH}")


if __name__ == "__main__":
    main()
