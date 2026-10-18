
"""Compare local-ground height estimates across search radii."""

from pathlib import Path
import json

import laspy
import numpy as np
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[2]
LAS_PATH = ROOT / "data/lidar/raw/autzen-classified.copc.laz"
OUT_PATH = ROOT / "outputs/reports/height_radius_comparison.json"

BOUNDS = (636477.79, 851882.15, 636777.79, 852182.15)
RADII_FT = (15.0, 30.0, 45.0)
MIN_NEIGHBORS = 3
K_NEIGHBORS = 8


def load_points():
    xmin, ymin, xmax, ymax = BOUNDS
    ground_parts, building_parts = [], []

    with laspy.open(LAS_PATH) as reader:
        for chunk in reader.chunk_iterator(500_000):
            x = np.asarray(chunk.x)
            y = np.asarray(chunk.y)
            z = np.asarray(chunk.z)
            cls = np.asarray(chunk.classification)

            inside = (
                (x >= xmin) & (x <= xmax)
                & (y >= ymin) & (y <= ymax)
            )

            gm = inside & (cls == 2)
            bm = inside & (cls == 6)

            if gm.any():
                ground_parts.append(
                    np.column_stack((x[gm], y[gm], z[gm]))
                )
            if bm.any():
                building_parts.append(
                    np.column_stack((x[bm], y[bm], z[bm]))
                )

    if not ground_parts or not building_parts:
        raise RuntimeError("Ground or building points were not found.")

    return np.vstack(ground_parts), np.vstack(building_parts)


def main():
    ground, buildings = load_points()
    tree = cKDTree(ground[:, :2])
    results = []

    for radius in RADII_FT:
        distances, indices = tree.query(
            buildings[:, :2],
            k=K_NEIGHBORS,
            distance_upper_bound=radius,
            workers=-1,
        )

        valid = np.isfinite(distances) & (indices < len(ground))
        neighbor_count = valid.sum(axis=1)
        supported = neighbor_count >= MIN_NEIGHBORS

        heights = []
        for i in np.flatnonzero(supported):
            local_ground_z = np.median(
                ground[indices[i, valid[i]], 2]
            )
            heights.append(buildings[i, 2] - local_ground_z)

        heights = np.asarray(heights, dtype=float)
        nonnegative = heights[heights >= 0]

        row = {
            "radius_ft": radius,
            "building_points": int(len(buildings)),
            "supported_points": int(supported.sum()),
            "unsupported_points": int((~supported).sum()),
            "negative_estimates": int((heights < 0).sum()),
            "nonnegative_estimates": int(len(nonnegative)),
            "height_summary_ft": None,
        }

        if len(heights):
            row["height_summary_ft"] = {
                "median": float(np.median(heights)),
                "mean": float(np.mean(heights)),
                "p95": float(np.percentile(heights, 95)),
            }

        if len(nonnegative):
            row["nonnegative_height_summary_ft"] = {
                "median": float(np.median(nonnegative)),
                "mean": float(np.mean(nonnegative)),
                "p95": float(np.percentile(nonnegative, 95)),
            }

        results.append(row)

    report = {
        "study_area": "Autzen building-rich bounding box",
        "method": "Building Z minus median of up to 8 nearby Class 2 ground points",
        "minimum_neighbors": MIN_NEIGHBORS,
        "results": results,
        "limitations": [
            "Point-height estimates are not validated individual building heights.",
            "A larger radius may use ground points from unrelated terrain or structures.",
            "Compare coverage and distributions; do not select a radius based only on maximum coverage.",
        ],
    }

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(json.dumps(report, indent=2))
    print(f"\nSaved: {OUT_PATH}")


if __name__ == "__main__":
    main()
