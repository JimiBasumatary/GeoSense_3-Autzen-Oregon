
from pathlib import Path
import json

import laspy
import numpy as np
import pandas as pd
import rasterio
from PIL import Image
from scipy.spatial import cKDTree

ROOT = Path(r"D:\GeoSense_Phase3")

LIDAR_PATH = ROOT / "data" / "lidar" / "raw" / "autzen-classified.copc.laz"
IMAGE_PATH = ROOT / "data" / "aerial" / "autzen_naip_06m_rgb.tif"
MASK_DIR = ROOT / "outputs" / "sam_georeferenced_06m" / "individual_masks"
OUT_DIR = ROOT / "outputs" / "sam_georeferenced_06m"

CANDIDATE_IDS = [53, 22, 21, 18, 43]
CHUNK_SIZE = 500_000
SEARCH_RADII_FT = [15.0, 30.0]

def main():
    for path in [LIDAR_PATH, IMAGE_PATH]:
        if not path.exists():
            raise FileNotFoundError(f"Missing input: {path}")

    print("Loading current candidate masks...")
    with rasterio.open(IMAGE_PATH) as src:
        transform = src.transform
        width, height = src.width, src.height
        bounds = src.bounds
        image_crs = src.crs

    masks = {}
    for mask_id in CANDIDATE_IDS:
        path = MASK_DIR / f"mask_{mask_id:04d}.png"
        if not path.exists():
            raise FileNotFoundError(f"Missing mask: {path}")
        mask = np.asarray(Image.open(path).convert("L")) > 0
        if mask.shape != (height, width):
            raise ValueError(f"Unexpected dimensions for Mask {mask_id}")
        masks[mask_id] = mask

    # Collect only points that fall inside at least one candidate mask.
    candidate_building_xy = {i: [] for i in CANDIDATE_IDS}
    candidate_hag = {i: [] for i in CANDIDATE_IDS}
    ground_xy_chunks = []

    print("Reading LiDAR in chunks...")
    with laspy.open(LIDAR_PATH) as reader:
        for chunk_no, chunk in enumerate(
            reader.chunk_iterator(CHUNK_SIZE), start=1
        ):
            x = np.asarray(chunk.x)
            y = np.asarray(chunk.y)
            cls = np.asarray(chunk.classification)

            ground = cls == 2
            if ground.any():
                ground_xy_chunks.append(
                    np.column_stack((x[ground], y[ground]))
                )

            building = cls == 6
            if not building.any():
                continue

            bx = x[building]
            by = y[building]

            cols = np.floor((bx - transform.c) / transform.a).astype(int)
            rows = np.floor((by - transform.f) / transform.e).astype(int)

            inside = (
                (cols >= 0) & (cols < width)
                & (rows >= 0) & (rows < height)
                & (bx >= bounds.left) & (bx < bounds.right)
                & (by >= bounds.bottom) & (by < bounds.top)
            )
            if not inside.any():
                continue

            bx = bx[inside]
            by = by[inside]
            cols = cols[inside]
            rows = rows[inside]

            # HAG is available in the processed file, not the raw file.
            # We will report ground support here independently of HAG.
            for mask_id, mask in masks.items():
                matched = mask[rows, cols]
                if matched.any():
                    candidate_building_xy[mask_id].append(
                        np.column_stack((bx[matched], by[matched]))
                    )

            if chunk_no % 5 == 0:
                print(f"Processed {chunk_no} LiDAR chunks...")

    if not ground_xy_chunks:
        raise RuntimeError("No Class 2 ground points found.")

    print("Building ground-point spatial index...")
    ground_xy = np.vstack(ground_xy_chunks)
    tree = cKDTree(ground_xy)

    records = []
    for mask_id in CANDIDATE_IDS:
        parts = candidate_building_xy[mask_id]
        building_xy = (
            np.vstack(parts) if parts else np.empty((0, 2), dtype=float)
        )

        record = {
            "mask_id": mask_id,
            "building_points_in_mask": int(len(building_xy)),
            "ground_points_in_dataset": int(len(ground_xy)),
            "registration_verified": False,
        }

        if len(building_xy):
            distances, _ = tree.query(building_xy, k=1, workers=-1)
            record["nearest_ground_distance_median_ft"] = float(
                np.median(distances)
            )
            record["nearest_ground_distance_p90_ft"] = float(
                np.percentile(distances, 90)
            )

            for radius in SEARCH_RADII_FT:
                record[f"points_with_ground_within_{int(radius)}ft"] = int(
                    np.sum(distances <= radius)
                )
                record[f"fraction_with_ground_within_{int(radius)}ft"] = float(
                    np.mean(distances <= radius)
                )
        else:
            record["note"] = "No building-class points matched this mask."

        records.append(record)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = OUT_DIR / "candidate_ground_support_06m.csv"
    json_path = OUT_DIR / "candidate_ground_support_06m.json"

    pd.DataFrame(records).to_csv(csv_path, index=False)
    json_path.write_text(json.dumps(records, indent=2), encoding="utf-8")

    print("\nCANDIDATE GROUND-SUPPORT SUMMARY")
    print(pd.DataFrame(records).to_string(index=False))
    print("\nCSV:", csv_path)
    print("JSON:", json_path)
    print(
        "\nCaution: proximity to ground-class points does not prove "
        "correct image registration or validated building height."
    )

if __name__ == "__main__":
    main()
