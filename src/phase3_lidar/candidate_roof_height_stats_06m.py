from pathlib import Path
import json
import csv
import numpy as np
import rasterio
import laspy
from PIL import Image
from pyproj import CRS

ROOT = Path(r"D:\GeoSense_Phase3")
IMAGE_PATH = ROOT / "data" / "aerial" / "autzen_naip_06m_rgb.tif"
LIDAR_PATH = ROOT / "data" / "lidar" / "processed" / "autzen_classified_hag.laz"
MASK_DIR = ROOT / "outputs" / "sam_georeferenced_06m" / "individual_masks"
OUTPUT_DIR = ROOT / "outputs" / "sam_georeferenced_06m"

MASK_IDS = [53, 22, 21, 18, 43]
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Load the masks only once.
masks = {}
for mask_id in MASK_IDS:
    path = MASK_DIR / f"mask_{mask_id:04d}.png"
    if not path.exists():
        raise FileNotFoundError(path)
    with Image.open(path) as im:
        masks[mask_id] = np.asarray(im.convert("L")) > 0

stats = {
    mid: {"count": 0, "negative": 0, "values": []}
    for mid in MASK_IDS
}

with rasterio.open(IMAGE_PATH) as image:
    transform = image.transform
    width, height = image.width, image.height
    image_crs = image.crs

with laspy.open(LIDAR_PATH) as reader:
    lidar_crs = reader.header.parse_crs()
    print("Image CRS:", image_crs)
    print("LiDAR CRS:", lidar_crs)

    if image_crs is None or lidar_crs is None:
        raise ValueError("Image or LiDAR CRS is missing.")

    image_horizontal_crs = CRS.from_user_input(image_crs)
    lidar_horizontal_crs = CRS.from_user_input(lidar_crs).to_2d()

    if not image_horizontal_crs.equals(lidar_horizontal_crs):
        raise ValueError(
            "Horizontal CRS mismatch; stop and reproject before matching."
        )

    for chunk_number, chunk in enumerate(
        reader.chunk_iterator(500_000), start=1
    ):
        x = np.asarray(chunk.x)
        y = np.asarray(chunk.y)
        z_hag = np.asarray(chunk.HeightAboveGround)
        classification = np.asarray(chunk.classification)

        # Map LiDAR coordinates to image pixels.
        cols = np.floor((x - transform.c) / transform.a).astype(np.int64)
        rows_px = np.floor((y - transform.f) / transform.e).astype(np.int64)

        inside = (
            (cols >= 0) & (cols < width) &
            (rows_px >= 0) & (rows_px < height)
        )
        building = inside & (classification == 6)

        if not np.any(building):
            continue

        rr = rows_px[building]
        cc = cols[building]
        hh = z_hag[building]

        for mask_id in MASK_IDS:
            selected = masks[mask_id][rr, cc]
            if np.any(selected):
                values = hh[selected]
                stats[mask_id]["count"] += int(values.size)
                stats[mask_id]["negative"] += int(np.sum(values < 0))
                stats[mask_id]["values"].append(values.copy())

        if chunk_number % 5 == 0:
            print(f"Processed {chunk_number * 500_000:,} points...")

summary = []

for mask_id in MASK_IDS:
    item = stats[mask_id]
    if item["count"] == 0:
        summary.append({
            "mask_id": mask_id,
            "building_points": 0,
            "registration_verified": False,
        })
        continue

    values = np.concatenate(item["values"])
    summary.append({
        "mask_id": mask_id,
        "building_points": item["count"],
        "median_hag_ft": float(np.median(values)),
        "p90_hag_ft": float(np.percentile(values, 90)),
        "median_hag_m": float(np.median(values) * 0.3048),
        "p90_hag_m": float(np.percentile(values, 90) * 0.3048),
        "negative_hag_points": item["negative"],
        "registration_verified": False,
    })

csv_path = OUTPUT_DIR / "candidate_roof_height_stats_06m.csv"
json_path = OUTPUT_DIR / "candidate_roof_height_stats_06m.json"

fields = sorted({key for row in summary for key in row})
with csv_path.open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=fields)
    writer.writeheader()
    writer.writerows(summary)

json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

print("\nCANDIDATE HEIGHT STATISTICS")
for row in summary:
    print(row)

print("\nSaved:", csv_path)
print("Saved:", json_path)
print(
    "WARNING: exploratory results only. Matching CRS does not prove "
    "image-to-LiDAR registration. Heights are not validated roof heights."
)