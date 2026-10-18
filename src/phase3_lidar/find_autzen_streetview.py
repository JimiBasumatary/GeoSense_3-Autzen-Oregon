
import os
import csv
import json
from pathlib import Path

import requests
from dotenv import load_dotenv
from pyproj import Transformer

ROOT = Path(r"D:\GeoSense_Phase3")
ENV_FILE = ROOT / ".env"

# Keep new candidate images separate from the original 24 images.
OUTPUT_DIR = ROOT / "data" / "streetview" / "autzen_candidates"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

METADATA_PATH = OUTPUT_DIR / "image_metadata.csv"
SUMMARY_PATH = OUTPUT_DIR / "spatial_matching_summary.json"

# Original LiDAR bounds, EPSG:2992.
XMIN = 635577.79
YMIN = 848882.15
XMAX = 639003.73
YMAX = 853537.66

# Transform Mapillary coordinates from longitude/latitude to EPSG:2992.
transformer = Transformer.from_crs(
    "EPSG:4326", "EPSG:2992", always_xy=True
)

load_dotenv(ENV_FILE)
token = os.getenv("MAPILLARY_TOKEN", "").strip()

if not token:
    raise SystemExit(
        "ERROR: MAPILLARY_TOKEN not found in D:\\GeoSense_Phase3\\.env"
    )

# Transform LiDAR corners back to geographic coordinates to construct
# a search area for Mapillary. Add a small margin around the footprint.
inverse = Transformer.from_crs(
    "EPSG:2992", "EPSG:4326", always_xy=True
)

corners = [
    inverse.transform(XMIN - 100, YMIN - 100),
    inverse.transform(XMIN - 100, YMAX + 100),
    inverse.transform(XMAX + 100, YMIN - 100),
    inverse.transform(XMAX + 100, YMAX + 100),
]

longitudes = [point[0] for point in corners]
latitudes = [point[1] for point in corners]

west = min(longitudes)
south = min(latitudes)
east = max(longitudes)
north = max(latitudes)

bbox = f"{west},{south},{east},{north}"

print("Autzen LiDAR footprint search")
print("Search bounding box:", bbox)
print("LiDAR CRS: EPSG:2992")
print("Searching Mapillary...")

session = requests.Session()
session.headers.update({"Authorization": f"OAuth {token}"})

records = {}
url = "https://graph.mapillary.com/images"

try:
    response = session.get(
        url,
        params={
            "bbox": bbox,
            "fields": "id,thumb_1024_url,computed_geometry,captured_at",
            "limit": 100,
        },
        timeout=(10, 45),
    )
    response.raise_for_status()
    payload = response.json()

except requests.RequestException as error:
    raise SystemExit(
        f"Mapillary request failed: {type(error).__name__}: {error}"
    )

for item in payload.get("data", []):
    image_id = item.get("id")
    geometry = item.get("computed_geometry") or {}
    coords = geometry.get("coordinates")

    if not image_id or not coords or len(coords) < 2:
        continue

    longitude, latitude = coords[0], coords[1]
    x, y = transformer.transform(longitude, latitude)

    inside = XMIN <= x <= XMAX and YMIN <= y <= YMAX

    records[image_id] = {
        "image_id": image_id,
        "filename": f"{image_id}.jpg",
        "captured_at": item.get("captured_at"),
        "longitude": longitude,
        "latitude": latitude,
        "x_epsg2992": x,
        "y_epsg2992": y,
        "inside_lidar_bounds": inside,
        "download_status": "not_downloaded",
        "thumbnail_url_available": bool(item.get("thumb_1024_url")),
        "_thumbnail_url": item.get("thumb_1024_url"),
    }

print("Unique images returned:", len(records))

# Download only images whose coordinates fall inside the actual LiDAR bounds.
downloaded = 0
failed = 0
inside_count = 0

for record in records.values():
    if not record["inside_lidar_bounds"]:
        continue

    inside_count += 1
    image_id = record["image_id"]
    filepath = OUTPUT_DIR / record["filename"]
    image_url = record.pop("_thumbnail_url")

    try:
        if filepath.exists() and filepath.stat().st_size > 0:
            record["download_status"] = "already_exists"
            continue

        if not image_url:
            record["download_status"] = "no_thumbnail_url"
            failed += 1
            continue

        image_response = session.get(image_url, timeout=(10, 45))
        image_response.raise_for_status()

        if not image_response.headers.get("Content-Type", "").startswith("image/"):
            raise ValueError("Response was not an image.")

        filepath.write_bytes(image_response.content)
        record["download_status"] = "downloaded"
        downloaded += 1

    except (requests.RequestException, OSError, ValueError) as error:
        record["download_status"] = "failed"
        record["download_error"] = type(error).__name__
        failed += 1

# Save metadata, including coordinates and explicit coverage status.
fields = [
    "image_id",
    "filename",
    "captured_at",
    "longitude",
    "latitude",
    "x_epsg2992",
    "y_epsg2992",
    "inside_lidar_bounds",
    "download_status",
    "thumbnail_url_available",
]

with METADATA_PATH.open("w", newline="", encoding="utf-8") as file:
    writer = csv.DictWriter(file, fieldnames=fields)
    writer.writeheader()

    for record in records.values():
        writer.writerow({key: record.get(key) for key in fields})

summary = {
    "study_area": "Autzen Stadium area, Eugene, Oregon, USA",
    "lidar_crs": "EPSG:2992",
    "lidar_bounds": {
        "xmin": XMIN,
        "ymin": YMIN,
        "xmax": XMAX,
        "ymax": YMAX,
    },
    "search_bbox_west_south_east_north": bbox,
    "images_returned": len(records),
    "images_inside_lidar_bounds": inside_count,
    "new_images_downloaded": downloaded,
    "download_failures": failed,
    "metadata_file": str(METADATA_PATH),
    "warning": (
        "Coordinate overlap is necessary but does not prove that a "
        "street-view camera faces the LiDAR sample area or that objects "
        "can be matched between image and LiDAR."
    ),
}

SUMMARY_PATH.write_text(
    json.dumps(summary, indent=2),
    encoding="utf-8",
)

print("\nSearch finished.")
print("Images returned:", len(records))
print("Images inside LiDAR bounds:", inside_count)
print("New images downloaded:", downloaded)
print("Download failures:", failed)
print("Metadata:", METADATA_PATH)
print("Summary:", SUMMARY_PATH)
