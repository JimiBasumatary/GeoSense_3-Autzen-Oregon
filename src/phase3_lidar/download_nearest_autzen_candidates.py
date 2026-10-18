
import csv
import os
import time
from pathlib import Path

import requests
from dotenv import load_dotenv
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

ROOT = Path(r"D:\GeoSense_Phase3")
SOURCE_CSV = (
    ROOT / "data" / "streetview" / "autzen_candidates"
    / "image_metadata.csv"
)
OUTPUT_DIR = ROOT / "data" / "streetview" / "nearest_candidates"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

load_dotenv(ROOT / ".env")
token = os.getenv("MAPILLARY_TOKEN", "").strip()

if not token:
    raise SystemExit("ERROR: Mapillary token not found in .env")

with SOURCE_CSV.open(newline="", encoding="utf-8") as file:
    rows = list(csv.DictReader(file))

xmin, ymin, xmax, ymax = 635577.79, 848882.15, 639003.73, 853537.66

for row in rows:
    x = float(row["x_epsg2992"])
    y = float(row["y_epsg2992"])
    dx = max(xmin - x, 0, x - xmax)
    dy = max(ymin - y, 0, y - ymax)
    row["distance_to_lidar_ft"] = (dx * dx + dy * dy) ** 0.5

nearest = sorted(rows, key=lambda row: row["distance_to_lidar_ft"])[:5]

session = requests.Session()
retry = Retry(
    total=3,
    connect=3,
    read=2,
    backoff_factor=1,
    status_forcelist=[429, 500, 502, 503, 504],
    allowed_methods=["GET"],
)
session.mount("https://", HTTPAdapter(max_retries=retry))
session.headers.update({"Authorization": f"OAuth {token}"})

downloaded = 0
failed = 0

for row in nearest:
    image_id = row["image_id"]
    output_path = OUTPUT_DIR / f"{image_id}.jpg"

    if output_path.exists() and output_path.stat().st_size > 0:
        print("Already available:", output_path.name)
        continue

    try:
        metadata_response = session.get(
            f"https://graph.mapillary.com/{image_id}",
            params={"fields": "id,thumb_1024_url"},
            timeout=(15, 45),
        )
        metadata_response.raise_for_status()
        image_url = metadata_response.json().get("thumb_1024_url")

        if not image_url:
            print("No thumbnail URL:", image_id)
            failed += 1
            continue

        # Use a separate request without the Mapillary OAuth header
        # for the externally hosted thumbnail.
        image_response = requests.get(image_url, timeout=(20, 90))
        image_response.raise_for_status()

        if not image_response.headers.get("Content-Type", "").startswith("image/"):
            raise ValueError("Response is not an image")

        output_path.write_bytes(image_response.content)
        downloaded += 1
        print("Downloaded:", output_path.name, "-", len(image_response.content), "bytes")

    except (requests.RequestException, OSError, ValueError) as error:
        failed += 1
        print("Failed:", image_id, type(error).__name__, str(error)[:180])

    time.sleep(1)

print("\nFinished.")
print("New images downloaded:", downloaded)
print("Failures:", failed)
print("Folder:", OUTPUT_DIR)
