
from pathlib import Path
from getpass import getpass
from datetime import datetime, timezone
import csv
import requests

PROJECT_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_DIR / "data" / "street_imagery" / "processed"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CSV_FILE = OUTPUT_DIR / "autzen_mapillary_metadata.csv"

# Approximate bounding box around Autzen Stadium and nearby roads.
# Format: west, south, east, north (longitude, latitude).
BBOX = "-123.0755,44.0537,-123.0614,44.0627"

token = getpass("Paste your NEW Mapillary access token (input hidden): ").strip()

if not token:
    raise SystemExit("No token provided. Run the script again with your token.")

url = "https://graph.mapillary.com/images"

params = {

    "fields": (
        "id,computed_geometry,captured_at,compass_angle,"
        "sequence,thumb_1024_url"
    ),
    "bbox": BBOX,
    "limit": 100,
}

print("\nSearching for Mapillary images near Autzen Stadium...")


try:
    headers = {"Authorization": f"OAuth {token}"}

    response = requests.get(
        url,
        params=params,
        headers=headers,
        timeout=60
    )
    response.raise_for_status()
    result = response.json()

except requests.RequestException as exc:
    print(f"API request failed: {exc}")

    if "response" in locals() and response is not None:
        print("API error details:")
        print(response.text[:1000])

    raise SystemExit(
        "Check the API error details above. Do not share your access token."
    )


images = result.get("data", [])

rows = []

for image in images:
    geometry = image.get("computed_geometry") or {}
    coordinates = geometry.get("coordinates") or [None, None]

    captured_at = image.get("captured_at")
    if captured_at is not None:
        captured_at = datetime.fromtimestamp(
            captured_at / 1000, tz=timezone.utc
        ).isoformat()

    rows.append({
        "image_id": image.get("id"),
        "longitude": coordinates[0],
        "latitude": coordinates[1],
        "captured_at_utc": captured_at,
        "compass_angle": image.get("compass_angle"),
        "sequence_id": image.get("sequence"),
        "thumbnail_url": image.get("thumb_1024_url"),
    })

fieldnames = [
    "image_id",
    "longitude",
    "latitude",
    "captured_at_utc",
    "compass_angle",
    "sequence_id",
    "thumbnail_url",
]

with CSV_FILE.open("w", newline="", encoding="utf-8-sig") as file:
    writer = csv.DictWriter(file, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)

print(f"\nImages returned: {len(rows)}")
print(f"CSV saved to: {CSV_FILE}")

if not rows:
    print(
        "No images returned for this bounding box. "
        "We can inspect the API response and adjust the search area."
    )
else:
    print("\nFirst five records:")
    for row in rows[:5]:
        print(
            f"ID={row['image_id']}, "
            f"lon={row['longitude']}, lat={row['latitude']}"
        )

print("\nMetadata extraction finished.")
