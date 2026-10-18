
import os
import csv
import requests
from dotenv import load_dotenv

load_dotenv(".env")
token = os.getenv("MAPILLARY_TOKEN", "").strip()

if not token:
    raise SystemExit("ERROR: Mapillary token not found in .env")

image_dir = r"data\streetview\images"
os.makedirs(image_dir, exist_ok=True)

metadata_path = os.path.join(image_dir, "image_metadata.csv")

# Search different parts of Eugene, Oregon.
# Format: west, south, east, north
boxes = [
    "-123.13,44.03,-123.11,44.05",
    "-123.12,44.05,-123.10,44.07",
    "-123.11,44.02,-123.09,44.04",
    "-123.10,44.04,-123.08,44.06",
    "-123.09,44.06,-123.07,44.08",
    "-123.08,44.03,-123.06,44.05",
    "-123.07,44.05,-123.05,44.07",
    "-123.06,44.07,-123.04,44.09",
    "-123.15,44.04,-123.13,44.06",
    "-123.14,44.06,-123.12,44.08",
]

# Preserve existing metadata.
existing = {}
if os.path.exists(metadata_path):
    with open(metadata_path, newline="", encoding="utf-8") as file:
        for row in csv.DictReader(file):
            if row.get("image_id"):
                existing[row["image_id"]] = row

records = {}
session = requests.Session()
session.headers.update({"Authorization": f"OAuth {token}"})

for bbox in boxes:
    if len(set(existing) | set(records)) >= 20:
        break

    print(f"\nSearching: {bbox}")

    try:
        response = session.get(
            "https://graph.mapillary.com/images",
            params={
                "bbox": bbox,
                "fields": "id,thumb_1024_url,computed_geometry,captured_at",
                "limit": 10,
            },
            timeout=(10, 30),
        )
        response.raise_for_status()

        for item in response.json().get("data", []):
            records[item["id"]] = item

    except requests.RequestException as error:
        print("Search failed:", error)

    print("Unique new records found so far:", len(records))

downloaded = 0
failed = 0

for image_id, item in records.items():
    filename = f"{image_id}.jpg"
    filepath = os.path.join(image_dir, filename)
    status = "failed"

    try:
        url = item.get("thumb_1024_url")

        if os.path.exists(filepath) and os.path.getsize(filepath) > 0:
            status = "already_exists"
            print("Already downloaded:", filename)

        elif not url:
            print("No thumbnail URL:", image_id)
            failed += 1

        else:
            image_response = session.get(url, timeout=(10, 45))
            image_response.raise_for_status()

            content_type = image_response.headers.get(
                "Content-Type", ""
            )
            if not content_type.startswith("image/"):
                raise ValueError(
                    f"Unexpected content type: {content_type}"
                )

            with open(filepath, "wb") as output:
                output.write(image_response.content)

            status = "downloaded"
            downloaded += 1
            print("Downloaded:", filename)

    except (requests.RequestException, OSError, ValueError) as error:
        print(f"Failed {image_id}: {error}")
        failed += 1

    geometry = item.get("computed_geometry") or {}
    coords = geometry.get("coordinates", [None, None])

    existing[image_id] = {
        "image_id": image_id,
        "filename": filename,
        "captured_at": item.get("captured_at"),
        "longitude": coords[0],
        "latitude": coords[1],
        "download_status": status,
    }

# Keep prior rows and add/update records found in this run.
with open(metadata_path, "w", newline="", encoding="utf-8") as file:
    fields = [
        "image_id",
        "filename",
        "captured_at",
        "longitude",
        "latitude",
        "download_status",
    ]
    writer = csv.DictWriter(file, fieldnames=fields)
    writer.writeheader()
    writer.writerows(existing.values())

print("\nDownload process finished.")
print("New images downloaded:", downloaded)
print("Download failures:", failed)
print("Metadata records retained:", len(existing))
print("Metadata:", os.path.abspath(metadata_path))
print("Image folder:", os.path.abspath(image_dir))
