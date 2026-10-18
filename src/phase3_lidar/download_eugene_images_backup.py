
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

# Areas that returned image records in the previous test.
boxes = [
    "-123.12,44.04,-123.11,44.05",
    "-123.10,44.05,-123.09,44.06",
    "-123.06,44.06,-123.05,44.07",
]

session = requests.Session()
session.headers.update({"Authorization": f"OAuth {token}"})

records = {}
for bbox in boxes:
    print(f"\nSearching: {bbox}")

    try:
        response = session.get(
            "https://graph.mapillary.com/images",
            params={
                "bbox": bbox,
                "fields": "id,thumb_1024_url,computed_geometry,captured_at",
                "limit": 3,
            },
            timeout=(10, 30),
        )
        response.raise_for_status()

        for item in response.json().get("data", []):
            records[item["id"]] = item

    except requests.RequestException as error:
        print("Search failed:", error)

print(f"\nUnique image records found: {len(records)}")

metadata_path = os.path.join(image_dir, "image_metadata.csv")
downloaded = 0

with open(metadata_path, "w", newline="", encoding="utf-8") as file:
    writer = csv.DictWriter(
        file,
        fieldnames=[
            "image_id",
            "filename",
            "captured_at",
            "longitude",
            "latitude",
            "download_status",
        ],
    )
    writer.writeheader()

    for image_id, item in records.items():
        filename = f"{image_id}.jpg"
        filepath = os.path.join(image_dir, filename)
        status = "failed"

        try:
            url = item.get("thumb_1024_url")

            if not url:
                print(f"No thumbnail URL for {image_id}")
            elif os.path.exists(filepath) and os.path.getsize(filepath) > 0:
                status = "already_exists"
                print(f"Already downloaded: {filename}")
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
                print(f"Downloaded: {filename}")

        except (requests.RequestException, OSError, ValueError) as error:
            print(f"Failed {image_id}: {error}")

        geometry = item.get("computed_geometry") or {}
        coords = geometry.get("coordinates", [None, None])

        writer.writerow({
            "image_id": image_id,
            "filename": filename,
            "captured_at": item.get("captured_at"),
            "longitude": coords[0],
            "latitude": coords[1],
            "download_status": status,
        })

print("\nDownload process finished.")
print("New images downloaded:", downloaded)
print("Metadata:", metadata_path)
print("Image folder:", os.path.abspath(image_dir))
