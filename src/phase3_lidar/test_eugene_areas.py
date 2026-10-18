
import os
import requests
from dotenv import load_dotenv

load_dotenv(".env")
token = os.getenv("MAPILLARY_TOKEN", "").strip()

if not token:
    raise SystemExit("ERROR: Mapillary token not found in .env")

boxes = [
    "-123.12,44.04,-123.11,44.05",
    "-123.10,44.05,-123.09,44.06",
    "-123.08,44.05,-123.07,44.06",
    "-123.06,44.06,-123.05,44.07",
]

session = requests.Session()
session.headers.update({"Authorization": f"OAuth {token}"})

for number, bbox in enumerate(boxes, start=1):
    print(f"\nTesting area {number}: {bbox}")

    try:
        response = session.get(
            "https://graph.mapillary.com/images",
            params={
                "bbox": bbox,
                "fields": "id",
                "limit": 3,
            },
            timeout=(10, 20),
        )

        print("HTTP:", response.status_code)

        if response.ok:
            images = response.json().get("data", [])
            print("Images returned:", len(images))

            for image in images:
                print("Image ID:", image.get("id"))
        else:
            print("API response:", response.text[:300])

    except requests.RequestException as error:
        print("Request error:", type(error).__name__, str(error)[:200])
