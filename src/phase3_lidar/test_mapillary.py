
import os
import requests
from dotenv import load_dotenv

load_dotenv()

token = os.getenv("MAPILLARY_TOKEN")

if not token or token == "PASTE_YOUR_TOKEN_HERE":
    raise SystemExit(
        "ERROR: Set your actual MAPILLARY_TOKEN in the project .env file."
    )

# Small test area around central Eugene, Oregon.
# Order: west, south, east, north
bbox = "-123.090,44.050,-123.085,44.055"
url = "https://graph.mapillary.com/images"

params = {
    "bbox": bbox,
    "fields": "id,thumb_1024_url,computed_geometry,captured_at",
    "limit": 3,
}

headers = {"Authorization": f"OAuth {token}"}

print("Testing Mapillary access for Eugene, Oregon...")

try:
    response = requests.get(
        url,
        params=params,
        headers=headers,
        timeout=90,
    )

    print("HTTP status:", response.status_code)

    if response.status_code != 200:
        print("API response:", response.text[:1000])
        raise SystemExit(
            "Mapillary request failed. Check the status and error above."
        )

    result = response.json()
    images = result.get("data", [])

    print("Images returned:", len(images))

    for image in images:
        print(
            "Image ID:", image.get("id"),
            "| Thumbnail available:", bool(image.get("thumb_1024_url"))
        )

    if images:
        print("SUCCESS: Mapillary returned imagery metadata.")
    else:
        print(
            "No images returned in this test area. "
            "We will try a wider area before concluding coverage is absent."
        )

except requests.RequestException as error:
    print("Network error:", error)
