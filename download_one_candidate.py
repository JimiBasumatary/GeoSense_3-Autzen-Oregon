
import os
from pathlib import Path

import requests
from dotenv import load_dotenv

ROOT = Path(r"D:\GeoSense_Phase3")
load_dotenv(ROOT / ".env")

token = os.getenv("MAPILLARY_TOKEN", "").strip()

if not token:
    raise SystemExit("Mapillary token not found in .env")

image_id = "1195814025796740"
output_dir = (
    ROOT / "data" / "streetview" / "nearest_candidates" / "inspection"
)
output_dir.mkdir(parents=True, exist_ok=True)

session = requests.Session()
session.headers.update({"Authorization": f"OAuth {token}"})

try:
    response = session.get(
        f"https://graph.mapillary.com/{image_id}",
        params={"fields": "id,thumb_1024_url"},
        timeout=(15, 45),
    )
    response.raise_for_status()

    image_url = response.json().get("thumb_1024_url")
    if not image_url:
        raise SystemExit("Mapillary returned no thumbnail URL.")

    # Do not send the OAuth header to the external thumbnail host.
    image_response = requests.get(image_url, timeout=(20, 90))
    image_response.raise_for_status()

    if not image_response.headers.get("Content-Type", "").startswith("image/"):
        raise SystemExit("The response was not an image.")

    output_path = output_dir / f"{image_id}.jpg"
    output_path.write_bytes(image_response.content)

    print("Download successful.")
    print("Image:", output_path)
    print("Size:", output_path.stat().st_size, "bytes")

except requests.RequestException as error:
    raise SystemExit(
        f"Download failed: {type(error).__name__}: {error}"
    )
