
import os
import time
import pandas as pd
import requests

csv_path = "data/street_imagery/processed/autzen_mapillary_metadata.csv"
output_dir = "data/street_imagery/raw"

os.makedirs(output_dir, exist_ok=True)

df = pd.read_csv(csv_path)
df = df.dropna(subset=["thumbnail_url"]).head(5)

session = requests.Session()
session.headers.update({
    "User-Agent": "GeoSense-Phase3/1.0 (educational research)"
})

success = 0

for _, row in df.iterrows():
    image_id = str(row["image_id"])
    url = row["thumbnail_url"]
    output_path = os.path.join(output_dir, f"{image_id}.jpg")

    try:
        response = session.get(url, timeout=30)
        response.raise_for_status()

        content_type = response.headers.get("Content-Type", "").lower()

        if not content_type.startswith("image/"):
            print(f"Skipped {image_id}: response is not an image.")
            continue

        with open(output_path, "wb") as f:
            f.write(response.content)

        print(f"Downloaded: {image_id}.jpg ({len(response.content):,} bytes)")
        success += 1

    except requests.RequestException as exc:
        print(f"Failed: {image_id} — {exc}")

    time.sleep(1)

print(f"\nCompleted: {success} of {len(df)} images downloaded.")
print("Output folder:", output_dir)
