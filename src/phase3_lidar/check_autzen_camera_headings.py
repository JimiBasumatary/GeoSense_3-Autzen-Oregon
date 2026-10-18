
import csv
import math
import os
from pathlib import Path

import requests
from dotenv import load_dotenv
from pyproj import Transformer

ROOT = Path(r"D:\GeoSense_Phase3")
CSV_PATH = (
    ROOT / "data" / "streetview" / "autzen_candidates"
    / "image_metadata.csv"
)
OUTPUT_PATH = (
    ROOT / "data" / "streetview" / "nearest_candidates"
    / "camera_heading_check.csv"
)

XMIN, YMIN = 635577.79, 848882.15
XMAX, YMAX = 639003.73, 853537.66

load_dotenv(ROOT / ".env")
token = os.getenv("MAPILLARY_TOKEN", "").strip()
if not token:
    raise SystemExit("Mapillary token not found in .env")

with CSV_PATH.open(newline="", encoding="utf-8") as f:
    rows = list(csv.DictReader(f))

for row in rows:
    x, y = float(row["x_epsg2992"]), float(row["y_epsg2992"])
    dx = max(XMIN - x, 0, x - XMAX)
    dy = max(YMIN - y, 0, y - YMAX)
    row["_distance"] = math.hypot(dx, dy)

nearest = sorted(rows, key=lambda r: r["_distance"])[:5]

session = requests.Session()
session.headers.update({"Authorization": f"OAuth {token}"})
to_geo = Transformer.from_crs("EPSG:2992", "EPSG:4326", always_xy=True)

results = []

for row in nearest:
    image_id = row["image_id"]
    x, y = float(row["x_epsg2992"]), float(row["y_epsg2992"])

    # Nearest point on the rectangular LiDAR footprint.
    target_x = min(max(x, XMIN), XMAX)
    target_y = min(max(y, YMIN), YMAX)

    try:
        response = session.get(
            f"https://graph.mapillary.com/{image_id}",
            params={"fields": "id,compass_angle,computed_geometry"},
            timeout=(15, 45),
        )
        response.raise_for_status()
        data = response.json()
        heading = data.get("compass_angle")

        if heading is None:
            print(image_id, "No compass heading returned")
            continue

        # Bearing from camera position toward nearest footprint point.
        east = target_x - x
        north = target_y - y
        bearing = math.degrees(math.atan2(east, north)) % 360

        # Smallest angular difference between heading and target bearing.
        difference = abs((float(heading) - bearing + 180) % 360 - 180)

        lon, lat = to_geo.transform(target_x, target_y)

        result = {
            "image_id": image_id,
            "filename": row["filename"],
            "distance_to_footprint_ft": round(row["_distance"], 2),
            "camera_heading_deg": round(float(heading), 2),
            "bearing_to_footprint_deg": round(bearing, 2),
            "heading_difference_deg": round(difference, 2),
            "nearest_point_lon": lon,
            "nearest_point_lat": lat,
            "interpretation": (
                "roughly_toward" if difference <= 45
                else "roughly_away" if difference >= 135
                else "sideways_or_uncertain"
            ),
        }
        results.append(result)

        print(
            f"{image_id} | distance={result['distance_to_footprint_ft']} ft"
            f" | heading={result['camera_heading_deg']}°"
            f" | target bearing={result['bearing_to_footprint_deg']}°"
            f" | difference={result['heading_difference_deg']}°"
            f" | {result['interpretation']}"
        )

    except (requests.RequestException, ValueError, TypeError) as error:
        print("Could not check", image_id, type(error).__name__)

    finally:
        pass

    import time
    time.sleep(1)

if results:
    with OUTPUT_PATH.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)

print("\nResults:", OUTPUT_PATH)
print("Images checked:", len(results))
print("Note: heading alignment is a screening test, not proof of scene overlap.")
