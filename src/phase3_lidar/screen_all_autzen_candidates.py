
import csv
import math
import os
import time
from pathlib import Path

import requests
from dotenv import load_dotenv

ROOT = Path(r"D:\GeoSense_Phase3")
SOURCE = ROOT / "data" / "streetview" / "autzen_candidates" / "image_metadata.csv"
OUTPUT = ROOT / "data" / "streetview" / "nearest_candidates" / "all_candidate_heading_screen.csv"

XMIN, YMIN = 635577.79, 848882.15
XMAX, YMAX = 639003.73, 853537.66

load_dotenv(ROOT / ".env")
token = os.getenv("MAPILLARY_TOKEN", "").strip()
if not token:
    raise SystemExit("ERROR: Mapillary token not found in .env")

with SOURCE.open(newline="", encoding="utf-8") as f:
    rows = list(csv.DictReader(f))

session = requests.Session()
session.headers.update({"Authorization": f"OAuth {token}"})
results = []

for index, row in enumerate(rows, start=1):
    image_id = row["image_id"]
    x, y = float(row["x_epsg2992"]), float(row["y_epsg2992"])

    target_x = min(max(x, XMIN), XMAX)
    target_y = min(max(y, YMIN), YMAX)
    dx, dy = target_x - x, target_y - y
    distance = math.hypot(dx, dy)

    # Bearing clockwise from north toward nearest footprint point.
    bearing = math.degrees(math.atan2(dx, dy)) % 360 if distance else None

    result = {
        "image_id": image_id,
        "filename": row["filename"],
        "x_epsg2992": x,
        "y_epsg2992": y,
        "distance_to_footprint_ft": round(distance, 2),
        "camera_heading_deg": "",
        "target_bearing_deg": round(bearing, 2) if bearing is not None else "",
        "heading_difference_deg": "",
        "screening_result": "",
        "api_status": "",
    }

    try:
        response = session.get(
            f"https://graph.mapillary.com/{image_id}",
            params={"fields": "id,compass_angle"},
            timeout=(15, 45),
        )
        response.raise_for_status()
        data = response.json()
        heading = data.get("compass_angle")
        result["api_status"] = "ok"

        if heading is None:
            result["screening_result"] = "heading_missing"
        elif distance == 0:
            result["camera_heading_deg"] = round(float(heading), 2)
            result["screening_result"] = "inside_footprint_check_scene"
        else:
            heading = float(heading)
            difference = abs((heading - bearing + 180) % 360 - 180)
            result["camera_heading_deg"] = round(heading, 2)
            result["heading_difference_deg"] = round(difference, 2)

            if difference <= 45:
                result["screening_result"] = "roughly_toward"
            elif difference >= 135:
                result["screening_result"] = "roughly_away"
            else:
                result["screening_result"] = "sideways_or_uncertain"

    except (requests.RequestException, ValueError, TypeError) as error:
        result["api_status"] = type(error).__name__
        result["screening_result"] = "check_failed"

    results.append(result)
    print(f"{index}/{len(rows)} | {image_id} | {result['screening_result']}")

    time.sleep(0.5)

fields = list(results[0].keys()) if results else []
OUTPUT.parent.mkdir(parents=True, exist_ok=True)

with OUTPUT.open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=fields)
    writer.writeheader()
    writer.writerows(sorted(results, key=lambda r: r["distance_to_footprint_ft"]))

print("\nFinished.")
print("Candidates processed:", len(results))
print("API failures:", sum(r["api_status"] != "ok" for r in results))
print("Results saved:", OUTPUT)
print("Note: heading alignment is only a screening test, not proof of scene overlap.")
