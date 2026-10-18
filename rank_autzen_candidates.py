
import csv
from pathlib import Path

path = Path("data/streetview/nearest_candidates/all_candidate_heading_screen.csv")

with path.open("r", encoding="utf-8", newline="") as f:
    rows = list(csv.DictReader(f))

rows = [
    r for r in rows
    if r["api_status"] == "ok"
    and r["heading_difference_deg"]
]

rows.sort(
    key=lambda r: (
        float(r["distance_to_footprint_ft"]),
        float(r["heading_difference_deg"])
    )
)

print("\nTOP 15 AUTZEN CANDIDATES")
print("-" * 75)

for i, r in enumerate(rows[:15], start=1):
    print(
        f"{i:02d}. {r['filename']} | "
        f"distance={float(r['distance_to_footprint_ft']):.2f} ft | "
        f"heading difference="
        f"{float(r['heading_difference_deg']):.2f} degrees | "
        f"{r['screening_result']}"
    )

print("\nTotal candidates ranked:", len(rows))
