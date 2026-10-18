
from pathlib import Path
import json
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]

PROFILE_DIR = ROOT / "outputs" / "reports" / "site_profiles"
FIGURE_DIR = ROOT / "docs" / "exercise5" / "figures"
FIGURE_DIR.mkdir(parents=True, exist_ok=True)

def load_profile(filename):
    path = PROFILE_DIR / filename
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)

area1 = load_profile("autzen_area_1.json")
area2 = load_profile("autzen_area_2.json")
profiles = [area1, area2]
labels = ["Area 1", "Area 2"]

def get_metric(profile, section, metric):
    return profile["bounding_box"]["metrics"][section][metric]

ground_points = [
    get_metric(p, "ground_point_count", "")
    if False else p["bounding_box"]["metrics"]["ground_point_count"]
    for p in profiles
]
vegetation_points = [
    get_metric(p, "classification_point_counts", "")
    if False else int(p["bounding_box"]["metrics"]
                      ["classification_point_counts"].get("5", 0))
    for p in profiles
]
mean_elevation = [
    p["bounding_box"]["metrics"]["dem"]["mean"] for p in profiles
]
mean_slope = [
    p["bounding_box"]["metrics"]["slope"]["mean"] for p in profiles
]

fig, axes = plt.subplots(2, 2, figsize=(12, 8))
fig.suptitle(
    "GeoSense Agent – Phase III\n"
    "Comparison of Two LiDAR–Terrain Site Profiles",
    fontsize=15
)

panels = [
    (axes[0, 0], ground_points, "Ground-classified points", "Point count"),
    (axes[0, 1], vegetation_points, "High-vegetation-classified points",
     "Point count"),
    (axes[1, 0], mean_elevation, "Mean DEM elevation", "Elevation (US survey ft)"),
    (axes[1, 1], mean_slope, "Mean slope", "Slope (degrees)")
]

for ax, values, title, ylabel in panels:
    bars = ax.bar(labels, values)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.bar_label(bars, fmt="%.2f" if max(values) < 1000 else "%.0f",
                 padding=3)

fig.text(
    0.5, 0.015,
    "Both boxes cover 90,000 sq ft. Descriptive results only; "
    "vegetation point counts are not canopy-cover percentages.",
    ha="center", fontsize=9
)
fig.tight_layout(rect=[0, 0.05, 1, 0.92])

output = FIGURE_DIR / "Fig_07_site_profile_comparison.png"
fig.savefig(output, dpi=200, bbox_inches="tight")
plt.close(fig)

print("Figure 07 created successfully:")
print(output)
print("\nValues read from existing site-profile JSON files:")
for label, ground, vegetation, elevation, slope in zip(
        labels, ground_points, vegetation_points, mean_elevation, mean_slope):
    print(
        f"{label}: ground points={ground:,}, "
        f"high vegetation points={vegetation:,}, "
        f"mean DEM elevation={elevation:.2f} ft, "
        f"mean slope={slope:.2f} degrees"
    )
