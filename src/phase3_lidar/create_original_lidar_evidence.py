
from pathlib import Path
import json
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]

METADATA_FILE = ROOT / "outputs" / "reports" / "original_lidar_summary.txt"
FIGURE_DIR = ROOT / "docs" / "exercise5" / "figures"
FIGURE_DIR.mkdir(parents=True, exist_ok=True)

# Read the existing PDAL metadata report.
with METADATA_FILE.open("r", encoding="utf-8") as file:
    text = file.read()

data = json.loads(text)
summary = data["summary"]
bounds = summary["bounds"]
srs = summary["srs"]

point_count = summary["num_points"]
filename = data["filename"]
file_size_mb = data["file_size"] / 1_000_000
pdal_version = data["pdal_version"]

fig, ax = plt.subplots(figsize=(11, 6.5))
ax.axis("off")

fig.suptitle(
    "GeoSense Agent – Phase III\n"
    "Original Classified LiDAR Metadata",
    fontsize=17,
    fontweight="bold",
    y=0.96
)

rows = [
    ["Input file", filename],
    ["File size", f"{file_size_mb:.2f} MB"],
    ["Total LiDAR points", f"{point_count:,}"],
    ["PDAL version", pdal_version],
    ["Point dimensions", str(len(summary["dimensions"].split(", ")))],
    ["Horizontal CRS", srs["horizontal"].split('AUTHORITY["EPSG","')[-1]
        .split('"')[0] if 'AUTHORITY["EPSG","' in srs["horizontal"]
        else "EPSG:2992"],
    ["Horizontal EPSG", "EPSG:2992"],
    ["Vertical EPSG", "EPSG:6360"],
    ["Minimum X", f"{bounds['minx']:,.2f}"],
    ["Maximum X", f"{bounds['maxx']:,.2f}"],
    ["Minimum Y", f"{bounds['miny']:,.2f}"],
    ["Maximum Y", f"{bounds['maxy']:,.2f}"],
    ["Minimum elevation", f"{bounds['minz']:.2f} US survey ft"],
    ["Maximum elevation", f"{bounds['maxz']:.2f} US survey ft"],
]

table = ax.table(
    cellText=rows,
    colLabels=["Property", "Verified metadata value"],
    cellLoc="left",
    colLoc="left",
    loc="center",
    colWidths=[0.28, 0.72]
)

table.auto_set_font_size(False)
table.set_fontsize(9)
table.scale(1, 1.55)

ax.set_title(
    "Source: saved PDAL readers.copc metadata report",
    fontsize=10,
    pad=12
)

fig.text(
    0.5, 0.025,
    "Coordinate values use the dataset's projected coordinate system; "
    "elevations are in US survey feet.",
    ha="center",
    fontsize=9
)

fig.tight_layout(rect=[0.02, 0.06, 0.98, 0.88])

output = FIGURE_DIR / "Fig_01_original_lidar_metadata.png"
fig.savefig(output, dpi=200, bbox_inches="tight")
plt.close(fig)

print("Figure 01 created successfully:")
print(output)
print(f"LiDAR points: {point_count:,}")
print(f"Input file: {filename}")
