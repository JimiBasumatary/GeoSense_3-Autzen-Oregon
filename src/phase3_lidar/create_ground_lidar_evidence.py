
from pathlib import Path
import json
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]

METADATA_FILE = ROOT / "outputs" / "reports" / "ground_lidar_summary.txt"
FIGURE_DIR = ROOT / "docs" / "exercise5" / "figures"
FIGURE_DIR.mkdir(parents=True, exist_ok=True)

with METADATA_FILE.open("r", encoding="utf-8") as file:
    data = json.load(file)

summary = data["summary"]
metadata = summary["metadata"]
bounds = summary["bounds"]
srs = summary["srs"]

rows = [
    ["Input file", data["filename"]],
    ["File size", f"{data['file_size']:,} bytes ({data['file_size']/1_000_000:.2f} MB)"],
    ["Ground LiDAR points", f"{summary['num_points']:,}"],
    ["PDAL version", data["pdal_version"]],
    ["LAS version", f"{metadata['major_version']}.{metadata['minor_version']}"],
    ["Point format", str(metadata["dataformat_id"])],
    ["Horizontal CRS", "EPSG:2992 — NAD83 / Oregon GIC Lambert"],
    ["Vertical CRS", "EPSG:6360 — NAVD88 height"],
    ["Minimum X", f"{bounds['minx']:,.2f}"],
    ["Maximum X", f"{bounds['maxx']:,.2f}"],
    ["Minimum Y", f"{bounds['miny']:,.2f}"],
    ["Maximum Y", f"{bounds['maxy']:,.2f}"],
    ["Minimum elevation", f"{bounds['minz']:.2f} US survey ft"],
    ["Maximum elevation", f"{bounds['maxz']:.2f} US survey ft"],
    ["Compressed", str(metadata["compressed"])],
]

fig, ax = plt.subplots(figsize=(11, 7))
ax.axis("off")

fig.suptitle(
    "GeoSense Agent – Phase III\nGround LiDAR Metadata",
    fontsize=17,
    fontweight="bold",
    y=0.96
)

table = ax.table(
    cellText=rows,
    colLabels=["Property", "Verified metadata value"],
    cellLoc="left",
    colLoc="left",
    loc="center",
    colWidths=[0.30, 0.70]
)
table.auto_set_font_size(False)
table.set_fontsize(9)
table.scale(1, 1.45)

ax.set_title(
    "Source: saved PDAL readers.las metadata report",
    fontsize=10,
    pad=12
)

fig.text(
    0.5, 0.025,
    "Ground-classified output generated from the original LiDAR dataset. "
    "Elevations are in US survey feet.",
    ha="center",
    fontsize=9
)

fig.tight_layout(rect=[0.02, 0.06, 0.98, 0.88])

output = FIGURE_DIR / "Fig_02_ground_lidar_metadata.png"
fig.savefig(output, dpi=200, bbox_inches="tight")
plt.close(fig)

print("Figure 02 created successfully:")
print(output)
print(f"Ground points: {summary['num_points']:,}")
print(f"Elevation range: {bounds['minz']:.2f} to {bounds['maxz']:.2f} US survey ft")
