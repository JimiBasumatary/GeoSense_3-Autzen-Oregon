
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(r"D:\GeoSense_Phase3")
INPUT_CSV = (
    ROOT / "outputs" / "sam_georeferenced"
    / "mask13_spatial_height_bins.csv"
)
OUTPUT_PNG = (
    ROOT / "outputs" / "sam_georeferenced"
    / "mask13_height_heatmap.png"
)

df = pd.read_csv(INPUT_CSV)

expected = {
    "row_bin", "column_bin", "point_count", "hag_median_ft"
}
missing = expected - set(df.columns)
if missing:
    raise ValueError(f"Missing required columns: {sorted(missing)}")

height_grid = np.full((4, 4), np.nan)
count_grid = np.zeros((4, 4), dtype=int)

for item in df.itertuples(index=False):
    r = int(item.row_bin) - 1
    c = int(item.column_bin) - 1
    height_grid[r, c] = item.hag_median_ft
    count_grid[r, c] = int(item.point_count)

fig, ax = plt.subplots(figsize=(9, 7))

masked_grid = np.ma.masked_invalid(height_grid)
image = ax.imshow(
    masked_grid,
    interpolation="nearest",
    aspect="equal",
)

fig.colorbar(
    image,
    ax=ax,
    label="Median height above ground (ft)",
)

for r in range(4):
    for c in range(4):
        if np.isfinite(height_grid[r, c]):
            label = (
                f"{height_grid[r, c]:.1f} ft\n"
                f"n={count_grid[r, c]:,}"
            )
        else:
            label = f"No points\nn={count_grid[r, c]:,}"

        ax.text(
            c, r, label,
            ha="center", va="center",
            fontsize=9,
        )

ax.set_xticks(range(4), labels=[1, 2, 3, 4])
ax.set_yticks(range(4), labels=[1, 2, 3, 4])
ax.set_xlabel("Column bin (left to right)")
ax.set_ylabel("Row bin (top to bottom)")
ax.set_title(
    "Mask 13: Spatial Distribution of Median LiDAR HAG"
)

fig.tight_layout()
OUTPUT_PNG.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUTPUT_PNG, dpi=180)
plt.close(fig)

print("Heatmap created:", OUTPUT_PNG)
print("Grid cells:", len(df))
print("Empty cells:", int(df["point_count"].eq(0).sum()))
print(
    "Reminder: Cells follow the mask bounding rectangle, "
    "not the exact roof boundary. HAG statistics and image "
    "registration still require validation."
)
