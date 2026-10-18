
from pathlib import Path
import json

import rasterio
from rasterio.transform import from_bounds

ROOT = Path(r"D:\GeoSense_Phase3")
AERIAL_DIR = ROOT / "data" / "aerial"

SOURCE = AERIAL_DIR / "autzen_naip_georeferenced.tif"
METADATA = AERIAL_DIR / "autzen_naip_georeferenced_metadata.json"
OUTPUT = AERIAL_DIR / "autzen_naip_georeferenced_fixed.tif"

if not SOURCE.exists():
    raise FileNotFoundError(f"Image not found: {SOURCE}")

if not METADATA.exists():
    raise FileNotFoundError(f"Metadata not found: {METADATA}")

metadata = json.loads(METADATA.read_text(encoding="utf-8"))
service = metadata["service_response"]
extent = service["extent"]

xmin = float(extent["xmin"])
ymin = float(extent["ymin"])
xmax = float(extent["xmax"])
ymax = float(extent["ymax"])

with rasterio.open(SOURCE) as src:
    image = src.read()
    height = src.height
    width = src.width
    count = src.count
    dtype = src.dtypes[0]

# Build a north-up affine transform using the service extent.
transform = from_bounds(
    xmin, ymin, xmax, ymax, width, height
)

profile = {
    "driver": "GTiff",
    "width": width,
    "height": height,
    "count": count,
    "dtype": dtype,
    "crs": "EPSG:2992",
    "transform": transform,
    "compress": "deflate",
}

with rasterio.open(OUTPUT, "w", **profile) as dst:
    dst.write(image)

# Verify the result.
with rasterio.open(OUTPUT) as check:
    print("Georeferenced GeoTIFF created.")
    print("File:", OUTPUT)
    print("Dimensions:", check.width, "x", check.height)
    print("Bands:", check.count)
    print("CRS:", check.crs)
    print("Bounds:", check.bounds)
    print("Pixel width (map units):", abs(check.transform.a))
    print("Pixel height (map units):", abs(check.transform.e))
    print("Pixel width (approx. metres):",
          round(abs(check.transform.a) * 0.3048, 3))
    print("Pixel height (approx. metres):",
          round(abs(check.transform.e) * 0.3048, 3))

print("\nImportant: This reconstructs georeferencing from the service "
      "extent. It does not independently verify image-to-LiDAR alignment.")
