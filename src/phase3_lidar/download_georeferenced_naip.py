
from pathlib import Path
import json
import requests
import rasterio

ROOT = Path(r"D:\GeoSense_Phase3")
AERIAL_DIR = ROOT / "data" / "aerial"
AERIAL_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_TIF = AERIAL_DIR / "autzen_naip_georeferenced.tif"
OUTPUT_JSON = AERIAL_DIR / "autzen_naip_georeferenced_metadata.json"

# Original LiDAR bounds in EPSG:2992.
# These bounds define the target area, with a small margin.
xmin = 635577.79 - 100
ymin = 848882.15 - 100
xmax = 639003.73 + 100
ymax = 853537.66 + 100

# USGS NAIP ImageServer.
service = (
    "https://imagery.nationalmap.gov/arcgis/rest/services/"
    "USGSNAIPImagery/ImageServer"
)
export_url = service + "/exportImage"

# Request a georeferenced image in the LiDAR's horizontal CRS.
params = {
    "bbox": f"{xmin},{ymin},{xmax},{ymax}",
    "bboxSR": 2992,
    "imageSR": 2992,
    "size": "1500,1500",
    "format": "jpgpng",
    "pixelType": "U8",
    "interpolation": "RSP_BilinearInterpolation",
    "f": "json",
}

print("Requesting NAIP image in EPSG:2992...")
response = requests.get(export_url, params=params, timeout=120)
response.raise_for_status()

result = response.json()

if "error" in result:
    raise RuntimeError(
        "ImageServer error: " + json.dumps(result["error"], indent=2)
    )

image_url = result.get("href")
if not image_url:
    raise RuntimeError(
        "The service did not return an image URL. Response: "
        + json.dumps(result, indent=2)
    )

print("Image URL received. Downloading GeoTIFF...")
image_response = requests.get(image_url, timeout=180)
image_response.raise_for_status()

if not image_response.content:
    raise RuntimeError("The downloaded image is empty.")

OUTPUT_TIF.write_bytes(image_response.content)

# Verify the downloaded raster's actual georeferencing.
with rasterio.open(OUTPUT_TIF) as src:
    summary = {
        "file": str(OUTPUT_TIF),
        "width_pixels": src.width,
        "height_pixels": src.height,
        "band_count": src.count,
        "crs": str(src.crs),
        "bounds": {
            "left": src.bounds.left,
            "bottom": src.bounds.bottom,
            "right": src.bounds.right,
            "top": src.bounds.top,
        },
        "pixel_size_map_units": {
            "x": abs(src.transform.a),
            "y": abs(src.transform.e),
        },
        "file_size_MB": round(OUTPUT_TIF.stat().st_size / 1e6, 2),
        "service_response": result,
    }

OUTPUT_JSON.write_text(
    json.dumps(summary, indent=2),
    encoding="utf-8",
)

print("\nImage downloaded and inspected.")
print("File:", OUTPUT_TIF)
print("Dimensions:", src.width, "x", src.height)
print("Bands:", src.count)
print("CRS:", src.crs)
print("Bounds:", src.bounds)
print("Pixel size in map units:", src.transform.a, src.transform.e)
print("File size (MB):", summary["file_size_MB"])
print("Metadata:", OUTPUT_JSON)
