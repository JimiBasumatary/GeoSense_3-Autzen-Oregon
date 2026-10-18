
import os
import glob
from PIL import Image, ImageDraw

folder = "outputs/street_imagery/overlays_cityscapes"
output_path = "outputs/street_imagery/corrected_contact_sheet.jpg"

files = sorted(glob.glob(os.path.join(folder, "*_overlay.png")))

if not files:
    raise FileNotFoundError(f"No corrected overlays found in {folder}")

thumb_w, thumb_h = 480, 270
label_h = 35
columns = 2
rows = (len(files) + columns - 1) // columns

sheet = Image.new(
    "RGB",
    (columns * thumb_w, rows * (thumb_h + label_h)),
    "white"
)
draw = ImageDraw.Draw(sheet)

for i, path in enumerate(files):
    image = Image.open(path).convert("RGB")
    image.thumbnail((thumb_w, thumb_h))

    x = (i % columns) * thumb_w
    y = (i // columns) * (thumb_h + label_h)

    sheet.paste(image, (x, y))
    draw.text(
        (x + 5, y + thumb_h + 5),
        os.path.basename(path),
        fill="black"
    )

sheet.save(output_path, quality=95)

print("Created:", output_path)
print("Images included:", len(files))
