
import os
import glob
from PIL import Image, ImageDraw

folder = "outputs/street_imagery/overlays"
output_path = "outputs/street_imagery/segmentation_contact_sheet.jpg"

files = sorted(glob.glob(os.path.join(folder, "*_overlay.png")))

if not files:
    raise FileNotFoundError(f"No overlay images found in {folder}")

thumb_width = 480
thumb_height = 270
label_height = 35
columns = 2
rows = (len(files) + columns - 1) // columns

sheet = Image.new(
    "RGB",
    (columns * thumb_width, rows * (thumb_height + label_height)),
    "white"
)
draw = ImageDraw.Draw(sheet)

for i, path in enumerate(files):
    image = Image.open(path).convert("RGB")
    image.thumbnail((thumb_width, thumb_height))

    x = (i % columns) * thumb_width
    y = (i // columns) * (thumb_height + label_height)

    sheet.paste(image, (x, y))
    draw.text(
        (x + 5, y + thumb_height + 5),
        os.path.basename(path),
        fill="black"
    )

os.makedirs(os.path.dirname(output_path), exist_ok=True)
sheet.save(output_path, quality=95)

print("Contact sheet created:", output_path)
print("Images included:", len(files))
