
from pathlib import Path
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
from matplotlib.colors import ListedColormap

ROOT = Path(r"D:\GeoSense_Phase3")
IMAGE_ID = "141300064660795"

IMAGE_PATH = ROOT / "data" / "streetview" / "images" / f"{IMAGE_ID}.jpg"
OUTPUT_DIR = ROOT / "outputs" / "streetview" / "validation"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

REFERENCE_PATH = OUTPUT_DIR / f"{IMAGE_ID}_manual_reference.png"

# Cityscapes class IDs; 255 means unlabelled.
CLASSES = {
    "Road": 0,
    "Sidewalk": 1,
    "Building": 2,
    "Vegetation": 8,
    "Sky": 10,
    "Car": 13,
}

COLOURS = {
    0: "#808080",
    1: "#FFD700",
    2: "#FF6347",
    8: "#228B22",
    10: "#1E90FF",
    13: "#BA55D3",
}

image = np.array(Image.open(IMAGE_PATH).convert("RGB"))
height, width = image.shape[:2]

reference = np.full((height, width), 255, dtype=np.uint8)

# Brush edits a circular area in the reference mask.
brush_radius = 8
current_class = 0

fig, ax = plt.subplots(figsize=(14, 8))
plt.subplots_adjust(bottom=0.24)

ax.imshow(image)
ax.set_title(
    "Manual reference annotation — choose a class, then paint visible regions"
)
ax.axis("off")

overlay_artist = ax.imshow(
    np.zeros((height, width, 4), dtype=float),
    interpolation="nearest",
)

buttons = []
button_classes = list(CLASSES.items())

def refresh_overlay():
    rgba = np.zeros((height, width, 4), dtype=float)
    for class_id, colour in COLOURS.items():
        rgb = np.array(
            [int(colour[i:i+2], 16) for i in (1, 3, 5)],
            dtype=float
        ) / 255.0
        selected = reference == class_id
        rgba[selected, :3] = rgb
        rgba[selected, 3] = 0.55
    overlay_artist.set_data(rgba)
    fig.canvas.draw_idle()

def choose_class(label):
    global current_class
    current_class = CLASSES[label]
    ax.set_title(
        f"Painting: {label} (class {current_class}) | "
        "Unlabelled pixels are excluded"
    )
    fig.canvas.draw_idle()

def paint(event):
    if event.inaxes != ax or event.xdata is None or event.ydata is None:
        return
    x = int(round(event.xdata))
    y = int(round(event.ydata))
    yy, xx = np.ogrid[:height, :width]
    circle = (xx - x) ** 2 + (yy - y) ** 2 <= brush_radius ** 2
    reference[circle] = current_class
    refresh_overlay()

def save_reference(event):
    Image.fromarray(reference, mode="L").save(REFERENCE_PATH)
    labelled = int(np.count_nonzero(reference != 255))
    print("Manual reference saved:", REFERENCE_PATH)
    print("Labelled pixels:", labelled)
    print("Unlabelled pixels:", reference.size - labelled)

def increase_brush(event):
    global brush_radius
    brush_radius = min(40, brush_radius + 2)
    print("Brush radius:", brush_radius)

def decrease_brush(event):
    global brush_radius
    brush_radius = max(2, brush_radius - 2)
    print("Brush radius:", brush_radius)

for index, (label, class_id) in enumerate(button_classes):
    left = 0.04 + index * 0.115
    button_ax = fig.add_axes([left, 0.13, 0.105, 0.055])
    button = Button(button_ax, label)
    button.on_clicked(lambda event, name=label: choose_class(name))
    buttons.append(button)

save_ax = fig.add_axes([0.08, 0.035, 0.18, 0.055])
save_button = Button(save_ax, "Save reference")
save_button.on_clicked(save_reference)

minus_ax = fig.add_axes([0.39, 0.035, 0.12, 0.055])
minus_button = Button(minus_ax, "Smaller brush")
minus_button.on_clicked(decrease_brush)

plus_ax = fig.add_axes([0.54, 0.035, 0.12, 0.055])
plus_button = Button(plus_ax, "Larger brush")
plus_button.on_clicked(increase_brush)

fig.canvas.mpl_connect("button_press_event", paint)
plt.show()
