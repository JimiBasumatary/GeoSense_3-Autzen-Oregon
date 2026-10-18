
import os
import pandas as pd

base = "outputs/street_imagery"
summary = pd.read_csv(
    os.path.join(base, "segmentation_summary_fixed.csv")
)
screening = pd.read_csv(
    os.path.join(base, "walkability_screening.csv")
)

report_path = os.path.join(base, "exercise4_results.txt")

classes = [
    "road_percent",
    "sidewalk_percent",
    "building_percent",
    "vegetation_percent",
    "sky_percent",
    "car_percent",
    "truck_percent",
    "person_percent",
    "bicycle_percent",
]

with open(report_path, "w", encoding="utf-8") as f:
    f.write("GEO SENSE PHASE III — EXERCISE 4\n")
    f.write("Street-Level Semantic Segmentation and Walkability Screening\n\n")

    f.write(f"Images processed: {len(summary)}\n")
    f.write("Model: SegFormer-B0 fine-tuned on Cityscapes\n")
    f.write("Inference device: CPU\n\n")

    f.write("MEAN PREDICTED CLASS PIXEL PERCENTAGES\n")
    for col in classes:
        if col in summary.columns:
            f.write(f"{col}: {summary[col].mean():.3f}%\n")

    f.write("\nSIDEWALK SCREENING\n")
    f.write(f"Images flagged for review: {int(screening['sidewalk_review_flag'].sum())} of {len(screening)}\n")
    f.write("Review threshold: predicted sidewalk coverage below 1%.\n")
    f.write("This is a review trigger, not a validated safety threshold.\n")

    f.write("\nLIMITATIONS\n")
    f.write("- Only five street images were processed.\n")
    f.write("- Predictions have not been validated against manually labelled ground truth.\n")
    f.write("- A low sidewalk prediction may reflect model error or image viewpoint.\n")
    f.write("- Pixel percentages do not measure sidewalk width, continuity, or accessibility.\n")
    f.write("- No definitive pedestrian safety or walkability conclusion can be drawn.\n")

print("Report saved:", report_path)
print("Images processed:", len(summary))
print("Images flagged for sidewalk review:", int(screening["sidewalk_review_flag"].sum()))
