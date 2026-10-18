import laspy
import numpy as np

path = r"data\lidar\raw\autzen-classified.copc.laz"

print("Opening Autzen LiDAR point cloud...")
cloud = laspy.read(path)

classes, counts = np.unique(
    cloud.classification,
    return_counts=True
)

total = len(cloud.classification)

print("\nTotal points:", f"{total:,}")
print("\nLiDAR classification summary")
print("-" * 45)

for cls, count in zip(classes, counts):
    percentage = (count / total) * 100
    print(
        f"Class {int(cls)}: "
        f"{int(count):,} points "
        f"({percentage:.2f}%)"
    )

print("-" * 45)
print("Classification counting completed.")