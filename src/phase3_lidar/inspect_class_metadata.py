
from pathlib import Path
import laspy

ROOT = Path(__file__).resolve().parents[2]
INPUT = ROOT / "data" / "lidar" / "raw" / "autzen-classified.copc.laz"

with laspy.open(INPUT) as reader:
    header = reader.header

    print("LIDAR CLASSIFICATION METADATA")
    print("-----------------------------")
    print(f"File: {INPUT.name}")
    print(f"LAS version: {header.version}")
    print(f"Point format: {header.point_format.id}")
    print(f"Point count: {header.point_count:,}")

    print("\nClassification lookup entries:")
    try:
        vlrs = list(header.vlrs)
        found = False

        for vlr in vlrs:
            description = str(getattr(vlr, "description", ""))
            user_id = str(getattr(vlr, "user_id", ""))

            if "class" in description.lower() or "class" in user_id.lower():
                print(f"\nUser ID: {user_id}")
                print(f"Record ID: {getattr(vlr, 'record_id', 'unknown')}")
                print(f"Description: {description}")
                print(vlr)
                found = True

        if not found:
            print("No clearly identified classification lookup found in VLRs.")

    except Exception as exc:
        print(f"Could not inspect VLR metadata: {exc}")

    print("\nNote: standard LAS class codes do not necessarily explain")
    print("project-specific codes. Source metadata may be needed.")
