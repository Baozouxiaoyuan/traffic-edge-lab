"""Download the official vehicle detectors for a fresh checkout."""
from pathlib import Path
import os


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    os.chdir(root)
    os.environ.setdefault("YOLO_CONFIG_DIR", str(root / ".ultralytics-config"))
    from ultralytics import YOLO

    for name in ("yolov8n.pt", "yolo11n.pt"):
        YOLO(name)
        print(f"Ready: {root / name}")


if __name__ == "__main__":
    main()
