from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def _writer(output: Path, width: int, height: int, fps: int) -> cv2.VideoWriter:
    output.parent.mkdir(parents=True, exist_ok=True)
    codec = "VP80" if output.suffix.lower() == ".webm" else "mp4v"
    writer = cv2.VideoWriter(str(output), cv2.VideoWriter_fourcc(*codec), fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Could not create video: {output}")
    return writer


def _base_frame(width: int, height: int, title: str) -> np.ndarray:
    frame = np.full((height, width, 3), (55, 58, 62), dtype=np.uint8)
    cv2.putText(frame, title, (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.68, (240, 240, 240), 2)
    return frame


def _car(frame: np.ndarray, x: int, y: int, colour: tuple[int, int, int]) -> None:
    cv2.rectangle(frame, (x, y), (x + 58, y + 43), colour, -1)
    cv2.rectangle(frame, (x + 35, y + 7), (x + 50, y + 22), (220, 220, 220), -1)


def generate_demo_video(output: Path, width: int = 640, height: int = 360, frames: int = 110, fps: int = 20) -> None:
    writer = _writer(output, width, height, fps)

    for frame_index in range(frames):
        frame = np.full((height, width, 3), (58, 58, 58), dtype=np.uint8)
        cv2.line(frame, (0, 120), (width, 120), (220, 220, 220), 3)
        cv2.line(frame, (0, 240), (width, 240), (220, 220, 220), 3)
        cv2.putText(frame, "Synthetic fixed-camera test", (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (240, 240, 240), 2)

        violating_x = 20 + frame_index * 5
        cv2.rectangle(frame, (violating_x, 155), (violating_x + 58, 198), (255, 70, 20), -1)
        cv2.rectangle(frame, (violating_x + 35, 162), (violating_x + 50, 177), (255, 220, 180), -1)

        safe_x = 20 + frame_index * 3
        cv2.rectangle(frame, (safe_x, 285), (safe_x + 58, 328), (30, 30, 235), -1)
        cv2.rectangle(frame, (safe_x + 35, 292), (safe_x + 50, 307), (180, 180, 255), -1)
        writer.write(frame)
    writer.release()


def generate_builtin_examples(output_dir: Path, width: int = 640, height: int = 360, frames: int = 100, fps: int = 20) -> None:
    """Create deterministic videos that exercise one rule at a time."""
    output_dir.mkdir(parents=True, exist_ok=True)

    writer = _writer(output_dir / "red_light.webm", width, height, fps)
    for index in range(frames):
        frame = _base_frame(width, height, "Built-in example: RED-LIGHT RUNNING")
        cv2.line(frame, (224, 180), (416, 180), (240, 240, 240), 4)
        cv2.rectangle(frame, (300, 18), (340, 62), (25, 25, 25), -1)
        cv2.circle(frame, (320, 38), 10, (0, 0, 255), -1)
        _car(frame, 291, 300 - index * 3, (255, 70, 20))
        _car(frame, 20 + index * 4, 95, (30, 30, 235))
        writer.write(frame)
    writer.release()

    writer = _writer(output_dir / "wrong_way.webm", width, height, fps)
    for index in range(frames):
        frame = _base_frame(width, height, "Built-in example: WRONG-WAY DRIVING")
        cv2.line(frame, (0, 125), (width, 125), (220, 220, 220), 3)
        cv2.line(frame, (0, 245), (width, 245), (220, 220, 220), 3)
        _car(frame, 550 - index * 5, 165, (255, 70, 20))
        _car(frame, 20 + index * 4, 270, (30, 30, 235))
        writer.write(frame)
    writer.release()

    writer = _writer(output_dir / "restricted_zone.webm", width, height, fps)
    for index in range(frames):
        frame = _base_frame(width, height, "Built-in example: RESTRICTED-ZONE ENTRY")
        cv2.line(frame, (416, 80), (416, 300), (180, 180, 180), 4)
        cv2.putText(frame, "RESTRICTED", (430, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (210, 210, 210), 2)
        _car(frame, 20 + index * 5, 165, (255, 70, 20))
        _car(frame, 30 + index * 3, 290, (30, 30, 235))
        writer.write(frame)
    writer.release()
