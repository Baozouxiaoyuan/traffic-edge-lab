from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
CATALOG = ROOT / "data" / "evaluation_candidates.json"
OUTPUT = ROOT / "runs" / "ground-truth-review"


def frame_at(capture: cv2.VideoCapture, seconds: float) -> np.ndarray:
    capture.set(cv2.CAP_PROP_POS_MSEC, seconds * 1000)
    ok, frame = capture.read()
    if not ok:
        raise RuntimeError(f"Could not read frame at {seconds:.3f} s")
    return frame


def sheet(video: Path, times: list[float], destination: Path, title: str) -> None:
    capture = cv2.VideoCapture(str(video))
    tiles = []
    for seconds in times:
        frame = frame_at(capture, seconds)
        height, width = frame.shape[:2]
        scale = min(400 / width, 225 / height)
        tile = cv2.resize(frame, (round(width * scale), round(height * scale)))
        canvas = np.full((255, 420, 3), 245, dtype=np.uint8)
        x = (420 - tile.shape[1]) // 2
        canvas[20 : 20 + tile.shape[0], x : x + tile.shape[1]] = tile
        cv2.putText(canvas, f"{seconds:.3f} s", (12, 247), cv2.FONT_HERSHEY_SIMPLEX, .55, (10, 10, 10), 1, cv2.LINE_AA)
        tiles.append(canvas)
    capture.release()
    rows = []
    for index in range(0, len(tiles), 3):
        row = tiles[index:index + 3]
        while len(row) < 3:
            row.append(np.full_like(tiles[0], 255))
        rows.append(np.hstack(row))
    body = np.vstack(rows)
    header = np.full((55, body.shape[1], 3), 255, dtype=np.uint8)
    cv2.putText(header, title, (16, 35), cv2.FONT_HERSHEY_SIMPLEX, .75, (0, 0, 0), 2, cv2.LINE_AA)
    destination.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(destination), np.vstack([header, body]))


def main() -> None:
    catalog = json.loads(CATALOG.read_text(encoding="utf-8"))
    for item in catalog["candidates"]:
        video = ROOT / item["path"]
        capture = cv2.VideoCapture(str(video))
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        capture.release()
        duration = frames / fps
        hint = float(item["review_time_seconds"])
        last_frame_time = max(0, duration - 1 / fps)
        overview = [last_frame_time * index / 11 for index in range(12)]
        detail = [max(0, min(duration - 1 / fps, hint + offset)) for offset in np.arange(-1.0, 1.01, .2)]
        stem = item["id"]
        sheet(video, overview, OUTPUT / f"{stem}-overview.jpg", f"{stem} full original-video overview")
        sheet(video, detail, OUTPUT / f"{stem}-detail.jpg", f"{stem} blind review around discovery time")
        print(f"{stem}: {width}x{height}, {fps:.3f} FPS, {frames} frames, {duration:.3f} s")


if __name__ == "__main__":
    main()
