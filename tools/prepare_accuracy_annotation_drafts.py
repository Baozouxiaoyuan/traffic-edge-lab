from __future__ import annotations

import base64
import hashlib
import json
from datetime import datetime
from pathlib import Path

import cv2

from traffic_edge.web_ui import detect_vehicle_proposals


ROOT = Path(__file__).resolve().parents[1]
CATALOG = ROOT / "data" / "evaluation_candidates.json"
DRAFT_DIR = ROOT / "data" / "annotation_drafts"
REVIEW_PRESETS = {
    "tar-accidentbench-b06-red-light": {
        "time": 4.30,
        "bbox": [0.0, 0.50, 0.20, 0.76],
        "vehicle_class": "car",
    },
    "tar-accidentbench-sa-ueha-red-light": {
        "time": 4.60,
        "bbox": [0.88, 0.54, 1.0, 0.79],
        "vehicle_class": "car",
    },
}


def source_identity(video: Path) -> tuple[str, str]:
    digest = hashlib.sha256(video.read_bytes()).hexdigest()
    safe_stem = "".join(character if character.isalnum() or character in "_-" else "-" for character in video.stem).strip("-")
    return digest, f"{safe_stem}-{digest[:12]}"


def frame_data_url(video: Path, seconds: float) -> tuple[str, float, int, int]:
    capture = cv2.VideoCapture(str(video))
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    capture.set(cv2.CAP_PROP_POS_MSEC, seconds * 1000)
    ok, frame = capture.read()
    capture.release()
    if not ok:
        raise RuntimeError(f"Could not read {video.name} at {seconds:.3f} s")
    ok, encoded = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 92])
    if not ok:
        raise RuntimeError("Could not encode review frame")
    return "data:image/jpeg;base64," + base64.b64encode(encoded).decode("ascii"), fps, frame.shape[1], frame.shape[0]


def best_proposal(proposals: list[dict], item_id: str) -> dict | None:
    if not proposals:
        return None
    # Target vehicles enter from the image border in all three accepted clips.
    if item_id.endswith("b06-red-light"):
        return min(proposals, key=lambda p: (p["bbox_normalized"][0], -p["bbox_normalized"][3]))
    return max(proposals, key=lambda p: (p["bbox_normalized"][0], p["bbox_normalized"][3]))


def main() -> None:
    catalog = json.loads(CATALOG.read_text(encoding="utf-8"))
    DRAFT_DIR.mkdir(parents=True, exist_ok=True)
    for item in catalog["candidates"]:
        if item["id"].endswith("b016-red-light"):
            continue
        video = ROOT / item["path"]
        source_hash, identity = source_identity(video)
        data_url, fps, width, height = frame_data_url(video, float(item["review_time_seconds"]))
        proposals = detect_vehicle_proposals(data_url, "yolov8n", 0.05)
        target = best_proposal(proposals, item["id"])
        label = None
        preset = REVIEW_PRESETS.get(item["id"])
        if preset:
            bbox = preset["bbox"]
            label = {
                "timestamp_seconds": preset["time"],
                "lane_id": "lane-1",
                "lane_name": "Lane 1",
                "candidate_event": "candidate_red_signal_violation",
                "status": "uncertain",
                "vehicle_class": preset["vehicle_class"],
                "selection_method": "visual_preannotation_requires_human_confirmation",
                "vehicle_bbox_normalized": bbox,
                "vehicle_point_normalized": [round((bbox[0] + bbox[2]) / 2, 6), bbox[3]],
                "description": "PRE-ANNOTATION ONLY: verify the exact vehicle and stop-line crossing on the original video",
            }
        capture = cv2.VideoCapture(str(video))
        frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        capture.release()
        duration = frames / fps
        draft = {
            "schema_version": 2,
            "video_name": f"Evaluation video: {item['id']}",
            "video_path": item["path"],
            "source_sha256": source_hash,
            "reviewer": "",
            "mode": "ground_truth",
            "labels": [label] if label else [],
            "reviewed_intervals": [],
            "review_coverage_percent": 0.0,
            "duration_seconds": round(duration, 6),
            "current_time_seconds": max(0.0, float(item["review_time_seconds"]) - 1.0),
            "updated_at": datetime.now().isoformat(timespec="seconds"),
            "review_complete": False,
            "preannotation": {
                "status": "uncertain_requires_human_confirmation",
                "source": "blind original-video review hint plus vehicle proposal",
                "frame_size": [width, height],
                "proposal_count": len(proposals),
            },
        }
        destination = DRAFT_DIR / f"{identity}.draft.json"
        destination.write_text(json.dumps(draft, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"{item['id']}: {len(proposals)} proposals; draft={destination.name}")


if __name__ == "__main__":
    main()
