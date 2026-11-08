from __future__ import annotations

import argparse
import base64
import binascii
import hashlib
import json
import math
import mimetypes
import os
import re
import subprocess
import sys
import threading
import time
import uuid
import webbrowser
from datetime import datetime
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, quote, urlparse

import cv2
import numpy as np

from .pipeline import run_pipeline
from .geometry import validate_polygon
from .lane_assist import suggest_lane_corridor
from .forward_lane import suggest_forward_lane
from .intersection_assist import analyze_intersection_video
from .road_understanding import assess_polygon_road_overlap, road_overlay_data_url, segment_road
from .evaluation import evaluate_events, read_jsonl
from .campaign import validate_campaign
from .storage import atomic_write_json
from .scope import ACTIVE_RULES, ACTIVE_SCENE_TYPES, validate_active_rules
from .compute import compute_status, normalize_device, resolve_device
from . import study
from .study_ui import STUDY_HTML, STUDY_SCRIPT, STUDY_STYLE


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
UPLOAD_DIR = DATA_DIR / "imported"
CONFIG_DIR = PROJECT_ROOT / "config" / "generated"
CAMERA_PROFILE_DIR = PROJECT_ROOT / "config" / "camera_profiles"
EXAMPLE_SETUP_DIR = PROJECT_ROOT / "config" / "example_setups"
ANNOTATION_DIR = DATA_DIR / "annotations"
ANNOTATION_DRAFT_DIR = DATA_DIR / "annotation_drafts"
ANNOTATION_WRITE_LOCK = threading.Lock()
SAMPLE_VIDEO = DATA_DIR / "zenodo_video3.mp4"
REAL_EXAMPLE_VIDEOS = {
    "red-light-multi": DATA_DIR / "real_examples" / "zenodo_red_light_event.webm",
    "idiap-pedestrian-area": DATA_DIR / "real_examples" / "idiap_pedestrian_entry.webm",
    "zenodo-video4-stress": DATA_DIR / "real_examples" / "zenodo_video4.mp4",
}
COLLISION_EXAMPLES = {
    'rear-end': ('Rear-end collision', DATA_DIR/'real_examples/curated/tu-v28.mp4', 0, 3.4, False),
    'truck-side-impact': ('Truck and car collision', DATA_DIR/'real_examples/curated/tu-v9.mp4', 1.4, 5.4, False),
    'intersection-impact': ('Intersection collision', DATA_DIR/'real_examples/collision_trials/v24.mp4', 1.605, 5.605, False),
    'day-crossing': ('Daytime intersection traffic', DATA_DIR/'real_examples/curated/jackson-day.mp4', 101.4, 105.4, True),
    'two-way-traffic': ('Vehicles passing in opposite directions', DATA_DIR/'real_examples/curated/tu-v12.mp4', 0, 4, True),
    'dusk-crossing': ('Dusk intersection traffic', DATA_DIR/'real_examples/curated/jackson-dusk.mp4', 30, 34, True),
}
MODEL_PATHS = {
    "yolo11n": PROJECT_ROOT / "yolo11n.pt",
    "yolov8n": PROJECT_ROOT / "yolov8n.pt",
}
RELEASE_VERSION = "1.4.0"
CANDIDATE_CATALOG_PATH = DATA_DIR / "evaluation_candidates.json"
CURATED_LIBRARY_PATH = DATA_DIR / "curated_video_library.json"
MINIMUM_REVIEW_COVERAGE = 0.98
_LABEL_MODELS: dict[str, object] = {}
_LABEL_MODEL_LOCK = threading.Lock()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _annotation_identity(video_path: Path) -> tuple[str, str]:
    source_hash = _sha256_file(video_path)
    return source_hash, f"{safe_name(video_path.stem)}-{source_hash[:12]}"


def _draft_review_content(draft: dict) -> dict:
    """Ignore playback position and thumbnails when comparing reviewed content."""
    keys = ('timestamp_seconds', 'lane_id', 'lane_name', 'candidate_event', 'vehicle_class',
            'selection_method', 'vehicle_bbox_normalized', 'vehicle_point_normalized', 'description')
    return {
        'labels': [{**{key: label.get(key) for key in keys},
                    'status': label.get('status', label.get('annotation_status'))}
                   for label in draft.get('labels', [])],
        'reviewer': draft.get('reviewer'), 'source_sha256': draft.get('source_sha256'),
        'duration_seconds': draft.get('duration_seconds'),
    }


def _review_coverage(intervals: list, duration: float) -> tuple[list[list[float]], float]:
    if not math.isfinite(duration) or duration <= 0:
        raise ValueError("A valid video duration is required")
    clean: list[list[float]] = []
    for interval in intervals:
        if not isinstance(interval, list) or len(interval) != 2:
            raise ValueError("Reviewed intervals must contain start and end times")
        start, end = float(interval[0]), float(interval[1])
        if not math.isfinite(start) or not math.isfinite(end):
            raise ValueError("Reviewed interval times must be finite")
        start, end = max(0.0, min(start, end)), min(duration, max(start, end))
        if end > start:
            clean.append([start, end])
    clean.sort()
    merged: list[list[float]] = []
    for start, end in clean:
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    covered = sum(end - start for start, end in merged)
    return merged, min(1.0, covered / duration)


def validate_annotation_quality(annotations: list, duration: float, lanes: list | None = None) -> list[dict]:
    issues: list[dict] = []
    lane_map = {str(lane.get("id")): lane for lane in (lanes or []) if isinstance(lane, dict)}
    event_to_rule = {
        "candidate_red_signal_violation": "red_light",
        "candidate_wrong_way_driving": "wrong_way",
        "candidate_restricted_zone_entry": "restricted_zone",
    }
    normalized: list[tuple[int, dict, list[float]]] = []
    for index, item in enumerate(annotations):
        prefix = f"Label {index + 1}"
        try:
            timestamp = float(item.get("timestamp_seconds", -1))
            bbox = [float(value) for value in item.get("vehicle_bbox_normalized", [])]
        except (TypeError, ValueError):
            issues.append({"severity": "error", "code": "invalid_value", "message": f"{prefix} has invalid numeric data."})
            continue
        if not 0 <= timestamp <= duration:
            issues.append({"severity": "error", "code": "timestamp_out_of_range", "message": f"{prefix} timestamp is outside the video."})
        if len(bbox) != 4 or any(not 0 <= value <= 1 for value in bbox) or (len(bbox) == 4 and (bbox[2] <= bbox[0] or bbox[3] <= bbox[1])):
            issues.append({"severity": "error", "code": "invalid_bbox", "message": f"{prefix} has an invalid vehicle box."})
            continue
        if (bbox[2] - bbox[0]) * (bbox[3] - bbox[1]) < 0.00015:
            issues.append({"severity": "warning", "code": "tiny_bbox", "message": f"{prefix} vehicle box is extremely small; verify the selected vehicle."})
        event = str(item.get("candidate_event", ""))
        if event not in event_to_rule:
            issues.append({"severity": "error", "code": "unknown_rule", "message": f"{prefix} has an unsupported violation type."})
        lane_id = str(item.get("lane_id", ""))
        if lane_map:
            lane = lane_map.get(lane_id)
            if lane is None:
                issues.append({"severity": "error", "code": "unknown_lane", "message": f"{prefix} refers to a lane that is not configured."})
            elif not bool((lane.get("enabled_rules") or {}).get(event_to_rule.get(event, ""))):
                issues.append({"severity": "error", "code": "rule_disabled_for_lane", "message": f"{prefix} uses a rule that is disabled for {lane.get('name', lane_id)}."})
            else:
                lane_name = lane.get("name", lane_id)
                geometry_requirements = {
                    "candidate_red_signal_violation": (
                        ("polygon", 3, "monitored lane"),
                        ("stop_line", 2, "stop line"),
                        ("signal_rectangle", 2, "controlling signal"),
                        ("direction", 2, "legal direction"),
                    ),
                    "candidate_wrong_way_driving": (
                        ("polygon", 3, "monitored lane"),
                        ("direction", 2, "legal direction"),
                    ),
                    "candidate_restricted_zone_entry": (
                        ("restricted_polygon", 3, "restricted zone"),
                    ),
                }
                for field, minimum_points, description in geometry_requirements.get(event, ()):
                    if len(lane.get(field) or []) < minimum_points:
                        issues.append({
                            "severity": "error",
                            "code": f"missing_{field}",
                            "message": f"{prefix} requires a configured {description} for {lane_name}.",
                        })
        if item.get("status") == "uncertain" or item.get("annotation_status") == "uncertain":
            issues.append({"severity": "error", "code": "unresolved_uncertain", "message": f"{prefix} is still uncertain."})
        normalized.append((index, item, bbox))
    for left in range(len(normalized)):
        index_a, item_a, box_a = normalized[left]
        for right in range(left + 1, len(normalized)):
            index_b, item_b, box_b = normalized[right]
            if item_a.get("candidate_event") != item_b.get("candidate_event") or item_a.get("lane_id") != item_b.get("lane_id"):
                continue
            if abs(float(item_a.get("timestamp_seconds", 0)) - float(item_b.get("timestamp_seconds", 0))) > 0.35:
                continue
            x1, y1 = max(box_a[0], box_b[0]), max(box_a[1], box_b[1])
            x2, y2 = min(box_a[2], box_b[2]), min(box_a[3], box_b[3])
            intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
            union = ((box_a[2] - box_a[0]) * (box_a[3] - box_a[1]) +
                     (box_b[2] - box_b[0]) * (box_b[3] - box_b[1]) - intersection)
            if union and intersection / union >= 0.5:
                issues.append({
                    "severity": "error", "code": "probable_duplicate",
                    "message": f"Labels {index_a + 1} and {index_b + 1} probably describe the same event.",
                })
    return issues


def detect_vehicle_proposals(image_data_url: str, model_name: str, confidence: float = 0.15) -> list[dict]:
    if not image_data_url.startswith("data:image/") or "," not in image_data_url:
        raise ValueError("A captured video frame is required")
    model_name = model_name.lower().strip()
    model_path = MODEL_PATHS.get(model_name)
    if model_path is None or not model_path.is_file():
        raise ValueError("A supported local vehicle detector is required")
    raw = base64.b64decode(image_data_url.split(",", 1)[1], validate=True)
    frame = cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), cv2.IMREAD_COLOR)
    if frame is None:
        raise ValueError("The captured frame could not be decoded")
    height, width = frame.shape[:2]
    with _LABEL_MODEL_LOCK:
        model = _LABEL_MODELS.get(model_name)
        if model is None:
            os.environ.setdefault("YOLO_CONFIG_DIR", str(PROJECT_ROOT / ".ultralytics-config"))
            from ultralytics import YOLO
            model = YOLO(str(model_path))
            _LABEL_MODELS[model_name] = model
        result = model.predict(
            frame, conf=max(0.05, min(float(confidence), 0.95)), iou=0.5,
            classes=[2, 5, 7], imgsz=960, agnostic_nms=True, verbose=False,
        )[0]
    if result.boxes is None:
        return []
    proposals = []
    for index, (bbox, score, class_id) in enumerate(zip(
        result.boxes.xyxy.cpu().tolist(),
        result.boxes.conf.cpu().tolist(),
        result.boxes.cls.int().cpu().tolist(),
    )):
        proposals.append({
            "proposal_id": index + 1,
            "bbox_normalized": [
                round(max(0.0, min(1.0, bbox[0] / width)), 6),
                round(max(0.0, min(1.0, bbox[1] / height)), 6),
                round(max(0.0, min(1.0, bbox[2] / width)), 6),
                round(max(0.0, min(1.0, bbox[3] / height)), 6),
            ],
            "class_name": str(result.names[class_id]),
            "confidence": round(float(score), 4),
        })
    return proposals


def load_evaluation_candidates() -> list[dict]:
    candidates = []
    seen = set()
    for catalogue in (CANDIDATE_CATALOG_PATH, CURATED_LIBRARY_PATH):
        if not catalogue.is_file():
            continue
        payload = json.loads(catalogue.read_text(encoding='utf-8'))
        for item in payload.get('candidates', payload.get('videos', [])):
            if item.get('rule') not in ACTIVE_RULES or item.get('scene_type') == 'roundabout':
                continue
            if item.get('status') != 'accepted_for_manual_review' or item.get('id') in seen:
                continue
            if item.get('library_tier') in ('curated', 'supplementary') and not item.get('full_decode_verified'):
                continue
            if item.get('library_tier') in ('curated', 'supplementary') and item.get('scene_type') not in ACTIVE_SCENE_TYPES:
                continue
            path = (PROJECT_ROOT / str(item['path'])).resolve()
            if not is_within(path, DATA_DIR) or not path.is_file():
                continue
            candidates.append({**item, 'resolved_path': path})
            seen.add(item['id'])
    return candidates


def safe_name(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_-]+", "-", value.strip()).strip("-")
    return cleaned or "video"


def register_campaign_case(camera_id: str, job: dict, ground_truth: Path) -> str:
    """Register one finalized review only after the same video has a completed frozen run."""
    if not job or job.get("status") != "complete":
        raise ValueError("Complete an analysis with the frozen camera geometry before campaign registration.")
    video_path = Path(job["video_path"]).resolve()
    ground_truth = ground_truth.resolve()
    if not is_within(video_path, DATA_DIR) or not video_path.is_file():
        raise ValueError("The completed run does not reference a valid campaign video.")
    if not is_within(ground_truth, ANNOTATION_DIR) or not ground_truth.is_file():
        raise ValueError("The finalized ground-truth file is missing.")
    template_path = DATA_DIR / "accuracy_campaign.template.json"
    active_path = DATA_DIR / "accuracy_campaign.active.json"
    source_path = active_path if active_path.is_file() else template_path
    campaign = json.loads(source_path.read_text(encoding="utf-8"))
    source_id = safe_name(camera_id)
    case = next((
        item for item in campaign.get("cases", [])
        if (item.get("source_group_id") or item.get("camera_id")) == source_id
    ), None)
    if case is None:
        raise ValueError(f"No campaign case matches source {source_id}.")
    expected_video = (DATA_DIR / str(case["input_video"])).resolve()
    if expected_video != video_path:
        raise ValueError("The completed run video does not match the selected campaign case.")
    config_path = Path(job["summary"]["outputs"]["configuration"]).resolve()
    if not config_path.is_file() or not is_within(config_path, PROJECT_ROOT):
        raise ValueError("The completed run has no frozen configuration file.")
    case["status"] = "labelled"
    case["config"] = os.path.relpath(config_path, DATA_DIR).replace("\\", "/")
    case["ground_truth"] = os.path.relpath(ground_truth, DATA_DIR).replace("\\", "/")
    case["input_video"] = os.path.relpath(video_path, DATA_DIR).replace("\\", "/")
    active_path.write_text(json.dumps(campaign, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return active_path.resolve().relative_to(PROJECT_ROOT.resolve()).as_posix()


def validate_camera_profile(payload: dict) -> dict:
    """Validate and normalize a reusable fixed-camera calibration profile."""
    if not isinstance(payload, dict):
        raise ValueError("Camera profile must be a JSON object")
    camera_id_raw = str(payload.get("camera_id", "")).strip()
    if not camera_id_raw:
        raise ValueError("Enter a camera profile ID")
    camera_id = safe_name(camera_id_raw).lower()
    lanes = payload.get("lanes")
    if not isinstance(lanes, list) or not lanes:
        raise ValueError("A camera profile requires at least one configured lane")
    signature = payload.get("source_signature") or {}
    width = int(signature.get("width", 0))
    height = int(signature.get("height", 0))
    if width <= 0 or height <= 0:
        raise ValueError("Camera profile requires the source frame width and height")
    aspect_ratio = width / height
    supplied_ratio = float(signature.get("aspect_ratio", aspect_ratio))
    if not math.isfinite(supplied_ratio) or abs(supplied_ratio - aspect_ratio) > 0.01:
        raise ValueError("Camera profile source dimensions and aspect ratio disagree")
    inference = payload.get("inference") or {}
    confidence = float(inference.get("confidence", 0.25))
    image_size = int(inference.get("image_size", 960))
    model_name = str(inference.get("model_name", "yolov8n"))
    tracker_name = str(inference.get("tracker_name", "bytetrack"))
    first = lanes[0]
    if not isinstance(first, dict):
        raise ValueError("Every configured lane must be an object")
    # Reuse the run configuration validator so saved profiles cannot bypass any
    # geometry, rule, detector or tracker checks used by the actual pipeline.
    create_config(
        camera_id=camera_id,
        polygon=first.get("polygon", []),
        confidence=confidence,
        signal_rectangle=first.get("signal_rectangle") or None,
        direction=first.get("direction") or None,
        stop_line=first.get("stop_line") or None,
        restricted_polygon=first.get("restricted_polygon") or None,
        enabled_rules=first.get("enabled_rules", {}),
        image_size=image_size,
        restricted_initial_presence=bool(first.get("restricted_initial_presence", False)),
        lane_assist=first.get("lane_assist"),
        tracker_name=tracker_name,
        lanes=lanes,
        model_name=model_name,
    )
    now = datetime.now().isoformat(timespec="seconds")
    return {
        "schema_version": 1,
        "profile_id": camera_id,
        "camera_id": camera_id,
        "display_name": str(payload.get("display_name") or camera_id_raw).strip()[:80],
        "source_signature": {
            "width": width,
            "height": height,
            "aspect_ratio": round(aspect_ratio, 6),
        },
        "calibration_time_seconds": round(float(payload.get("calibration_time_seconds", 0)), 3),
        "lanes": json.loads(json.dumps(lanes)),
        "inference": {
            "confidence": confidence,
            "image_size": image_size,
            "model_name": model_name,
            "tracker_name": tracker_name,
        },
        "created_at": str(payload.get("created_at") or now),
        "updated_at": now,
    }


def load_camera_profiles() -> list[dict]:
    profiles: list[dict] = []
    if not CAMERA_PROFILE_DIR.is_dir():
        return profiles
    for path in sorted(CAMERA_PROFILE_DIR.glob("*.json")):
        try:
            profile = json.loads(path.read_text(encoding="utf-8"))
            profiles.append({
                "profile_id": profile["profile_id"],
                "camera_id": profile["camera_id"],
                "display_name": profile.get("display_name", profile["camera_id"]),
                "source_signature": profile.get("source_signature", {}),
                "lane_count": len(profile.get("lanes", [])),
                "updated_at": profile.get("updated_at"),
            })
        except (OSError, KeyError, TypeError, json.JSONDecodeError):
            continue
    return profiles


def _normalized_points(points: list[list[float]], expected: int | None = None) -> list[list[float]]:
    if expected is not None and len(points) != expected:
        raise ValueError(f"Expected {expected} points")
    normalized: list[list[float]] = []
    for point in points:
        if len(point) != 2:
            raise ValueError("Each point must contain x and y")
        x, y = float(point[0]), float(point[1])
        if not 0.0 <= x <= 1.0 or not 0.0 <= y <= 1.0:
            raise ValueError("Coordinates must be normalized from 0 to 1")
        normalized.append([round(x, 6), round(y, 6)])
    return normalized


def create_config(
    camera_id: str,
    polygon: list[list[float]],
    confidence: float,
    signal_rectangle: list[list[float]] | None = None,
    direction: list[list[float]] | None = None,
    stop_line: list[list[float]] | None = None,
    restricted_polygon: list[list[float]] | None = None,
    enabled_rules: dict[str, bool] | None = None,
    image_size: int = 960,
    restricted_initial_presence: bool = False,
    lane_assist: dict | None = None,
    tracker_name: str = "bytetrack",
    lanes: list[dict] | None = None,
    model_name: str = "yolov8n",
    device: str = "auto",
) -> dict:
    device = normalize_device(device)
    if device == "cuda":
        resolve_device(device)
    enabled = enabled_rules or {"red_light": True, "wrong_way": False, "restricted_zone": False}
    validate_active_rules(enabled)
    if not any(enabled.get(name, False) for name in ACTIVE_RULES):
        raise ValueError("At least one violation rule must be enabled")
    if len(polygon) < 3:
        if enabled.get("restricted_zone", False) and restricted_polygon is not None and len(restricted_polygon) >= 3:
            polygon = restricted_polygon
        else:
            raise ValueError("The controlled approach requires at least three points")
    if enabled.get("red_light", False) and signal_rectangle is None:
        raise ValueError("Red-light detection requires a traffic-light rectangle")
    if enabled.get("red_light", False) and stop_line is None:
        raise ValueError("Red-light detection requires a stop line")
    if (enabled.get("red_light", False) or enabled.get("wrong_way", False)) and direction is None:
        raise ValueError("Red-light and wrong-way detection require a legal-direction arrow")
    normalized = _normalized_points(polygon)
    validate_polygon(normalized)
    normalized_signal = _normalized_points(signal_rectangle, expected=2) if signal_rectangle else None
    normalized_direction = _normalized_points(direction, expected=2) if direction else None
    if normalized_direction and math.dist(*normalized_direction) < 0.01:
        raise ValueError("Draw a legal-direction arrow with two separate endpoints.")
    normalized_stop_line = _normalized_points(stop_line, expected=2) if stop_line else None
    normalized_restricted = None
    if enabled.get("restricted_zone", False):
        if restricted_polygon is None or len(restricted_polygon) < 3:
            raise ValueError("Restricted-zone detection requires at least three polygon points")
        normalized_restricted = _normalized_points(restricted_polygon)
        validate_polygon(normalized_restricted)
    if not 0.05 <= confidence <= 0.95:
        raise ValueError("YOLO confidence must be between 0.05 and 0.95")
    if image_size not in {640, 960, 1280}:
        raise ValueError("YOLO image size must be 640, 960 or 1280")
    tracker_name = tracker_name.lower().strip()
    if tracker_name not in {"bytetrack", "botsort"}:
        raise ValueError("Tracker must be ByteTrack or BoT-SORT")
    model_name = model_name.lower().strip()
    if model_name not in MODEL_PATHS:
        raise ValueError("Model must be YOLO11n or YOLOv8n")
    config = {
        "camera_id": safe_name(camera_id),
        "enabled_rules": {
            "red_light": bool(enabled.get("red_light", False)),
            "wrong_way": bool(enabled.get("wrong_way", False)),
            "restricted_zone": bool(enabled.get("restricted_zone", False)),
        },
        "forbidden_region": {
            "name": "controlled_approach_zone",
            "normalized_polygon": normalized,
        },
        "rule": {
            "event_type": "candidate_red_signal_violation",
            "min_track_age_frames": 5,
        },
        "yolo": {
            "model": str(MODEL_PATHS[model_name].resolve()),
            "confidence": confidence,
            "iou": 0.5,
            # COCO: car, bus and truck. Motorcycles are intentionally out of scope.
            "vehicle_classes": [2, 5, 7],
            "image_size": image_size,
            "tracker": tracker_name,
            "device": device,
        },
        "evidence": {
            "event_clips": True,
            "clip_pre_seconds": 2.0,
            "clip_post_seconds": 3.0,
            "incident_merge_window_seconds": 2.0,
            "export_bundle": True,
        },
    }
    if lane_assist:
        method = safe_name(str(lane_assist.get("method", "manual")))
        config["lane_assist"] = {"method": method}
        confidence_value = lane_assist.get("confidence")
        if confidence_value is not None:
            confidence_number = float(confidence_value)
            if not math.isfinite(confidence_number) or not 0.0 <= confidence_number <= 1.0:
                raise ValueError("Lane-assist confidence must be between 0 and 1")
            config["lane_assist"]["confidence"] = round(confidence_number, 3)
        boundary_lines = lane_assist.get("boundary_lines", [])
        if boundary_lines:
            if len(boundary_lines) != 2:
                raise ValueError("Lane assist must contain exactly two boundary lines")
            config["lane_assist"]["boundary_lines_normalized"] = [
                _normalized_points([[line[0], line[1]], [line[2], line[3]]], expected=2)
                for line in boundary_lines
            ]
        for key in ("start_width_percent", "end_width_percent"):
            if key in lane_assist:
                width_percent = float(lane_assist[key])
                if not math.isfinite(width_percent) or not 0.0 < width_percent <= 100.0:
                    raise ValueError("Lane corridor widths must be between 0 and 100 percent")
                config["lane_assist"][key] = round(width_percent, 2)
        for key in ("road_model", "road_task", "road_device", "semantic_road_overlap", "road_fraction", "mean_road_confidence"):
            if key in lane_assist:
                config["lane_assist"][key] = lane_assist[key]
    if normalized_stop_line is not None:
        config["stop_line"] = {"normalized_points": normalized_stop_line}
    if normalized_signal is not None:
        config["traffic_signal"] = {
            "normalized_rectangle": normalized_signal,
            "minimum_red_ratio": 0.005,
            "history_frames": 5,
            "minimum_colour_ratio": 0.003,
            "red_dominance_ratio": 1.35,
            "minimum_blob_ratio": 0.001,
        }
    if normalized_direction is not None:
        config["controlled_movement"] = {
            "normalized_direction": normalized_direction,
            "minimum_alignment": 0.75,
            "minimum_motion_pixels": 12.0,
            "minimum_motion_bbox_height_ratio": 0.15,
        }
        config["wrong_way"] = {
            "minimum_inside_seconds": 0.2,
            "reference_frame_height": 720,
            "maximum_alignment": -0.7,
            "minimum_motion_pixels": 12.0,
            "consecutive_frames": 2,
            "minimum_opposite_step_ratio": 0.6,
            "maximum_occlusion_seconds": 0.5,
            # Disabled until cross-camera validation demonstrates that stitching
            # improves recall without merging different vehicles.
            "enable_tracklet_stitching": False,
            "maximum_tracklet_stitch_seconds": 0.8,
            "maximum_stitch_distance_bbox_diagonals": 1.5,
        }
    if normalized_restricted is not None:
        config["restricted_zone"] = {
            "name": "restricted_vehicle_zone",
            "normalized_polygon": normalized_restricted,
            "trigger_on_initial_presence": bool(restricted_initial_presence),
        }
    if lanes is not None:
        if not 1 <= len(lanes) <= 8:
            raise ValueError("Configure between one and eight lanes")
        configured_lanes = []
        lane_ids: set[str] = set()
        for index, lane in enumerate(lanes):
            lane_id = safe_name(str(lane.get("id", f"lane-{index + 1}"))).lower()
            if lane_id in lane_ids:
                raise ValueError("Every lane must have a unique ID")
            lane_ids.add(lane_id)
            lane_name = str(lane.get("name", f"Lane {index + 1}")).strip()[:60] or f"Lane {index + 1}"
            lane_config = create_config(
                camera_id=camera_id,
                polygon=lane.get("polygon", []),
                confidence=confidence,
                signal_rectangle=lane.get("signal_rectangle") or None,
                direction=lane.get("direction") or None,
                stop_line=lane.get("stop_line") or None,
                restricted_polygon=lane.get("restricted_polygon") or None,
                enabled_rules=lane.get("enabled_rules", {}),
                image_size=image_size,
                restricted_initial_presence=bool(lane.get("restricted_initial_presence", False)),
                lane_assist=lane.get("lane_assist"),
                tracker_name=tracker_name,
                model_name=model_name,
            )
            lane_entry = {
                key: value for key, value in lane_config.items()
                if key not in {"camera_id", "yolo", "lanes"}
            }
            lane_entry["id"] = lane_id
            lane_entry["name"] = lane_name
            configured_lanes.append(lane_entry)
        config["lanes"] = configured_lanes
        config["enabled_rules"] = {
            rule: any(lane["enabled_rules"].get(rule, False) for lane in configured_lanes)
            for rule in ACTIVE_RULES
        }
    return config


def is_within(path: Path, parent: Path) -> bool:
    try:
        path.resolve().relative_to(parent.resolve())
        return True
    except ValueError:
        return False


class AppState:
    def __init__(self) -> None:
        self.jobs: dict[str, dict] = {}
        self.lock = threading.Lock()
        self.processing_lock = threading.Lock()

    def new_job(
        self,
        video_path: Path,
        polygon: list[list[float]],
        signal_rectangle: list[list[float]],
        direction: list[list[float]],
        stop_line: list[list[float]],
        restricted_polygon: list[list[float]],
        enabled_rules: dict[str, bool],
        camera_id: str,
        confidence: float,
        image_size: int = 960,
        backend_name: str = "yolo",
        restricted_initial_presence: bool = False,
        lane_assist: dict | None = None,
        tracker_name: str = "bytetrack",
        lanes: list[dict] | None = None,
        model_name: str = "yolov8n",
        device: str = "auto",
    ) -> str:
        config = create_config(
            camera_id,
            polygon,
            confidence,
            signal_rectangle,
            direction,
            stop_line,
            restricted_polygon,
            enabled_rules,
            image_size,
            restricted_initial_presence,
            lane_assist,
            tracker_name,
            lanes,
            model_name,
            device,
        )
        if backend_name == "yolo" and not Path(config["yolo"]["model"]).is_file():
            raise ValueError("Detector weights are missing. Run python tools/prepare_models.py first.")
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        stem = safe_name(video_path.stem)
        job_id = uuid.uuid4().hex[:12]
        CONFIG_DIR.mkdir(parents=True, exist_ok=True)
        config_path = CONFIG_DIR / f"{stem}-{timestamp}-{job_id}.json"
        output_dir = PROJECT_ROOT / "runs" / f"{stem}-{timestamp}-{job_id}"
        config_path.write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8")
        job = {
            "id": job_id,
            "status": "queued",
            "current_frame": 0,
            "total_frames": 0,
            "video_path": str(video_path),
            "output_dir": str(output_dir),
            "summary": None,
            "error": None,
            "backend": backend_name,
        }
        with self.lock:
            self.jobs[job_id] = job
        threading.Thread(
            target=self._run_job,
            args=(job_id, video_path, output_dir, config_path, backend_name),
            daemon=True,
        ).start()
        return job_id

    def _run_job(self, job_id: str, video_path: Path, output_dir: Path, config_path: Path, backend_name: str) -> None:
        # Each run loads its own detector/tracker. Queue runs instead of loading
        # several large models and competing for the same CPU/GPU memory.
        with self.processing_lock:
            self._execute_job(job_id, video_path, output_dir, config_path, backend_name)

    def _execute_job(self, job_id: str, video_path: Path, output_dir: Path, config_path: Path, backend_name: str) -> None:
        with self.lock:
            self.jobs[job_id]["status"] = "running"

        def progress(current: int, total: int) -> None:
            with self.lock:
                self.jobs[job_id]["current_frame"] = current
                self.jobs[job_id]["total_frames"] = total

        try:
            summary = run_pipeline(video_path, output_dir, config_path, backend_name, progress_callback=progress)
            with self.lock:
                self.jobs[job_id]["status"] = "complete"
                self.jobs[job_id]["summary"] = summary
        except Exception as exc:
            with self.lock:
                self.jobs[job_id]["status"] = "error"
                self.jobs[job_id]["error"] = f"{type(exc).__name__}: {exc}"

    def get_job(self, job_id: str) -> dict | None:
        with self.lock:
            job = self.jobs.get(job_id)
            return dict(job) if job else None

    def new_study_job(self, split: str, model_name: str, tracker: str, device: str) -> str:
        cases = study.snapshot_cases(split)
        model = MODEL_PATHS.get(model_name)
        if model is None or not model.is_file():
            raise ValueError("Prepare the selected detector weights before starting a study.")
        if tracker not in {"bytetrack", "botsort"}:
            raise ValueError("Choose ByteTrack or BoT-SORT.")
        device = normalize_device(device)
        resolve_device(device)
        job_id = uuid.uuid4().hex[:12]
        run_name = f"study-{datetime.now():%Y%m%d-%H%M%S}-{job_id}"
        output = PROJECT_ROOT / "runs" / run_name
        with self.lock:
            self.jobs[job_id] = {"id": job_id, "kind": "study", "status": "queued",
                                 "run_name": run_name, "completed_cases": 0, "total_cases": len(cases),
                                 "stage": "Waiting for the processing queue", "error": None}

        def worker() -> None:
            def progress(current: int, total: int, name: str) -> None:
                with self.lock:
                    self.jobs[job_id].update(completed_cases=current, total_cases=total, stage=name)
            with self.processing_lock:
                with self.lock:
                    self.jobs[job_id]["status"] = "running"
                try:
                    study.run_study(cases, output, model=model, tracker=tracker, device=device, progress=progress)
                    with self.lock:
                        self.jobs[job_id]["status"] = "complete"
                except Exception as error:
                    with self.lock:
                        self.jobs[job_id].update(status="error", error=str(error))

        threading.Thread(target=worker, daemon=True).start()
        return job_id

    def new_diagnosis_job(self, run: str, error_id: str, device: str) -> str:
        from . import diagnostics
        directory, _inputs, case, _error = diagnostics._context(run, error_id)
        device = normalize_device(device)
        resolve_device(device)
        job_id = uuid.uuid4().hex[:12]
        output = directory / "diagnostic-replays" / f"{case['id']}-{job_id}"
        with self.lock:
            self.jobs[job_id] = {"id": job_id, "kind": "diagnosis", "status": "queued",
                "stage": "Waiting for the processing queue", "error": None}

        def worker() -> None:
            def progress(current, total):
                with self.lock:
                    self.jobs[job_id]["stage"] = f"Recording decisions: {current}/{total} frames"
            with self.processing_lock:
                with self.lock:
                    self.jobs[job_id]["status"] = "running"
                try:
                    diagnostics.replay_error(run, error_id, output, device, progress)
                    with self.lock:
                        self.jobs[job_id].update(status="complete", stage="Diagnostic replay complete")
                except Exception as error:
                    with self.lock:
                        self.jobs[job_id].update(status="error", error=str(error))
        threading.Thread(target=worker, daemon=True).start()
        return job_id

    def new_collision_job(self, video_path: Path, start: float, end: float, device: str = "cpu") -> str:
        device = normalize_device(device)
        resolve_device(device)
        from .collision_review import model_available, validated_interval
        if not model_available():
            raise ValueError('The local collision-review model has not been prepared.')
        cap = cv2.VideoCapture(str(video_path))
        try:
            if not cap.isOpened():
                raise ValueError('Cannot decode the source video.')
            fps = cap.get(cv2.CAP_PROP_FPS)
            if not math.isfinite(fps) or fps <= 0:
                raise ValueError('Source timing is unavailable.')
            start, end = validated_interval(start, end, cap.get(cv2.CAP_PROP_FRAME_COUNT)/fps)
        finally:
            cap.release()
        job_id = uuid.uuid4().hex[:12]
        output = PROJECT_ROOT/'runs'/f'collision-{datetime.now():%Y%m%d-%H%M%S}-{job_id}'
        with self.lock:
            self.jobs[job_id] = {'id': job_id, 'kind': 'collision_review', 'status': 'queued',
                'video_path': str(video_path), 'output_dir': str(output), 'current_window': 0,
                'total_windows': 0, 'stage': 'Waiting for the processing queue', 'summary': None, 'error': None}
        threading.Thread(target=self._run_collision_job,args=(job_id,video_path,output,start,end,device),daemon=True).start()
        return job_id

    def _run_collision_job(self, job_id: str, path: Path, output: Path, start: float, end: float, device: str = "cpu") -> None:
        # A worker releases the large optional VLM after each review. It shares
        # the existing queue with violation analysis, preventing model overload.
        with self.processing_lock:
            output.mkdir(parents=True,exist_ok=True)
            with self.lock:
                self.jobs[job_id]['status'] = 'running'
            try:
                with (output/'worker.stderr.log').open('w',encoding='utf-8') as errors:
                    process = subprocess.Popen([sys.executable,'-m','traffic_edge.collision_review',
                        '--input',str(path),'--output-dir',str(output),'--start',str(start),'--end',str(end),'--device',device],
                        cwd=str(PROJECT_ROOT),stdout=subprocess.PIPE,stderr=errors,text=True,encoding='utf-8',
                        creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
                    assert process.stdout is not None
                    for line in process.stdout:
                        try:
                            progress=json.loads(line)
                            if not isinstance(progress,dict):continue
                            with self.lock:
                                self.jobs[job_id].update(current_window=int(progress['current_window']),
                                    total_windows=int(progress['total_windows']),stage=str(progress['stage'])[:200])
                        except (ValueError,TypeError,KeyError):
                            continue
                    code=process.wait()
                if code:
                    raise RuntimeError('Collision review failed; see worker.stderr.log in the result folder.')
                summary=json.loads((output/'collision_review.json').read_text(encoding='utf-8'))
                with self.lock:
                    self.jobs[job_id].update(status='complete',summary=summary)
            except Exception as error:
                with self.lock:
                    self.jobs[job_id].update(status='error',error=f'{type(error).__name__}: {error}')


HTML = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Traffic Edge Lab | Traffic Video Workbench</title>
<style>

:root{--bg:#eeefeb;--card:#fff;--ink:#242924;--muted:#697068;--line:#d4d8cf;--accent:#b44c20;--blue:#a8431a;--blue-dark:#783314;--red:#ad3434;--green:#376448;--amber:#86551e;--mono:Consolas,"SFMono-Regular",monospace}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:14px/1.5 "Segoe UI",Arial,sans-serif}button,input,select{font:inherit}button,.button{display:inline-flex;align-items:center;justify-content:center;gap:7px;min-height:35px;padding:7px 12px;border:1px solid #cbd0c5;border-radius:3px;background:#fafbf8;color:#30392d;font-size:12px;font-weight:600;cursor:pointer;text-decoration:none;line-height:1.35}button:hover,.button:hover{background:#e8ebe3;border-color:#a5af9d}button.primary,.button.primary{background:var(--accent);border-color:var(--accent);color:#fff}button.primary:hover{background:#933e1b}button:disabled{opacity:.45;cursor:not-allowed}button.danger{color:#a53732;background:#fff;border-color:#d9bcb8}button.text-button{min-height:0;padding:0;background:transparent;border:0;color:#c1c8ba;text-decoration:underline}button:focus-visible,a:focus-visible,input:focus-visible,select:focus-visible,summary:focus-visible,.button:focus-visible{outline:2px solid #a8431a;outline-offset:3px}input[type=text],input[type=number],select{min-height:35px;border:1px solid #cbd0c5;border-radius:2px;background:#fff;color:var(--ink);padding:6px 9px;max-width:100%;font-size:12px}input[type=checkbox]{accent-color:var(--accent)}input[type=file].hidden-file{position:absolute;width:1px;height:1px;opacity:0;pointer-events:none}.sr-only{position:absolute;width:1px;height:1px;overflow:hidden;clip:rect(0,0,0,0);white-space:nowrap}[hidden]{display:none!important}
.app-header{height:66px;display:flex;align-items:center;gap:26px;background:#242b25;color:#edf0e7;padding:0 32px;border-bottom:1px solid #40483e}.wordmark{display:flex;align-items:center;gap:12px;color:inherit;text-decoration:none;font-size:16px;font-weight:750;letter-spacing:1px}.wordmark svg{height:30px;width:29px;fill:none;stroke:#e7ebdf;stroke-width:2}.wordmark .centre-mark{stroke:#d78d46}.wordmark-light{font-weight:400}.product-type{border-left:1px solid #525a4f;padding-left:25px;font:10px var(--mono);letter-spacing:1.3px;color:#b4beae}.local-status{margin-left:auto;display:flex;align-items:center;gap:8px;font:11px var(--mono);color:#c0c8b9}.local-status i{width:6px;height:6px;background:#b9c69d;border-radius:50%}.workspace-nav{position:sticky;top:0;z-index:30;display:flex;align-items:stretch;padding:0 32px;gap:32px;background:#f8f9f5;border-bottom:1px solid var(--line)}.workspace-nav a{display:flex;align-items:center;gap:9px;position:relative;padding:17px 0 14px;color:#636d5f;text-decoration:none;font-size:12px;font-weight:600;white-space:nowrap;border-bottom:3px solid transparent}.workspace-nav a span{font:10px var(--mono);color:#8b9384}.workspace-nav a[aria-current=page]{color:var(--ink);border-bottom-color:var(--accent)}.workspace-nav a[aria-current=page] span{color:var(--accent)}.workspace-nav a:hover{color:var(--accent)}.session-strip{display:flex;align-items:center;gap:14px;min-height:36px;padding:7px 32px;background:#e7eae1;border-bottom:1px solid var(--line);font:11px var(--mono)}.session-label{color:#747f6b;font-size:9px;letter-spacing:1px}.session-media{margin-left:auto;color:#737d6b;white-space:nowrap}#sessionSource{overflow:hidden;text-overflow:ellipsis;white-space:nowrap}.workbench{padding:26px 32px 76px;max-width:1700px;margin:auto}.workspace-heading{display:flex;justify-content:space-between;gap:24px;align-items:center;margin:0 0 22px}.section-index,.panel-label,.field-label{display:block;font:10px var(--mono);letter-spacing:.8px;color:#66715d}.section-index{color:var(--accent);margin-bottom:4px}.workspace-heading h1{font-size:28px;font-weight:550;letter-spacing:-.8px;line-height:1.2;margin:0 0 7px}.workspace-heading p{font-size:12px;color:var(--muted);margin:0}.workspace-heading>button{flex-shrink:0}.workspace{min-width:0}.workspace>summary.sr-only{display:none}.app-footer{position:fixed;bottom:0;left:0;right:0;z-index:35;display:flex;align-items:center;justify-content:space-between;gap:15px;height:32px;padding:0 32px;border-top:1px solid var(--line);background:#f8f9f5;color:#62705a;font:9px var(--mono);letter-spacing:.4px}.footer-version{color:#899180}#sessionState{max-width:45%;overflow:hidden;white-space:nowrap;text-overflow:ellipsis}
.source-layout{display:grid;grid-template-columns:minmax(0,1fr) 310px;gap:0;border:1px solid #c9cfc2;background:#fff}.source-monitor{min-width:0;background:#202620;display:flex;flex-direction:column}.monitor-label{display:flex;align-items:center;justify-content:space-between;gap:10px;color:#afbba5;background:#293128;padding:10px 14px;font:9px var(--mono);letter-spacing:1.05px}.monitor-label>span+span{color:#819078;font-size:8px}.source-stage{position:relative;aspect-ratio:16/9;max-height:365px;min-height:220px;display:grid;place-items:center;background:#1b211c}.source-stage #preview{width:100%;height:100%;max-height:365px;object-fit:contain;display:none}.has-source .source-stage #preview{display:block}.source-empty{display:flex;flex-direction:column;align-items:center;text-align:center;gap:10px;padding:24px;color:#d4dccb}.source-empty svg{width:66px;height:54px;fill:none;stroke:#68765f;stroke-width:1.3;margin-bottom:7px}.source-empty strong{font-size:16px;font-weight:400}.source-empty>span{color:#899780;font-size:12px}.source-empty .button{margin-top:12px;background:#333d2f;border-color:#56634d;color:#d7dfcf}.has-source .source-empty{display:none}.monitor-footer{padding:8px 14px;display:flex;justify-content:space-between;gap:10px;font:9px var(--mono);color:#9ca78e;border-top:1px solid #353e2f;margin-top:auto}.source-inspector{padding:18px;border-left:1px solid var(--line);font-size:12px;min-width:0;overflow-wrap:anywhere}.source-inspector .panel-label{margin-bottom:12px}.source-controls{display:grid;gap:8px}.source-controls select{width:100%;text-overflow:ellipsis}.source-controls .button{background:#f1f3ec}.selected-source{border-top:1px solid var(--line);margin-top:20px;padding-top:16px}.file-name{display:block;font-weight:600;font-size:13px;overflow-wrap:anywhere;margin:6px 0}#sampleDescription{font-size:11px;line-height:1.6;max-height:84px;overflow:auto}.inspector-section{border-top:1px solid var(--line);margin:12px 0;padding-top:10px}.inspector-section p{color:#68705f;font-size:11px;margin:6px 0}.inspector-actions{display:grid;gap:7px}.recording-library{margin-top:28px}.library-heading{display:flex;align-items:center;justify-content:space-between;gap:18px;border-bottom:1px solid #bfc7b6;padding-bottom:12px}.library-heading h2{margin:0;font-size:19px;font-weight:550;letter-spacing:-.3px}.filter-label{display:flex;align-items:center;gap:10px;font:10px var(--mono);color:#67745b}.case-gallery{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:16px;margin:15px 0}.case-card{display:grid;grid-template-columns:112px minmax(0,1fr);align-items:start;gap:0;border:1px solid #d1d7c9;border-radius:0;background:#f9faf6;padding:0;text-align:left;min-height:108px;overflow:hidden}.case-card:hover{background:#fff;border-color:#9eaa8f}.case-card[aria-pressed=true]{border-color:var(--accent);box-shadow:inset 0 -2px var(--accent)}.case-card img,.case-card video{width:112px;height:100%;min-height:108px;object-fit:contain;background:#202620;display:block;pointer-events:none;margin:0;border:0}.case-body{padding:10px;min-width:0}.case-title{font-size:12px;font-weight:600;line-height:1.4;margin:4px 0}.case-badge{font:8px var(--mono);letter-spacing:.5px;color:#a14b24;display:block}.case-fact{font-size:10px;color:var(--muted);line-height:1.5}.case-fact.evidence-note{display:none}.case-rule{font:9px var(--mono);color:#7f886f;text-transform:uppercase;letter-spacing:.5px}.supporting-library{border-top:1px solid var(--line);padding:13px 0;margin-top:18px}.supporting-library summary{cursor:pointer;font-size:12px;font-weight:600}.supporting-library summary span{font-size:11px;font-weight:400;color:var(--muted);margin-left:12px}.library-footnote,#curatedLibrarySummary{font-size:11px;color:#748067}#curatedLibrarySummary{margin:10px 0}
.calibration-layout{display:grid;grid-template-columns:minmax(0,1fr) 330px;gap:24px;align-items:start}.calibration-main{min-width:0}.canvas-chrome{border:1px solid #bdc5b3;background:#202620}canvas{display:block;max-width:100%;height:auto;cursor:crosshair}#roiCanvas{width:min(100%,calc(clamp(270px,100vh - 335px,650px)*var(--frame-ratio,1.7778)));max-width:100%;height:auto;margin:0 auto}.canvas-legend{display:flex;gap:16px;flex-wrap:wrap;padding:10px 13px;color:#b5c0aa;font:9px var(--mono);background:#293128}.canvas-legend span{display:flex;align-items:center;gap:6px}.canvas-legend i{height:2px;width:12px;background:var(--key)}.geometry-feedback{padding:10px 0;border-bottom:1px solid var(--line)}#modeHelp{margin:3px 0 9px;color:#4d5b44;font-size:12px}#pointInfo{display:block;font:10px/1.7 var(--mono);color:#7b856f}#configurationHealth{font-size:11px;line-height:1.6;margin:5px 0}.geometry-reset{display:flex;gap:7px;flex-wrap:wrap;padding:12px 0}.geometry-reset button{font-size:11px}.profile-disclosure summary{font:11px var(--mono);cursor:pointer;color:#6b765f}.profile-disclosure[open] summary{margin-bottom:12px}.calibration-inspector,.analysis-inspector{background:#f8f9f4;border:1px solid #ced4c4}.inspector-block{padding:16px;border-bottom:1px solid var(--line);min-width:0}.inspector-block:last-child{border-bottom:0}.inspector-block>.panel-label{margin-bottom:12px}.profile-controls{display:grid;grid-template-columns:1fr 1fr;gap:7px}.profile-controls>strong{display:none}.profile-controls select,.profile-controls .profile-status{grid-column:1/-1}.profile-controls select{width:100%}.profile-controls button{font-size:10px;min-width:0;padding:7px 4px}.profile-status{font-size:10px;line-height:1.6;color:#7b866e}.profile-controls #refreshProfilesBtn{grid-column:1/-1;background:transparent;border:0;justify-content:flex-start;padding:0;min-height:22px;color:#77826c}#saveExampleSetupBtn{width:100%;margin-top:8px;font-size:10px}.lane-controls{display:grid;grid-template-columns:1fr 1fr;gap:8px}.lane-controls strong{display:none}.lane-controls select,.lane-controls label{grid-column:1/-1;width:100%;min-width:0}.lane-controls label{display:flex;flex-direction:column;gap:5px;font-size:11px}.lane-controls input{width:100%}.lane-controls button{font-size:11px}.rule-grid{display:grid;gap:0}.rule-card{display:grid;grid-template-columns:18px 1fr;column-gap:7px;padding:9px 0;border-bottom:1px solid #e1e5d9;cursor:pointer}.rule-card:last-child{border-bottom:0}.rule-card input{grid-row:1/3;margin:3px 0}.rule-name{font-size:12px;font-weight:600}.rule-desc{font-size:10px;color:#7c866e}.drawing-tools{display:grid;grid-template-columns:1fr 1fr;gap:7px}.drawing-tools button{font-size:11px;justify-content:flex-start;min-height:39px}.rule-geometry.is-hidden{display:none}.small-note,.hint{color:var(--muted);font-size:11px;line-height:1.6}.advanced-panel{border:1px solid var(--line);padding:13px;background:#f9faf6;margin:12px 0}.advanced-panel summary{font-size:12px;font-weight:600;cursor:pointer}.advanced-panel[open]>summary{margin-bottom:12px}.recognition-panel .row{align-items:flex-start}.row{display:flex;align-items:center;gap:8px;flex-wrap:wrap}.grow{flex:1}.health-good{color:var(--green);font-size:11px}.health-warn{color:var(--amber);font-size:11px}.warning{padding:10px 12px;border-left:2px solid #aa792f;background:#f4eedf;color:#7e5a27;font-size:11px}.setup-time{font:11px var(--mono)}
.analysis-layout{display:grid;grid-template-columns:minmax(0,1fr) 310px;gap:24px}.analysis-main{min-width:0}.run-console{padding:18px;background:#f9faf5;border:1px solid var(--line)}.run-console .panel-label{margin-bottom:14px}.run-console .small-note{margin-bottom:0}.run-console #status{font:11px var(--mono);font-weight:400}.progress-shell{height:5px;flex:1;min-width:100px;background:#dce2d3;overflow:hidden}.progress-bar{height:100%;width:0;background:var(--accent);transition:width .2s}.analysis-empty{min-height:310px;border:1px solid var(--line);border-top:0;display:flex;align-items:center;justify-content:center;flex-direction:column;text-align:center;padding:32px;background:#f4f6ef}.empty-index{font:10px var(--mono);letter-spacing:1.7px;color:#8a957c}.analysis-empty h2{font-size:22px;font-weight:400;margin:18px 0 0}.analysis-empty p{font-size:12px;color:var(--muted);max-width:320px}.analysis-empty button{margin-top:10px}.analysis-inspector .advanced-panel{border:0;padding:0;margin:0;background:none}.analysis-inspector .advanced-panel summary{display:none}.analysis-inspector .settings{display:grid;grid-template-columns:1fr;gap:6px}.settings label{font:10px var(--mono);color:#77826a;margin-top:8px}.settings select,.settings input{width:100%}.analysis-inspector p{font-size:11px;color:var(--muted);line-height:1.9}.review-notice{background:#f6efdf;border:1px solid #d5c39d;margin-top:16px;padding:20px;font-size:12px}.review-notice p{color:#756343}.result-panel{display:none;padding:20px 0}.result-panel.show{display:block}.result-panel h2{font-size:18px;font-weight:500}.metric-grid,.readiness-grid{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));margin:14px 0;border:1px solid var(--line);background:#f9faf6}.metric-card,.readiness-item{padding:16px;border-right:1px solid var(--line)}.metric-card:last-child,.readiness-item:last-child{border-right:0}.metric-value,.readiness-item b{display:block;font:25px var(--mono);color:#3a4930}.metric-label,.readiness-item span{display:block;font-size:10px;color:var(--muted);margin-top:5px}.result-detail{font-size:12px;line-height:1.9;padding:16px 0;border-bottom:1px solid var(--line);color:#5c6852}.metric{color:#446833;font-weight:600}.incident-summary{border-left:2px solid var(--accent);padding:10px 12px;background:#e8ecdf;font-size:12px}.card{border:1px solid var(--line);background:#f9faf6;padding:16px;margin-top:15px}video{max-width:100%;background:#1b211c;border-radius:0!important}#resultVideo{width:100%;margin-top:20px;max-height:520px}
.research-workspace>.hint{max-width:850px}.annotation-toolbar{display:grid;grid-template-columns:auto minmax(140px,260px) auto minmax(120px,240px);gap:10px;align-items:center;padding:14px 0;border-bottom:1px solid var(--line);margin-bottom:15px;font-size:12px}#annotationVideo{width:100%;max-height:430px;border:1px solid #424e39;display:block;margin:16px 0;background:#1b211c}.annotation-status-grid{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));border:1px solid var(--line);margin:14px 0;background:#f9faf6}.annotation-status{padding:12px 16px;border-right:1px solid var(--line)}.annotation-status:last-child{border-right:0}.annotation-status b{display:block;font:20px var(--mono);color:#405333}.annotation-status span{font-size:10px;color:var(--muted)}.annotation-timeline{position:relative;height:28px;border:1px solid #bcc6ae;background:#e0e7d6;overflow:hidden;cursor:pointer;margin:14px 0}.timeline-reviewed{position:absolute;top:0;height:100%;background:#73945b55}.timeline-marker{position:absolute;top:1px;width:3px;height:26px;background:#a83a32}.timeline-marker.uncertain{background:#af7825}.timeline-marker.prediction{background:#427186;height:13px;top:14px}.timeline-playhead{position:absolute;top:0;height:100%;width:2px;background:#26331f;pointer-events:none}.proposal-legend{display:flex;gap:16px;flex-wrap:wrap;font:10px var(--mono);color:var(--muted)}#labelCanvas{width:100%;max-height:none;background:#1b211c;border:1px solid #6f7e5f;margin:14px 0}.label-card{display:grid;grid-template-columns:120px minmax(0,1fr) auto;gap:14px;align-items:center;border-bottom:1px solid var(--line);padding:14px 0}.label-card img{width:120px;height:76px;object-fit:contain;background:#202620}.confirmed{color:var(--green)}.uncertain{color:var(--amber)}.time-code{font:12px var(--mono)}.history-table,.label-table{width:100%;border-collapse:collapse;font-size:12px;margin-top:14px}.history-table th,.history-table td,.label-table th,.label-table td{padding:11px;border-bottom:1px solid var(--line);text-align:left}.history-table th{font:10px var(--mono);background:#e7ebdf;color:#627154}.empty-labels{font-size:12px;color:var(--muted)}#labels-step .row{padding:10px 0}#labels-step input[type=text]{min-width:160px}#readinessStatus{border-left:2px solid #b28c42;padding:12px;background:#f3efdf}
@media(min-width:1500px){.source-layout{grid-template-columns:minmax(0,1fr) 340px}.source-stage{max-height:520px}.source-stage #preview{max-height:520px}.case-gallery{grid-template-columns:repeat(3,minmax(0,1fr))}.case-card{grid-template-columns:145px minmax(0,1fr)}.case-card video,.case-card img{width:145px}}
@media(max-width:1100px){.workbench{padding:22px 22px 65px}.app-header,.workspace-nav,.session-strip{padding-left:22px;padding-right:22px}.workspace-nav{gap:23px}.case-gallery{grid-template-columns:repeat(2,minmax(0,1fr))}.calibration-layout{grid-template-columns:minmax(0,1fr) 290px;gap:16px}.source-layout{grid-template-columns:minmax(0,1fr) 280px}.analysis-layout{grid-template-columns:minmax(0,1fr) 275px;gap:16px}.product-type{display:none}.profile-controls{grid-template-columns:1fr}.profile-controls>*{grid-column:1/-1}.annotation-toolbar{grid-template-columns:auto 1fr auto 1fr}}
@media(max-width:760px){.app-header{height:56px;padding:0 16px}.wordmark{font-size:13px;gap:8px}.local-status{font-size:9px}.workspace-nav{overflow-x:auto;gap:24px;padding:0 16px}.workspace-nav a{padding:13px 0 10px;font-size:11px}.workspace-nav a span{display:none}.session-strip{padding:7px 16px}.session-media{display:none}.workbench{padding:20px 16px 65px}.workspace-heading{align-items:flex-start;gap:12px}.workspace-heading h1{font-size:25px}.workspace-heading>button{font-size:10px;max-width:145px}.source-layout,.calibration-layout,.analysis-layout{grid-template-columns:1fr}.source-stage{min-height:190px}.source-inspector{border-left:0;border-top:1px solid var(--line);padding:16px}.source-controls{grid-template-columns:1fr 1fr}.source-controls select{grid-column:1/-1}.selected-source{margin-top:14px}.inspector-section{display:none}.inspector-actions{grid-template-columns:1fr 1fr}.source-inspector #sampleDescription{max-height:none}.case-gallery{grid-template-columns:1fr}.case-card{grid-template-columns:125px minmax(0,1fr)}.case-card video,.case-card img{width:125px}.calibration-inspector{display:grid;grid-template-columns:1fr 1fr}.calibration-inspector .inspector-block{border-right:1px solid var(--line)}.library-heading{align-items:flex-start}.filter-label{display:flex;align-items:flex-start;flex-direction:column;gap:5px;font-size:9px}.supporting-library summary span{display:block;margin-left:16px;margin-top:5px}.app-footer{padding:0 16px}.app-footer>span:last-child{display:none}.footer-version{display:none}.annotation-toolbar{grid-template-columns:1fr 1fr}.metric-grid,.readiness-grid{grid-template-columns:1fr 1fr}.metric-card,.readiness-item{border-bottom:1px solid var(--line)}.annotation-status{padding:10px 8px}.annotation-status b{font-size:15px}.label-card{grid-template-columns:90px 1fr}.label-card img{width:90px}.label-card>div:last-child{grid-column:1/-1}.drawing-tools{grid-template-columns:1fr}.canvas-legend{gap:10px;font-size:8px}}
@media(prefers-reduced-motion:reduce){*{scroll-behavior:auto!important;transition:none!important}}


.annotation-grid{display:grid;grid-template-columns:minmax(0,1fr) 400px;gap:24px;align-items:start}.annotation-main{min-width:0}.annotation-inspector{border:1px solid var(--line);background:#f9faf6;min-width:0}.annotation-grid #annotationVideo{margin:0;border:0;max-height:420px;aspect-ratio:16/9;object-fit:contain}.annotation-grid #labelCanvas{margin:12px 0;width:100%}.annotation-playback{padding:0 12px 10px;background:#f8f9f4;border:1px solid var(--line);border-top:0}.annotation-fields{display:grid;gap:7px}.annotation-fields label{font:10px var(--mono);color:#738066;margin-top:5px}.annotation-fields input,.annotation-fields select{width:100%;min-width:0!important}.annotation-mark{margin-top:12px}.annotation-save{padding:18px 0;border-top:1px solid var(--line);margin-top:24px}.annotation-events h2{font-size:17px;font-weight:500;margin-top:22px}.legend-help{font-size:10px;color:var(--muted)}.legend-help summary{cursor:pointer}.legend-help .proposal-legend{margin-top:8px;display:grid;gap:4px}.playback-meta{font-size:11px}.playback-meta #sourceFps{font-size:10px}.annotation-mark #markBtn{background:var(--accent);color:white}.annotation-toolbar{margin-top:0;padding-top:0}.annotation-grid .label-card{grid-template-columns:90px minmax(0,1fr)}.annotation-grid .label-card img{width:90px}.annotation-grid .label-card>div:last-child{grid-column:1/-1}
@media(max-width:1100px){.annotation-grid{grid-template-columns:minmax(0,1fr) 325px;gap:16px}}
@media(max-width:760px){.annotation-grid{grid-template-columns:1fr}.annotation-grid #annotationVideo{max-height:300px}}

.app-header{height:56px}.workspace-nav a{padding-top:13px;padding-bottom:11px}.session-strip{min-height:32px}.workbench{padding-top:20px}.workspace-heading{margin-bottom:18px}.workspace-heading .section-index{display:none}.annotation-status-grid{margin:8px 0 14px}.annotation-status{padding:8px 14px}.annotation-status b{font-size:18px}.annotation-grid #annotationVideo{max-height:clamp(220px,calc(100vh - 460px),440px)}
#runHistory{max-width:100%;overflow-x:auto}.result-detail,.label-card{overflow-wrap:anywhere}.history-table th{white-space:nowrap}.source-controls{grid-template-columns:1fr 1fr}.source-controls select{order:0;grid-column:1/-1}.source-controls>.button,.source-controls>button{order:1;font-size:11px;padding-left:5px;padding-right:5px}

.collision-stage{max-height:370px}.collision-stage video{width:100%;height:100%;max-height:370px;object-fit:contain}.collision-stage video:not([src]){display:none}.has-source .collision-empty{display:none}.collision-run{display:flex;gap:12px;flex-wrap:wrap;align-items:center;padding:16px 0;border-bottom:1px solid var(--line)}.collision-run .progress-shell{flex-basis:100%}.collision-times{display:grid;grid-template-columns:1fr 100px;gap:10px;align-items:center;font-size:11px}#collisionExampleSelect{width:100%;margin:10px 0 7px}#collisionSourceName{overflow-wrap:anywhere}.collision-summary{font-size:12px;padding:16px 0;border-bottom:1px solid var(--line)}.collision-window{padding:16px 0;border-bottom:1px solid var(--line)}.collision-window h3{font-size:15px;font-weight:550;margin:0 0 8px}.collision-window img{width:100%;display:block;border:1px solid var(--line);margin-top:12px}.collision-window .row{margin-top:10px}.collision-window.uncertain h3{color:var(--amber)}.collision-window.candidate h3{color:var(--accent)}.collision-window p{font-size:11px;color:var(--muted)}
</style>
</head>
<body>
<header class="app-header"><a aria-label="Traffic Edge Lab home" class="wordmark" href="#video-step"><svg aria-hidden="true" viewbox="0 0 32 34"><path d="M5 3v28M27 3v28"></path><path class="centre-mark" d="M16 3v6m0 5v6m0 5v6"></path></svg><span>TRAFFIC<span class="wordmark-light"> EDGE LAB</span></span></a><span class="product-type">TRAFFIC VIDEO ANALYSIS</span><span class="local-status"><i></i>Local workspace</span></header>
<nav aria-label="Workspaces" class="workspace-nav">
<a data-workspace-link="video-step" href="#video-step"><span>01</span>Recordings</a>
<a data-workspace-link="geometry-step" href="#geometry-step"><span>02</span>Lane setup</a>
<a data-workspace-link="review-step" href="#review-step"><span>03</span>Analysis &amp; evidence</a>
<a data-workspace-link="collision-step" href="#collision-step"><span>04</span>Collision review</a><a data-workspace-link="labels-step" href="#labels-step"><span>05</span>Ground truth</a>
<a data-workspace-link="readiness-step" href="#readiness-step"><span>06</span>Study register</a>
</nav>
<div class="session-strip"><span class="session-label">SOURCE</span><span id="sessionSource">No recording selected</span><span class="session-media" id="sessionMedia">Fixed-camera video</span></div>
<main class="workbench">
<section class="workspace" data-workspace="" id="video-step">
<header class="workspace-heading"><div><span class="section-index">01</span><h1>Recordings</h1><p>Choose a camera recording, inspect the scene, then configure its lanes.</p></div><button class="primary" data-go="geometry-step">Continue to lane setup <span aria-hidden="true">→</span></button></header>
<div class="source-layout">
<div class="source-monitor"><div class="monitor-label"><span>ORIGINAL RECORDING</span><span>UNANNOTATED SOURCE</span></div><div class="source-stage"><video controls="" id="preview" muted=""></video><div class="source-empty" id="sourceEmpty"><svg aria-hidden="true" viewbox="0 0 80 64"><path d="M7 12h50v40H7zM57 25l16-9v32l-16-9M24 24h15v15H24z"></path></svg><strong>No recording loaded</strong><span>Open a video file or select a recording from the library.</span><label class="button" for="file">Open video file</label></div></div><div class="monitor-footer"><span>Playback preserves the original scene.</span><span id="sourceDuration">--:--</span></div></div>
<aside class="source-inspector"><div class="panel-label">SOURCE SELECTION</div><div class="source-controls">
<label class="button" for="file">Open video file</label><input accept="video/*" class="hidden-file" id="file" type="file"/>
<select aria-label="Built-in example" id="sampleSelect"><option value="">Select a task example below</option></select><button id="sampleBtn">Load selected example</button>
</div><div class="selected-source"><span class="field-label">Selected recording</span><span class="file-name" id="fileName">No recording selected</span><p class="hint" id="sampleDescription">Choose an example to see its purpose and validation status.</p></div><div class="inspector-actions"><button id="captureBtn">Use current video frame for setup</button><button id="reviewHintBtn" style="display:none">Go to candidate review hint</button></div></aside>
</div>
<section class="recording-library"><div class="library-heading"><div><span class="section-index">LIBRARY</span><h2>Task recordings</h2></div><label class="filter-label">Filter by task <select id="curatedRuleFilter"><option value="all">All task examples</option><option value="red_light">Red-light evidence and controls</option><option value="wrong_way">Direction evidence and controls</option><option value="restricted_zone">Restricted-area evidence</option><option value="collision">Collision review</option></select></label></div><p class="hint" id="curatedLibrarySummary">Loading task examples...</p><div aria-label="Task examples" class="case-gallery" id="curatedCaseGallery"></div><details class="supporting-library"><summary>Supporting recordings <span>Normal traffic, calibration and tracking stress tests</span></summary><div aria-label="Real-video case library" class="case-gallery" id="caseGallery"></div></details><p class="library-footnote">Labelled replays and recordings awaiting review have separate badges. A collision alone is not evidence of a traffic violation.</p></section>
</section>
<section class="workspace" data-workspace="" hidden="" id="geometry-step">
<header class="workspace-heading"><div><span class="section-index">02</span><h1>Lane setup</h1><p>Assign the physical lane, its legal movement and the rule-defining geometry.</p></div><button class="primary" data-go="review-step">Analysis settings <span aria-hidden="true">→</span></button></header>
<div class="calibration-layout"><div class="calibration-main"><div class="canvas-chrome"><div class="monitor-label"><span>CAMERA CALIBRATION</span><button class="text-button" data-go="video-step">Change source / frame</button></div><canvas height="540" id="roiCanvas" width="960"></canvas><div class="canvas-legend"><span><i style="--key:#e00000"></i>Lane</span><span><i style="--key:#ff00ff"></i>Stop line</span><span><i style="--key:#00a9bb"></i>Signal</span><span><i style="--key:#dc8300"></i>Direction</span><span><i style="--key:#9c27b0"></i>Restricted area</span></div></div><div class="geometry-feedback"><p class="hint" id="modeHelp">Click around one monitored traffic movement. Add as many vertices as needed, then choose the next tool. Use Fine-edit vertices to drag any point.</p><p class="health-warn" id="configurationHealth">Configuration check: draw or restore a lane setup.</p><span class="grow" id="pointInfo">Approach zone: 0 points; stop line: missing; traffic-light box: missing; direction arrow: missing; restricted zone: 0 points.</span></div><div class="geometry-reset"><button id="undoBtn">Undo current mode</button><button disabled="" id="defaultBtn">Restore selected example preset</button><button class="danger" id="clearBtn">Clear all</button></div><div id="roadContextPanel" style="display:none;padding:10px;border:1px solid #cfd9e5;border-radius:9px;background:#f6fbf7;margin-top:8px"><strong>Road-surface sanity check</strong><p class="hint" id="roadModelStatus">Not run.</p><p class="hint">This check does not identify a lane, direction, stop line or controlling signal. It checks only vehicle travel lanes; restricted zones may intentionally cover non-road areas. The red polygon remains authoritative.</p></div><details class="advanced-panel recognition-panel"><summary>Road and signal recognition</summary><p class="hint">Find visible road markings, separate marked lanes and detect small traffic lights. Vehicle paths provide additional direction evidence. Inspect the scene preview before applying proposals.</p><div class="row"><label><input checked="" id="sceneLearned" type="checkbox"/> Recognize road markings</label><label>Traffic observation <select id="sceneWindow"><option value="20">20 seconds</option><option value="60">60 seconds</option></select></label><button id="intersectionAssistBtn">Analyze road + signals</button><span class="hint" id="intersectionAssistStatus">Freeze a clear setup frame first. Recognition runs locally and may take several minutes on CPU.</span></div><div id="intersectionAssistResults"></div></details><details class="advanced-panel"><summary>Optional lane-line model</summary><p class="hint">Draw the orange legal direction first, then let the forward-view model propose lane boundaries. It may abstain on high-angle CCTV and never prevents manual drawing.</p><div class="row"><button id="forwardLaneBtn">Try lane-line model</button><span class="hint" id="forwardLaneStatus">Manual polygon drawing remains available.</span></div></details></div><aside class="calibration-inspector"><div class="inspector-block"><div class="panel-label">ACTIVE LANE</div><div class="lane-controls"><strong>Configured lanes</strong><select aria-label="Active lane" id="laneSelect"></select><label>Lane name <input id="laneName" maxlength="60" type="text" value="Lane 1"/></label><button id="addLaneBtn">Add lane</button><button class="danger" id="deleteLaneBtn">Delete active lane</button></div></div><div class="inspector-block"><div class="panel-label">DRAWING TOOLS</div><div class="drawing-tools"><button class="primary" id="laneMode">Lane boundary</button><button id="zoneMode">Edit vertices</button><button class="rule-geometry" id="lineMode">Stop line</button><button class="rule-geometry" id="signalMode">Traffic signal</button><button class="rule-geometry" id="directionMode">Legal direction</button><button class="rule-geometry" id="restrictedMode">Restricted area</button></div><p class="small-note">Select a tool, then click on the frame. Coordinates belong to the selected lane.</p></div><div class="inspector-block"><div class="panel-label">RULES FOR THIS LANE</div><div aria-label="Enabled rules" class="rule-grid"><label class="rule-card"><input checked="" id="redRule" type="checkbox"/><span class="rule-name">Red-light running</span><span class="rule-desc">Lane + direction + stop line + signal.</span></label><label class="rule-card"><input checked="" id="wrongWayRule" type="checkbox"/><span class="rule-name">Wrong-way driving</span><span class="rule-desc">Lane + legal direction.</span></label><label class="rule-card"><input id="restrictedRule" type="checkbox"/><span class="rule-name">Restricted-zone entry</span><span class="rule-desc">Separate restricted area.</span></label></div></div><details class="inspector-block profile-disclosure"><summary>Saved camera profiles</summary><div class="profile-controls"><strong>Camera profile</strong><select aria-label="Saved camera profile" id="cameraProfileSelect"><option value="">No saved profile selected</option></select><button id="loadProfileBtn">Load profile for this video</button><button class="primary" id="saveProfileBtn">Save current camera profile</button><button id="refreshProfilesBtn">Refresh</button><span class="profile-status" id="cameraProfileStatus">New camera: calibrate once. Same unchanged camera: load its profile after freezing a frame.</span></div><button disabled="" id="saveExampleSetupBtn">Save setup to example</button></details></aside></div>
</section>
<section class="workspace" data-workspace="" hidden="" id="review-step">
<header class="workspace-heading"><div><span class="section-index">03</span><h1>Analysis &amp; evidence</h1><p>Run the configured rules and inspect the vehicle-specific evidence.</p></div></header>
<div class="analysis-layout"><div class="analysis-main"><div class="run-console"><div class="panel-label">PROCESSING</div><div class="row">
<button class="primary" id="runBtn">Start analysis</button><div class="progress-shell"><div class="progress-bar" id="progress"></div></div><strong id="status">Ready</strong>
</div><p class="small-note">Candidate events require human review.</p></div><div class="analysis-empty" id="analysisEmpty"><span class="empty-index">03 / REVIEW</span><h2>No result for this recording</h2><p>Configure the lane and its rules, then start an analysis.</p><button data-go="geometry-step">Return to lane setup</button></div><div class="review-notice" hidden="" id="blindReviewNotice"><strong>Independent annotation is active</strong><p>Model predictions are hidden to preserve blind ground-truth review. Complete the annotation, then select Model error analysis in Ground truth.</p><button data-go="labels-step">Open ground truth</button></div><section class="result-panel" id="result"><h2>Run summary</h2><div id="metrics"></div><div class="row" style="margin-top:12px">
<button class="primary" id="playBtn">Play annotated result</button><button id="reportBtn">Open English run report</button><button id="bundleBtn">Download verified evidence package</button><button id="folderBtn">Open result folder</button><span class="hint" id="resultVideoStatus"></span></div>
<video controls="" id="resultVideo" muted="" style="display:none"></video></section></div><aside class="analysis-inspector"><div class="inspector-block" id="analysis-step"><div class="panel-label">RUN CONFIGURATION</div><details class="advanced-panel" open=""><summary>Detector and tracker settings</summary><div class="settings"><label for="cameraId">Source ID</label><input id="cameraId" type="text" value="source-01"/><label for="confidence">YOLO confidence</label><input id="confidence" max="0.95" min="0.05" step="0.05" type="number" value="0.25"/><label for="imageSize">Inference image size</label><select id="imageSize"><option value="640">640 (faster)</option><option selected="" value="960">960 (balanced)</option><option value="1280">1280 (small objects)</option></select><label for="computeDevice">Compute device</label><select id="computeDevice" class="compute-device"><option value="auto">Automatic (NVIDIA when available)</option><option value="cpu">CPU</option><option value="cuda">NVIDIA CUDA</option></select><p class="small-note compute-status">Checking compute hardware...</p><label for="modelName">Vehicle detector</label><select id="modelName"><option selected="" value="yolov8n">YOLOv8n (validated baseline)</option><option value="yolo11n">YOLO11n (experimental comparison)</option></select><label for="trackerName">Vehicle tracker</label><select id="trackerName"><option selected="" value="bytetrack">ByteTrack (recommended)</option><option value="botsort">BoT-SORT (comparison)</option></select></div><p class="hint">Source ID is optional experiment metadata. Alternative models and trackers are for controlled comparisons.</p></details></div><div class="inspector-block"><div class="panel-label">PROCESSING SCOPE</div><p>Cars, buses and trucks<br/>Fixed, normal-perspective cameras<br/>3 rules with lane-specific geometry</p><p class="small-note">Roundabouts, stop-sign compliance, motorcycles and fisheye video are outside the current evaluation scope.</p></div></aside></div>
</section>
<section class="workspace" data-workspace="" hidden="" id="collision-step">
<header class="workspace-heading"><div><h1>Collision review</h1><p>Experimental scene review for possible two-vehicle impacts in fixed-camera footage.</p></div></header>
<div class="analysis-layout"><div class="analysis-main"><div class="source-monitor"><div class="monitor-label"><span>ORIGINAL RECORDING</span><span>SCENE REVIEW / OFFLINE</span></div><div class="source-stage collision-stage"><video aria-label="Collision review source video" controls="" id="collisionPreview" muted="" preload="metadata"></video><div class="source-empty collision-empty"><strong>Choose a recording</strong><span>Load an example here, or select your own recording.</span></div></div></div><div class="collision-run"><button class="primary" id="collisionRunBtn">Review selected interval</button><span class="hint" id="collisionStatus" role="status">Select a source video first.</span><div class="progress-shell"><div class="progress-bar" id="collisionProgress"></div></div></div><div aria-live="polite" id="collisionResults"></div></div>
<aside class="analysis-inspector"><div class="inspector-block"><div class="panel-label">SOURCE</div><label class="button" for="file">Open video file</label><select aria-label="Collision research example" id="collisionExampleSelect"><option value="">Select a collision example</option></select><button id="collisionExampleBtn">Load example</button><p class="small-note" id="collisionSourceName">No recording selected.</p></div><div class="inspector-block"><label for="collisionDevice">Compute device</label><select id="collisionDevice" class="compute-device"><option value="auto">Automatic (NVIDIA when available)</option><option value="cpu">CPU</option><option value="cuda">NVIDIA CUDA</option></select><p class="small-note compute-status">Checking compute hardware...</p><div class="panel-label">REVIEW INTERVAL</div><div class="collision-times"><label for="collisionStart">Start (seconds)</label><input id="collisionStart" min="0" step="0.1" type="number" value="0"/><label for="collisionEnd">End (seconds)</label><input id="collisionEnd" min="0" step="0.1" type="number" value="8"/></div><p class="small-note">Up to 30 seconds per request. CPU review can take several minutes. No lane drawing is needed.</p></div><div class="inspector-block"><div class="panel-label">READING THE RESULTS</div><p class="small-note">A candidate requires agreement between the full scene and a local interaction view. Disagreement remains uncertain.</p><p class="small-note">Results are review windows, not confirmed accident counts. The model does not establish legal fault or an exact impact frame.</p><span class="small-note" id="collisionModelStatus">Checking local model...</span></div></aside></div></section><details class="workspace research-workspace" data-workspace="" hidden="" id="labels-step"><summary class="sr-only">Vehicle-level annotation</summary><header class="workspace-heading"><div><span class="section-index">05</span><h1>Ground truth</h1><p>Review the original recording and label the vehicle at the rule-defining frame.</p></div></header><div class="annotation-toolbar"><label for="annotationMode"><strong>Workspace mode</strong></label><select id="annotationMode"><option value="ground_truth">Blind ground-truth review</option><option value="error_analysis">Model error analysis</option></select><label for="reviewerName"><strong>Reviewer</strong></label><input id="reviewerName" maxlength="80" placeholder="Reviewer name or ID" type="text"/></div><div class="annotation-status-grid"><div class="annotation-status"><b id="reviewCoverage">0.0%</b><span>Video reviewed</span></div><div class="annotation-status"><b id="confirmedCount">0</b><span>Confirmed events</span></div><div class="annotation-status"><b id="uncertainCount">0</b><span>Uncertain events</span></div><div class="annotation-status"><b id="draftState">Not saved</b><span>Draft state</span></div></div><div class="annotation-grid"><div class="annotation-main"><div class="monitor-label"><span>ORIGINAL RECORDING</span><span>INDEPENDENT REVIEW</span></div><video aria-label="Ground-truth review video" controls="" id="annotationVideo" muted="" preload="metadata"></video><div class="annotation-playback"><div class="annotation-timeline" id="annotationTimeline" title="Click to seek in the original video"><div class="timeline-playhead" id="timelinePlayhead"></div></div><div class="row playback-buttons"><button class="primary" id="reviewPlayPauseBtn">Play review video</button><button id="backFiveBtn">− 5 s</button><button id="backOneBtn">− 1 s</button><button id="backFrameBtn">− 1 frame</button><button id="forwardFrameBtn">+ 1 frame</button><button id="forwardOneBtn">+ 1 s</button><button id="forwardFiveBtn">+ 5 s</button></div><div class="row playback-meta"><label for="playbackRate">Speed</label><select id="playbackRate"><option value="0.25">0.25×</option><option value="0.5">0.5×</option><option selected="" value="1">1×</option><option value="2">2×</option></select><span class="hint" id="sourceFps">Source FPS: loading</span><span class="grow"></span><span>Time <span class="time-code" id="currentTime">00:00.000</span></span><button id="nextGapBtn">Next unreviewed gap</button></div><p class="hint" id="coverageHelp">Play the original video normally to record reviewed intervals. At least 98% measured coverage is required before final completion.</p></div><div class="annotation-events"><h2>Labelled events</h2><div id="labelList"></div></div></div><aside class="annotation-inspector"><div class="inspector-block"><div class="panel-label">SELECT A VEHICLE</div><p class="small-note">Pause at the rule-defining frame. Capture it, then select the vehicle or draw its box.</p><div class="row"><button class="primary" id="captureLabelBtn">Capture frame and find vehicles</button><button id="manualBoxBtn">Draw vehicle box manually</button></div><canvas height="540" id="labelCanvas" width="960"></canvas><p class="hint" id="labelBoxInfo">Capture an event frame. Click a proposed vehicle box, or choose manual drawing and click two opposite corners.</p><details class="legend-help"><summary>Box colours</summary><div class="proposal-legend"><span>Blue: selectable vehicle proposal</span><span>Cyan: selected vehicle</span><span>Manual drawing remains available for missed vehicles</span></div></details></div><div class="inspector-block"><div class="panel-label">EVENT DETAILS</div><div class="annotation-fields"><label for="labelLaneId">Lane</label><select id="labelLaneId"></select><label for="labelEventType">Violation type</label><select id="labelEventType"><option value="candidate_red_signal_violation">Red-light running</option><option value="candidate_wrong_way_driving">Wrong-way driving</option><option value="candidate_restricted_zone_entry">Restricted-zone entry</option></select><label for="labelDescription">Annotation note</label><input class="grow" id="labelDescription" type="text" value="Vehicle crosses the stop line while the controlled signal is red"/></div><div class="row annotation-mark"><button class="primary" id="markBtn">Mark selected vehicle</button><button id="uncertainBtn">Mark selected vehicle uncertain</button><button id="undoLabelBtn">Undo last label</button></div></div></aside></div><div class="annotation-save"><div class="panel-label">REVIEW &amp; SAVE</div><div class="row"><button id="saveDraftBtn">Save draft now</button><button id="qualityCheckBtn">Run quality checks</button><button class="primary" id="saveLabelsBtn">Save completed labels</button><button id="saveNegativeBtn">Save fully reviewed negative</button><button id="exportLabelsBtn">Download JSON copy</button><button class="danger" id="clearLabelsBtn">Clear labels</button></div><div class="hint" id="qualityCheckResult">Quality checks have not run yet.</div><p class="hint" id="labelSaveStatus"></p><p class="small-note">Complete independent ground truth before model error analysis. Uncertain labels remain available for review.</p></div></details><details class="workspace research-workspace" data-workspace="" hidden="" id="readiness-step"><summary class="sr-only">Evaluation register</summary><header class="workspace-heading"><div><span class="section-index">06</span><h1>Study register</h1><p>Review annotation completeness, frozen inputs and experiment history.</p></div></header><div class="readiness-grid" id="readinessMetrics"><div class="readiness-item"><b>—</b><span>Loading study state</span></div></div><p class="hint" id="readinessStatus">Checking completed labels and frozen validation/test videos...</p><div class="row"><button id="refreshReadinessBtn">Refresh readiness</button><span class="hint">Controlled model comparison remains blocked until every registered video has complete ground truth and frozen geometry.</span></div><h3>Recent experiment history</h3><div id="runHistory"><p class="hint">Loading comparable run records...</p></div></details>
</main><footer class="app-footer"><span>TRAFFIC EDGE LAB <span class="footer-version">/ RESEARCH PROTOTYPE</span></span><span aria-live="polite" id="sessionState" role="status">Ready</span><span>Video stays on this computer</span></footer>
</body><script>
const fileInput=document.getElementById('file'), preview=document.getElementById('preview'), canvas=document.getElementById('roiCanvas'), ctx=canvas.getContext('2d'), labelCanvas=document.getElementById('labelCanvas'), labelCtx=labelCanvas.getContext('2d');
const fileName=document.getElementById('fileName'), pointInfo=document.getElementById('pointInfo'), statusEl=document.getElementById('status'), progress=document.getElementById('progress');
const sampleCatalog={
real:{name:'Real fixed-camera development sample: zenodo_video3.mp4',description:'General integration footage without event-level ground truth. Use it to practise geometry and compare rule settings.',url:'/media/sample',path:'data/zenodo_video3.mp4',backend:'yolo',imageSize:960,confidence:.25,camera:'zenodo-development-camera',points:[[.26,.47],[.78,.47],[.91,.96],[.10,.96]],stop:[[.26,.47],[.78,.47]],signal:[[.57,.02],[.62,.10]],direction:[[.78,.86],[.50,.30]],restricted:[[.78,.31],[.98,.30],[.98,.72],[.83,.72]],rules:{red_light:true,wrong_way:true,restricted_zone:true}},
'red-light-multi':{name:'Night intersection: red-light preset',description:'The saved night setup matched the labelled vehicle at 4.933 s. Another red-light candidate has no matching label. The wrong-way and restricted-zone rules produced no candidates.',libraryTier:'task_example',primaryRule:'red_light',url:'/media/example/red-light-multi',path:'data/real_examples/zenodo_red_light_event.webm',backend:'yolo',imageSize:960,confidence:.10,camera:'zenodo-night-red-light-multi-rule',points:[[.18,.57],[.91,.56],[1,1],[0,1]],stop:[[.25,.635],[.91,.61]],signal:[[.515,.035],[.555,.18]],direction:[[.43,.90],[.73,.43]],restricted:[[0,.54],[.13,.53],[.12,.72],[0,.76]],rules:{red_light:true,wrong_way:true,restricted_zone:true}},
'idiap-pedestrian-area':{name:'Pedestrian-area vehicle entry',description:'Officially annotated car-entering-pedestrian-area interval. Initial-presence mode also reviews vehicles first detected after they have entered the purple zone.',url:'/media/example/idiap-pedestrian-area',path:'data/real_examples/idiap_pedestrian_entry.webm',backend:'yolo',imageSize:1280,confidence:.05,camera:'idiap-junction-pedestrian-area',points:[[.43,0],[1,0],[1,.72],[.42,.72]],stop:[],signal:[],direction:[],restricted:[[.58,0],[.995,0],[.995,.34],[.69,.40],[.55,.19]],restrictedInitialPresence:true,rules:{red_light:false,wrong_way:false,restricted_zone:true}},
'zenodo-video4-stress':{name:'Daytime intersection: red-light replay (Zenodo video 4)',description:'The saved setup matched the labelled pickup at 11.100 s. This development clip also exercises tracking through dense traffic and partial occlusion.',libraryTier:'task_example',primaryRule:'red_light',url:'/media/example/zenodo-video4-stress',path:'data/real_examples/zenodo_video4.mp4',backend:'yolo',imageSize:960,confidence:.15,camera:'zenodo-video4-tracking-stress',points:[[.24,.48],[.77,.48],[.92,1],[.08,1]],stop:[],signal:[],direction:[[.55,.92],[.53,.37]],restricted:[],rules:{red_light:false,wrong_way:true,restricted_zone:false}},
};
const caseFacts={
real:{badge:'CALIBRATION',expected:'No event-level ground truth',challenge:'General fixed-camera geometry and mixed traffic'},
'red-light-multi':{badge:'LABELLED DEVELOPMENT REPLAY',expected:'Label matched at 4.933 s. One additional candidate awaits review.',challenge:'Night scene, glare and signal association'},
'idiap-pedestrian-area':{badge:'REVIEW REQUIRED',expected:'Restricted-zone candidates; target needs spatial labelling',challenge:'Clip begins during an event and contains occlusion'},
'zenodo-video4-stress':{badge:'LABELLED DEVELOPMENT REPLAY',expected:'Labelled pickup matched at 11.100 s',challenge:'Dense daytime traffic, overlapping vehicles and ID switches'},
};
const reviewVideo=document.getElementById('annotationVideo');
let selectedFile=null, serverVideoPath=null, selectedBackend='yolo', pendingSample=null, currentSample=null, currentSampleKey=null, objectUrl=null, points=[], stopLinePoints=[], signalPoints=[], directionPoints=[], restrictedPoints=[], restrictedInitialPresence=false, drawMode='lane', drawingArmed=false, geometryDrag=null, frameReady=false, setupFrameImage=null, setupFrameData=null, setupFrameTime=0, jobId=null, labels=[], labelBoxPoints=[], labelFrameReady=false, labelFrameTime=0, labelBaseImage=null, labelFrameDataUrl=null, vehicleProposals=[], selectedProposalIndex=null, manualBoxMode=false, editingLabelIndex=null, reviewedIntervals=[], lastReviewedTime=null, predictionEvents=[], annotationDraftTimer=null, annotationRestoredFor=null, groundTruthFinalized=false, sourceVideoFps=30, autoLaneLines=[], laneAssistMetadata={method:'unconfigured'}, roadMaskImage=null, roadUnderstandingMetadata=null, laneConfigurations=[], activeLaneIndex=0;

function resetSetup(){clearIntersectionAssist();points=[];stopLinePoints=[];signalPoints=[];directionPoints=[];restrictedPoints=[];restrictedInitialPresence=false;autoLaneLines=[];laneAssistMetadata={method:'unconfigured'};setupFrameImage=null;setupFrameData=null;setupFrameTime=0;roadMaskImage=null;roadUnderstandingMetadata=null;const roadPanel=document.getElementById('roadContextPanel');if(roadPanel)roadPanel.style.display='none';const roadStatus=document.getElementById('roadModelStatus');if(roadStatus)roadStatus.textContent='Not run.';const capture=document.getElementById('captureBtn');if(capture)capture.textContent='Use current video frame for setup';laneConfigurations=[];activeLaneIndex=0;renderLaneSelector()}
function currentLaneState(name){return{id:`lane-${activeLaneIndex+1}`,name:name||document.getElementById('laneName').value||`Lane ${activeLaneIndex+1}`,polygon:points.map(p=>[...p]),stop_line:stopLinePoints.map(p=>[...p]),signal_rectangle:signalPoints.map(p=>[...p]),direction:directionPoints.map(p=>[...p]),restricted_polygon:restrictedPoints.map(p=>[...p]),restricted_initial_presence:restrictedInitialPresence,lane_assist:{...laneAssistMetadata,boundary_lines:autoLaneLines.map(line=>[...line])},enabled_rules:{red_light:document.getElementById('redRule').checked,wrong_way:document.getElementById('wrongWayRule').checked,restricted_zone:document.getElementById('restrictedRule').checked}}}
function commitActiveLane(){if(!laneConfigurations.length)return;const previousId=laneConfigurations[activeLaneIndex]?.id||`lane-${activeLaneIndex+1}`;laneConfigurations[activeLaneIndex]={...currentLaneState(),id:previousId};renderLaneSelector()}
function loadLaneState(lane){points=(lane.polygon||[]).map(p=>[...p]);stopLinePoints=(lane.stop_line||[]).map(p=>[...p]);signalPoints=(lane.signal_rectangle||[]).map(p=>[...p]);directionPoints=(lane.direction||[]).map(p=>[...p]);restrictedPoints=(lane.restricted_polygon||[]).map(p=>[...p]);restrictedInitialPresence=Boolean(lane.restricted_initial_presence);laneAssistMetadata=lane.lane_assist||{method:'unconfigured'};autoLaneLines=laneAssistMetadata.boundary_lines||[];const rules=lane.enabled_rules||{};document.getElementById('redRule').checked=Boolean(rules.red_light);document.getElementById('wrongWayRule').checked=Boolean(rules.wrong_way);document.getElementById('restrictedRule').checked=Boolean(rules.restricted_zone);document.getElementById('laneName').value=lane.name||`Lane ${activeLaneIndex+1}`;drawingArmed=false;redraw()}
function renderLaneSelector(){const select=document.getElementById('laneSelect'),labelSelect=document.getElementById('labelLaneId');if(!select)return;const previousLabel=labelSelect?.value;select.innerHTML='';if(labelSelect)labelSelect.innerHTML='';laneConfigurations.forEach((lane,index)=>{const option=document.createElement('option');option.value=String(index);option.textContent=lane.name||`Lane ${index+1}`;option.selected=index===activeLaneIndex;select.appendChild(option);if(labelSelect){const labelOption=document.createElement('option');labelOption.value=lane.id;labelOption.textContent=lane.name||`Lane ${index+1}`;labelSelect.appendChild(labelOption)}});if(labelSelect&&laneConfigurations.some(lane=>lane.id===previousLabel))labelSelect.value=previousLabel;document.getElementById('deleteLaneBtn').disabled=laneConfigurations.length<=1;syncLabelRuleOptions()}
function initialiseSingleLane(name='Lane 1'){laneConfigurations=[currentLaneState(name)];activeLaneIndex=0;renderLaneSelector()}
function setVideoSource(url,name,path=null,backend='yolo'){if(annotationDraftTimer!==null){clearTimeout(annotationDraftTimer);annotationDraftTimer=null;if(serverVideoPath)void saveAnnotationDraft()}annotationSourceVersion++;annotationRevision=0;annotationUpload=null;preview.pause();reviewVideo.pause();preview.src=url;reviewVideo.src=url;preview.load();scheduleSourceFrameCapture();reviewVideo.load();fileName.textContent=name;serverVideoPath=path;selectedBackend=backend;sourceVideoFps=30;document.getElementById('sourceFps').textContent='Source FPS: loading';resetSetup();drawingArmed=false;labels=[];reviewedIntervals=[];lastReviewedTime=null;predictionEvents=[];annotationRestoredFor=null;groundTruthFinalized=false;editingLabelIndex=null;renderLabels();renderAnnotationTimeline();updateAnnotationStatus();updateReviewPlaybackButton();frameReady=false;labelFrameReady=false;labelBoxPoints=[];vehicleProposals=[];document.getElementById('defaultBtn').disabled=!currentSample;drawPlaceholder();drawLabelPlaceholder()}
function applySampleSetup(sample){points=(sample.points||[]).map(p=>[...p]);stopLinePoints=(sample.stop||[]).map(p=>[...p]);signalPoints=(sample.signal||[]).map(p=>[...p]);directionPoints=(sample.direction||[]).map(p=>[...p]);restrictedPoints=(sample.restricted||[]).map(p=>[...p]);restrictedInitialPresence=Boolean(sample.restrictedInitialPresence);autoLaneLines=[];laneAssistMetadata=sample.laneAssist||{method:sample.description?'built_in_preset':'saved_setup'};drawingArmed=false;document.querySelectorAll('#laneMode,#zoneMode,#lineMode,#signalMode,#directionMode,#restrictedMode').forEach(b=>b.classList.remove('primary'));document.getElementById('modeHelp').textContent='Preset loaded. Use Draw monitored lane to replace it with any number of points, or Fine-edit vertices to correct individual points.';document.getElementById('redRule').checked=sample.rules.red_light;document.getElementById('wrongWayRule').checked=sample.rules.wrong_way;document.getElementById('restrictedRule').checked=sample.rules.restricted_zone;document.getElementById('cameraId').value=sample.camera;document.getElementById('imageSize').value=String(sample.imageSize||960);document.getElementById('confidence').value=String(sample.confidence??.25);document.getElementById('modelName').value=sample.modelName||'yolov8n';document.getElementById('trackerName').value=sample.trackerName||'bytetrack';if(sample.lanes?.length){laneConfigurations=sample.lanes.map((lane,index)=>({...lane,id:lane.id||`lane-${index+1}`}));activeLaneIndex=0;loadLaneState(laneConfigurations[0]);renderLaneSelector()}else{initialiseSingleLane('Lane 1')}if(!points.length){drawMode='lane';drawingArmed=true;document.getElementById('laneMode').classList.add('primary');document.getElementById('modeHelp').textContent='Uncalibrated evaluation candidate: click around one monitored movement, then set its legal direction, stop line and controlling signal.'}redraw()}
function drawPlaceholder(){ctx.fillStyle='#24282d';ctx.fillRect(0,0,canvas.width,canvas.height);ctx.fillStyle='#d4dbe2';ctx.font='20px Segoe UI';ctx.textAlign='center';ctx.fillText('Select a video and capture a setup frame',canvas.width/2,canvas.height/2)}
function captureFrame(){
  if(!preview.videoWidth||preview.readyState<2){alert('The video frame is still loading.');return}
  preview.pause();
  clearIntersectionAssist();
  const scale=Math.min(960/preview.videoWidth,540/preview.videoHeight,1),temporary=document.createElement('canvas');
  temporary.width=Math.round(preview.videoWidth*scale);temporary.height=Math.round(preview.videoHeight*scale);
  temporary.getContext('2d').drawImage(preview,0,0,temporary.width,temporary.height);
  canvas.width=temporary.width;canvas.height=temporary.height;
  setupFrameTime=preview.currentTime;setupFrameData=temporary.toDataURL('image/jpeg',.92);
  const frameVersion=annotationSourceVersion,image=new Image();setupFrameImage=image;frameReady=false;roadMaskImage=null;roadUnderstandingMetadata=null;
  document.getElementById('roadContextPanel').style.display='none';
  image.onload=()=>{if(frameVersion!==annotationSourceVersion||setupFrameImage!==image)return;frameReady=true;redraw()};image.src=setupFrameData;
  document.getElementById('captureBtn').textContent=`Setup frame frozen at ${formatTime(setupFrameTime)}`;
  if(!laneConfigurations.length)initialiseSingleLane('Lane 1');
  if(!currentSample&&!points.length&&!directionPoints.length){drawMode='lane';drawingArmed=true;document.querySelectorAll('#laneMode,#zoneMode,#lineMode,#signalMode,#directionMode,#restrictedMode').forEach(b=>b.classList.remove('primary'));document.getElementById('laneMode').classList.add('primary');document.getElementById('modeHelp').textContent='Click around one monitored traffic movement. Add as many vertices as needed.'}
}
function pointerPoint(e){const r=canvas.getBoundingClientRect();return [Math.max(0,Math.min(1,(e.clientX-r.left)/r.width)),Math.max(0,Math.min(1,(e.clientY-r.top)/r.height))]}
function setupFrameDataUrl(){if(!setupFrameData)throw Error('Freeze a setup frame first.');return setupFrameData}
function applyRoadUnderstanding(data){
  roadUnderstandingMetadata=data.road_understanding||null;
  const panel=document.getElementById('roadContextPanel'),info=roadUnderstandingMetadata;
  panel.style.display='block';
  if(data.road_overlay_data_url){const image=new Image();image.onload=()=>{roadMaskImage=image;redraw()};image.src=data.road_overlay_data_url}
  document.getElementById('roadModelStatus').textContent=info?`Single-frame road-surface check: ${(info.road_fraction*100).toFixed(1)}% of the frame was classified as road in ${info.inference_seconds.toFixed(2)} s on ${info.device}. This is a calibration diagnostic, not an accuracy score.`:'Road-surface check complete.';
}
let annotationSourceVersion=0,annotationRevision=0,annotationSaveQueue=Promise.resolve(),annotationUpload=null;
let intersectionAssistData=null,intersectionAssistVersion=0;
function clearIntersectionAssist(){intersectionAssistData=null;intersectionAssistVersion++;document.getElementById('intersectionAssistResults')?.replaceChildren();const status=document.getElementById('intersectionAssistStatus');if(status)status.textContent='Freeze a clear frame, then run recognition for that frame.'}
function sceneLaneKey(index){return intersectionAssistData?.proposal_set_id ? `${intersectionAssistData.proposal_set_id}:${index}` : null}
function refreshSceneProposalButtons(){document.querySelectorAll('.scene-stopline-proposal').forEach(button=>{const key=sceneLaneKey(Number(button.dataset.proposalLane));button.disabled=!key||laneAssistMetadata.source_proposal_key!==key;button.title=button.disabled?'Apply this marked lane to the active lane before using its stop line.':'Use the stop-line candidate belonging to this marked lane.'})}
function applyMarkedLane(index){const lane=intersectionAssistData?.lane_proposals?.[index];if(!lane)return;if((points.length||stopLinePoints.length||signalPoints.length||directionPoints.length)&&!confirm('Replace the current lane geometry? Its stop line, controlling signal and direction will need verification.'))return;points=lane.polygon.map(p=>[...p]);directionPoints=(lane.direction||[]).map(p=>[...p]);stopLinePoints=[];signalPoints=[];restrictedPoints=[];restrictedInitialPresence=false;autoLaneLines=[];for(const curve of lane.boundary_polylines||[]){for(let i=1;i<curve.length;i++)autoLaneLines.push([...curve[i-1],...curve[i]])}laneAssistMetadata={method:lane.method,review_required:true,direction_status:lane.direction_status,source_proposal_key:sceneLaneKey(index),source_proposal_name:lane.name,boundary_lines:autoLaneLines.map(line=>[...line])};drawingArmed=false;setMode('edit');document.getElementById('modeHelp').textContent=directionPoints.length?'Marked lane applied. Verify observed direction, then select the lane stop line and controlling signal.':'Marked lane applied. Travel direction is unknown: draw the legal-direction arrow, then select the stop line and signal.';redraw()}
function applyMarkedStopLine(laneIndex,index){const key=sceneLaneKey(laneIndex);if(!key||laneAssistMetadata.source_proposal_key!==key){alert(`Apply marked lane L${laneIndex+1} to the active lane first. Its stop line cannot be assigned to a different lane.`);return false}const candidate=intersectionAssistData?.lane_proposals?.[laneIndex]?.stop_line_candidates?.[index];if(!candidate)return;if(stopLinePoints.length&&!confirm('Replace the current stop line?'))return;stopLinePoints=candidate.points.map(p=>[...p]);setMode('edit');document.getElementById('modeHelp').textContent='Transverse marking applied. Confirm it is the stop line for the active lane.';redraw()}

function applyMovementProposal(index){const proposal=intersectionAssistData?.movement_proposals?.[index];if(!proposal)return;if((points.length||directionPoints.length)&&!confirm('Replace the current monitored lane and legal direction with this observed vehicle-flow proposal?'))return;points=proposal.polygon.map(p=>[...p]);directionPoints=proposal.direction.map(p=>[...p]);autoLaneLines=[];laneAssistMetadata={method:'video_trajectory_flow',confidence:proposal.confidence,track_count:proposal.track_count,review_required:true};drawingArmed=false;setMode('edit');document.getElementById('modeHelp').textContent=`Applied ${proposal.name} from ${proposal.track_count} observed track(s). Drag any vertex and verify the orange direction before saving.`;redraw()}
function applyStopLineProposal(movementIndex,candidateIndex){const candidate=intersectionAssistData?.movement_proposals?.[movementIndex]?.stop_line_candidates?.[candidateIndex];if(!candidate)return;if(stopLinePoints.length&& !confirm('Replace the current stop line with this painted-line candidate?'))return;stopLinePoints=candidate.points.map(p=>[...p]);drawingArmed=false;setMode('edit');document.getElementById('modeHelp').textContent='Stop-line candidate applied. Verify that it is the physical stop line for this movement, not a crosswalk edge or shadow.';redraw()}
function applySignalProposal(index){const proposal=intersectionAssistData?.traffic_light_proposals?.[index];if(!proposal)return;if(signalPoints.length&& !confirm('Replace the current controlling-signal box with this traffic-light proposal?'))return;signalPoints=proposal.rectangle.map(p=>[...p]);drawingArmed=false;setMode('edit');document.getElementById('modeHelp').textContent='Traffic-light box applied. You must confirm that this specific signal controls the selected movement.';redraw()}
function renderIntersectionAssist(data){
 const host=document.getElementById('intersectionAssistResults');host.replaceChildren();
 const lanes=data.lane_proposals||[],movements=data.movement_proposals||[],signals=data.traffic_light_proposals||[];
 const summary=document.createElement('p');summary.className='hint';summary.textContent=`${lanes.length} marked lane(s), ${signals.length} signal candidate(s), ${movements.length} observed path(s). Observation: ${data.seconds_analyzed.toFixed(1)} s. Processing: ${data.processing_seconds.toFixed(1)} s.`;host.appendChild(summary);
 for(const warning of data.warnings||[]){const p=document.createElement('p');p.className='health-warn';p.textContent=warning;host.appendChild(p)}
 if(data.scene_preview){const image=document.createElement('img');image.src=data.scene_preview;image.alt='Scene proposals: cyan lane outlines, yellow marking segments and green signal boxes';image.style='width:100%;height:auto;border-radius:8px';host.appendChild(image);const legend=document.createElement('p');legend.className='hint';legend.textContent='L = marked lane, S = signal, T = stop-line candidate. Yellow = marking evidence, blue = crosswalk, pink = curb. Paint revealed in other stable frames may lie under a vehicle here. Verify each proposed lane.';host.appendChild(legend)}
 lanes.forEach((lane,index)=>{const card=document.createElement('div');card.className='result-detail';const title=document.createElement('strong');title.textContent=`L${index+1}: ${lane.name} | ${lane.direction?.length?'observed direction available':'direction unknown'}`;card.appendChild(title);const row=document.createElement('div');row.className='row';const button=document.createElement('button');button.textContent='Use marked lane';button.onclick=()=>applyMarkedLane(index);row.appendChild(button);(lane.stop_line_candidates||[]).forEach((line,i)=>{const button=document.createElement('button');button.textContent=`Use T${index+1} stop-line candidate`;button.className='scene-stopline-proposal';button.dataset.proposalLane=String(index);button.onclick=()=>applyMarkedStopLine(index,i);row.appendChild(button)});card.appendChild(row);host.appendChild(card)});
 if(!lanes.length&&data.scene_model){const p=document.createElement('p');p.className='health-warn';p.textContent='No complete marked lane could be formed. Inspect the visible marking evidence, choose a clearer frame, or draw the missing boundaries.';host.appendChild(p)}
 if(signals.length){const card=document.createElement('div');card.className='result-detail';const title=document.createElement('strong');title.textContent='Signal candidates: inspect the detail, then select the light controlling the active lane';card.appendChild(title);const row=document.createElement('div');row.className='row';signals.forEach((signal,index)=>{const item=document.createElement('div');item.style='display:grid;gap:6px;justify-items:start;max-width:240px';const evidence=signal.reference_evidence;if(evidence?.detail_preview){const image=document.createElement('img');image.src=evidence.detail_preview;image.alt=`Signal S${index+1} detail from the frozen setup frame`;image.style='max-height:140px;max-width:180px;image-rendering:pixelated;border:1px solid #ccd3dc;border-radius:4px';item.appendChild(image);const note=document.createElement('span');note.className=evidence.reference_red_state==='red'?'hint':'health-warn';note.textContent=evidence.reference_red_state==='red'?'Red lamp visible in the frozen frame.':evidence.reason==='cropped_at_frame_edge'?'Signal reaches the image edge; state is not confirmed.':evidence.reason==='too_few_pixels'?'Too few source pixels to confirm the light state.':'Red state is not confirmed in this frame.';item.appendChild(note)}const button=document.createElement('button');button.textContent=`Use S${index+1} | ${signal.observations} sampled frame(s)`;button.onclick=()=>applySignalProposal(index);item.appendChild(button);row.appendChild(item)});card.appendChild(row);host.appendChild(card)}
 if(data.scene_model){const p=document.createElement('p');p.className='hint';p.textContent=`Marking evidence combined from ${data.scene_model.temporal_frames_used||1} stable frame(s). ${data.scene_model.occlusion_filtering?'Vehicles and people are excluded; covered road is recovered only where another frame shows it. ':''}Unresolved road regions are left unconfigured.`;host.appendChild(p)}
 if(data.road_marking_proposals?.length){const p=document.createElement('p');p.textContent='Road-marking detections: '+data.road_marking_proposals.map(item=>item.label).join(', ')+'. Verify orientation and lane ownership.';host.appendChild(p)}
 if(movements.length){const details=document.createElement('details');const title=document.createElement('summary');title.textContent='Observed vehicle paths (may cover only part of a physical lane)';details.appendChild(title);movements.forEach((movement,index)=>{const row=document.createElement('div');row.className='row';const text=document.createElement('span');text.textContent=`${movement.name} | ${movement.track_count} track(s)`;row.appendChild(text);const button=document.createElement('button');button.textContent='Use observed path';button.onclick=()=>applyMovementProposal(index);row.appendChild(button);details.appendChild(row)});host.appendChild(details)}
 refreshSceneProposalButtons();
}
async function analyzeIntersection(){
 const button=document.getElementById('intersectionAssistBtn'),status=document.getElementById('intersectionAssistStatus');
 if(!frameReady||!setupFrameImage){alert('Freeze a clear setup frame first.');return}
 if(!serverVideoPath&&!selectedFile){alert('Select a video first.');return}
 const version=intersectionAssistVersion,referenceSeconds=setupFrameTime;
 button.disabled=true;button.textContent='Recognizing road + signals...';status.textContent='Recognizing markings, tracking traffic and inspecting small signals. Local CPU processing can take several minutes.';
 try{let path=serverVideoPath;if(!path){path=await uploadFile(selectedFile);if(version!==intersectionAssistVersion)return;serverVideoPath=path}
 const response=await fetch('/api/analyze-intersection',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({video_path:path,model_name:document.getElementById('modelName').value,confidence:Math.max(.10,Number(document.getElementById('confidence').value)),image_size:Number(document.getElementById('imageSize').value),max_seconds:Number(document.getElementById('sceneWindow').value),reference_seconds:referenceSeconds,learned_markings:document.getElementById('sceneLearned').checked})}),data=await response.json();if(version!==intersectionAssistVersion)return;if(!response.ok)throw Error(data.error||'Scene analysis failed');intersectionAssistData=data;renderIntersectionAssist(data);status.textContent='Scene proposals ready. Inspect the preview, apply each lane and verify its controls.'}
 catch(error){if(version===intersectionAssistVersion)status.textContent='Recognition failed: '+error.message}
 finally{button.disabled=false;button.textContent='Analyze road + signals'}
}
async function forwardFitLane(){if(!frameReady||directionPoints.length!==2){alert('Draw the orange legal-direction arrow before trying the lane-line model.');return}const button=document.getElementById('forwardLaneBtn'),status=document.getElementById('forwardLaneStatus'),previousPoints=points.map(p=>[...p]);button.disabled=true;button.textContent='Running lane-line model...';status.textContent='Detecting lane-line point sequences...';try{const response=await fetch('/api/suggest-forward-lane',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({image_data_url:setupFrameDataUrl(),direction:directionPoints})}),data=await response.json();if(!response.ok)throw Error(data.error||'Lane-line detection failed');points=data.polygon;autoLaneLines=[];for(const lane of data.selected_boundaries||[]){for(let i=1;i<lane.length;i++)autoLaneLines.push([lane[i-1][0],lane[i-1][1],lane[i][0],lane[i][1]])}laneAssistMetadata={method:data.method,model:data.model,confidence:data.confidence,review_required:true,selected_slots:data.selected_slots,boundary_lines:autoLaneLines,inference_seconds:data.inference_seconds,domain_warning:data.domain_warning};drawingArmed=false;setMode('edit');status.textContent=`Model proposal: ${Math.round(data.confidence*100)}% pair-fit confidence, ${data.inference_seconds.toFixed(2)} s on CPU. Drag any vertex before saving.`;document.getElementById('modeHelp').textContent='Model proposal loaded. Inspect the blue boundaries and drag any polygon vertex that needs correction.';redraw()}catch(error){points=previousPoints;status.textContent='Model abstained: '+error.message;redraw()}finally{button.disabled=false;button.textContent='Try lane-line model'}}
function dot(p,colour,label){const x=p[0]*canvas.width,y=p[1]*canvas.height;ctx.beginPath();ctx.arc(x,y,8,0,Math.PI*2);ctx.fillStyle=colour;ctx.fill();ctx.strokeStyle='#111';ctx.lineWidth=2;ctx.stroke();ctx.fillStyle='white';ctx.font='bold 14px Segoe UI';ctx.fillText(label,x+11,y-10)}
function redraw(){
  if(!frameReady||!setupFrameImage){drawPlaceholder();return}
  ctx.drawImage(setupFrameImage,0,0,canvas.width,canvas.height);
  if(roadMaskImage){ctx.save();ctx.globalAlpha=.42;ctx.drawImage(roadMaskImage,0,0,canvas.width,canvas.height);ctx.restore()}
  ctx.save();
  laneConfigurations.forEach((lane,index)=>{if(index===activeLaneIndex||(lane.polygon||[]).length<3)return;const polygon=lane.polygon;ctx.beginPath();ctx.moveTo(polygon[0][0]*canvas.width,polygon[0][1]*canvas.height);for(let i=1;i<polygon.length;i++)ctx.lineTo(polygon[i][0]*canvas.width,polygon[i][1]*canvas.height);ctx.closePath();ctx.setLineDash([8,6]);ctx.strokeStyle='rgba(190,25,25,.75)';ctx.lineWidth=2;ctx.stroke();ctx.setLineDash([]);ctx.fillStyle='rgba(255,255,255,.9)';ctx.font='bold 12px Segoe UI';ctx.fillText(lane.name||`Lane ${index+1}`,polygon[0][0]*canvas.width+6,polygon[0][1]*canvas.height-6)});
  if(points.length){ctx.beginPath();ctx.moveTo(points[0][0]*canvas.width,points[0][1]*canvas.height);for(let i=1;i<points.length;i++)ctx.lineTo(points[i][0]*canvas.width,points[i][1]*canvas.height);if(points.length>=3){ctx.closePath();ctx.fillStyle='rgba(225,24,24,.34)';ctx.fill()}ctx.strokeStyle='#e00000';ctx.lineWidth=4;ctx.stroke();points.forEach((p,i)=>dot(p,'#ffeb3b','A'+(i+1)));if(points.length>=3){const cx=points.reduce((s,p)=>s+p[0],0)/points.length*canvas.width,cy=points.reduce((s,p)=>s+p[1],0)/points.length*canvas.height;ctx.font='bold 14px Segoe UI';const label='MONITORED LANE';const width=ctx.measureText(label).width+14;ctx.fillStyle='rgba(130,0,0,.88)';ctx.fillRect(cx-width/2,cy-22,width,24);ctx.fillStyle='white';ctx.textAlign='center';ctx.fillText(label,cx,cy-5);ctx.textAlign='start'}}
  if(restrictedPoints.length){ctx.beginPath();ctx.moveTo(restrictedPoints[0][0]*canvas.width,restrictedPoints[0][1]*canvas.height);for(let i=1;i<restrictedPoints.length;i++)ctx.lineTo(restrictedPoints[i][0]*canvas.width,restrictedPoints[i][1]*canvas.height);if(restrictedPoints.length>=3){ctx.closePath();ctx.fillStyle='rgba(156,39,176,.24)';ctx.fill()}ctx.strokeStyle='#9c27b0';ctx.lineWidth=3;ctx.stroke();restrictedPoints.forEach((p,i)=>dot(p,'#9c27b0','R'+(i+1)))}
  if(stopLinePoints.length){stopLinePoints.forEach((p,i)=>dot(p,'#ff4dff','L'+(i+1)));if(stopLinePoints.length===2){ctx.strokeStyle='#ff00ff';ctx.lineWidth=5;ctx.beginPath();ctx.moveTo(stopLinePoints[0][0]*canvas.width,stopLinePoints[0][1]*canvas.height);ctx.lineTo(stopLinePoints[1][0]*canvas.width,stopLinePoints[1][1]*canvas.height);ctx.stroke()}}
  if(signalPoints.length){signalPoints.forEach((p,i)=>dot(p,'#00e5ff','S'+(i+1)));if(signalPoints.length===2){const a=signalPoints[0],b=signalPoints[1];ctx.strokeStyle='#00e5ff';ctx.lineWidth=3;ctx.strokeRect(Math.min(a[0],b[0])*canvas.width,Math.min(a[1],b[1])*canvas.height,Math.abs(a[0]-b[0])*canvas.width,Math.abs(a[1]-b[1])*canvas.height)}}
  if(directionPoints.length){directionPoints.forEach((p,i)=>dot(p,'#ff9800','D'+(i+1)));if(directionPoints.length===2){const a=directionPoints[0],b=directionPoints[1],x1=a[0]*canvas.width,y1=a[1]*canvas.height,x2=b[0]*canvas.width,y2=b[1]*canvas.height,angle=Math.atan2(y2-y1,x2-x1);ctx.strokeStyle='#ff9800';ctx.lineWidth=4;ctx.beginPath();ctx.moveTo(x1,y1);ctx.lineTo(x2,y2);ctx.stroke();ctx.beginPath();ctx.moveTo(x2,y2);ctx.lineTo(x2-16*Math.cos(angle-.5),y2-16*Math.sin(angle-.5));ctx.lineTo(x2-16*Math.cos(angle+.5),y2-16*Math.sin(angle+.5));ctx.closePath();ctx.fillStyle='#ff9800';ctx.fill()}}
  ctx.restore();
  pointInfo.textContent=`Monitored lane: ${points.length>=3?'ready':'missing'} · Stop line: ${stopLinePoints.length===2?'ready':'missing'} · Controlling signal: ${signalPoints.length===2?'ready':'missing'} · Legal direction: ${directionPoints.length===2?'ready':'missing'} · Restricted zone: ${restrictedPoints.length>=3?'ready':'not drawn'}`;
}
const baseGeometryRedraw=redraw;
function updateRuleToolVisibility(){const red=document.getElementById('redRule').checked,wrong=document.getElementById('wrongWayRule').checked,restricted=document.getElementById('restrictedRule').checked;document.getElementById('lineMode').classList.toggle('is-hidden',!(red));document.getElementById('signalMode').classList.toggle('is-hidden',!red);document.getElementById('directionMode').classList.toggle('is-hidden',!(red||wrong));document.getElementById('restrictedMode').classList.toggle('is-hidden',!restricted)}
function updateConfigurationHealth(){updateRuleToolVisibility();const health=document.getElementById('configurationHealth'),enabled={red:document.getElementById('redRule').checked,wrong:document.getElementById('wrongWayRule').checked,restricted:document.getElementById('restrictedRule').checked},issues=[];let area=0;if(points.length>=3){for(let i=0;i<points.length;i++){const a=points[i],b=points[(i+1)%points.length];area+=a[0]*b[1]-b[0]*a[1]}area=Math.abs(area)/2;if(area<.01)issues.push('monitored lane is extremely narrow');if(area>.75)issues.push('monitored lane covers most of the frame')}else if(enabled.red||enabled.wrong)issues.push('monitored lane is missing');if((enabled.red||enabled.wrong)&&directionPoints.length!==2)issues.push('legal direction is missing');if((enabled.red)&&stopLinePoints.length!==2)issues.push('stop line is missing');if(enabled.red&&signalPoints.length!==2)issues.push('controlling signal is missing');if(enabled.restricted&&restrictedPoints.length<3)issues.push('restricted zone is missing');health.className=issues.length?'health-warn':'health-good';health.textContent=issues.length?'Configuration check: '+issues.join('; ')+'.':'Configuration check: ready for the selected rules. Confirm the red lane follows only one physical traffic movement.'}
redraw=function(){baseGeometryRedraw();if(frameReady&&autoLaneLines.length){ctx.save();ctx.strokeStyle='#00b7ff';ctx.lineWidth=5;ctx.setLineDash([12,8]);for(const line of autoLaneLines){ctx.beginPath();ctx.moveTo(line[0]*canvas.width,line[1]*canvas.height);ctx.lineTo(line[2]*canvas.width,line[3]*canvas.height);ctx.stroke()}ctx.restore()}updateConfigurationHealth();refreshSceneProposalButtons()};
function clearModeGeometry(mode){if(mode==='lane'){points=[];autoLaneLines=[];laneAssistMetadata={method:'manual_polygon'}}else if(mode==='restricted')restrictedPoints=[];else if(mode==='line')stopLinePoints=[];else if(mode==='signal')signalPoints=[];else if(mode==='direction'){directionPoints=[];autoLaneLines=[];laneAssistMetadata={method:'manual_direction_edit'}}}
function geometryExists(mode){if(mode==='lane')return points.length>0;if(mode==='line')return stopLinePoints.length>0;if(mode==='signal')return signalPoints.length>0;if(mode==='direction')return directionPoints.length>0;if(mode==='restricted')return restrictedPoints.length>0;return false}
function setMode(mode,beginRedraw=false){drawMode=mode;geometryDrag=null;if(mode==='edit'&&points.length<3){alert('Draw at least three monitored-lane points before editing its vertices.');mode='lane';drawMode='lane'}if(beginRedraw){if(geometryExists(mode)&&!confirm('Replace the existing geometry for this tool?'))return;clearModeGeometry(mode);drawingArmed=true;redraw()}else drawingArmed=mode==='edit';document.querySelectorAll('#laneMode,#zoneMode,#lineMode,#signalMode,#directionMode,#restrictedMode').forEach(b=>b.classList.remove('primary'));document.getElementById(mode==='edit'?'zoneMode':mode+'Mode').classList.add('primary');const help={lane:'Click around only this monitored traffic movement. Add as many vertices as needed; choose another tool when finished.',edit:'Drag any yellow vertex directly. Every other point stays fixed.',line:'Click the two endpoints of the physical stop line.',signal:'Click two opposite corners around only the signal controlling this lane.',direction:'Click FROM then TO in the legal vehicle direction.',restricted:'Click around the road area where entry is prohibited.'};document.getElementById('modeHelp').textContent=help[mode]}
canvas.addEventListener('click',e=>{if(drawMode==='edit')return;if(!frameReady)return;if(!drawingArmed){alert('Select the drawing tool for the shape you want to replace.');return}const r=canvas.getBoundingClientRect(),p=[(e.clientX-r.left)/r.width,(e.clientY-r.top)/r.height];if(drawMode==='lane'){points.push(p);autoLaneLines=[];laneAssistMetadata={method:'manual_polygon',review_required:false};document.getElementById('modeHelp').textContent=`Monitored lane: ${points.length} point${points.length===1?'':'s'}. Add more points or choose the next tool when finished.`}else if(drawMode==='restricted')restrictedPoints.push(p);else if(drawMode==='line'){if(stopLinePoints.length===2)stopLinePoints=[];stopLinePoints.push(p)}else if(drawMode==='signal'){if(signalPoints.length===2)signalPoints=[];signalPoints.push(p)}else{if(directionPoints.length===2)directionPoints=[];directionPoints.push(p);if(directionPoints.length===2)drawingArmed=false}redraw()});
canvas.addEventListener('pointerdown',e=>{if(!frameReady||drawMode!=='edit')return;e.preventDefault();const p=pointerPoint(e),rect=canvas.getBoundingClientRect(),nearest=points.map((point,index)=>({index,distance:Math.hypot((point[0]-p[0])*rect.width,(point[1]-p[1])*rect.height)})).sort((a,b)=>a.distance-b.distance)[0];if(!nearest||nearest.distance>34)return;geometryDrag=nearest.index;canvas.setPointerCapture(e.pointerId);canvas.style.cursor='grabbing'});
canvas.addEventListener('pointermove',e=>{if(drawMode==='edit'&&geometryDrag!==null&&canvas.hasPointerCapture(e.pointerId)){points[geometryDrag]=pointerPoint(e);autoLaneLines=[];laneAssistMetadata={...laneAssistMetadata,method:'human_vertex_refinement',review_required:false};redraw()}});
canvas.addEventListener('pointerup',e=>{if(drawMode==='edit'&&geometryDrag!==null){points[geometryDrag]=pointerPoint(e);geometryDrag=null;canvas.style.cursor='crosshair';if(canvas.hasPointerCapture(e.pointerId))canvas.releasePointerCapture(e.pointerId);redraw()}});
fileInput.addEventListener('change',()=>{sourceSelectionVersion++;selectedFile=fileInput.files[0]||null;if(!selectedFile)return;if(objectUrl)URL.revokeObjectURL(objectUrl);objectUrl=URL.createObjectURL(selectedFile);currentSample=null;currentSampleKey=null;pendingSample=null;document.getElementById('saveExampleSetupBtn').disabled=true;document.getElementById('reviewHintBtn').style.display='none';document.getElementById('sampleDescription').textContent='Custom video: draw and validate camera-specific geometry before interpreting candidates.';setVideoSource(objectUrl,selectedFile.name,null,'yolo');document.getElementById('cameraId').value=selectedFile.name.replace(/\.[^.]+$/,'').replace(/[^A-Za-z0-9_-]+/g,'-')||'camera-01'});
function sampleWithSavedSetup(sample,profile){if(!profile)return sample;const settings=profile.inference||{},first=profile.lanes?.[0]||{};return{...sample,camera:profile.camera_id||sample.camera,lanes:profile.lanes,rules:first.enabled_rules||sample.rules,imageSize:settings.image_size||sample.imageSize,confidence:settings.confidence??sample.confidence,modelName:settings.model_name||sample.modelName,trackerName:settings.tracker_name||sample.trackerName,laneAssist:{method:'saved_example_setup'}}}
let sourceSelectionVersion=0;
async function loadSample(key){const selectionVersion=++sourceSelectionVersion;const sample=sampleCatalog[key];if(!sample){alert('Select a recording from the library first.');return}selectedFile=null;if(objectUrl){URL.revokeObjectURL(objectUrl);objectUrl=null}currentSample=sample;currentSampleKey=key;let effective=sample,saved=false;try{const response=await fetch('/api/example-setup?id='+encodeURIComponent(key));if(response.ok){const data=await response.json();effective=sampleWithSavedSetup(sample,data.profile);saved=true}}catch(error){console.warn(error)}if(selectionVersion!==sourceSelectionVersion)return;pendingSample=effective;document.getElementById('sampleSelect').value=key;document.getElementById('saveExampleSetupBtn').disabled=false;document.getElementById('defaultBtn').textContent=saved?'Restore saved example setup':'Restore original example preset';document.getElementById('sampleDescription').textContent=sample.description+(saved?' Saved setup loaded in place of the original preset.':' Original example preset loaded.');const hintButton=document.getElementById('reviewHintBtn');hintButton.style.display=sample.reviewTime===undefined?'none':'inline-block';hintButton.textContent=sample.reviewTime===undefined?'Go to candidate review hint':`Go to review hint (${formatTime(sample.reviewTime)})`;setVideoSource(sample.url,sample.name,sample.path,sample.backend);showWorkspace('video-step');syncWorkspaceSource();preview.scrollIntoView({behavior:'smooth',block:'center'})}
let caseOrder=['red-light-multi','idiap-pedestrian-area','real','zenodo-video4-stress'];
Object.assign(sampleCatalog['red-light-multi'],{libraryTier:'task_example',primaryRule:'red_light'});
Object.assign(sampleCatalog['idiap-pedestrian-area'],{libraryTier:'task_example',primaryRule:'restricted_zone'});
const collisionExamples={};
function renderCaseGallery(){const legacy=document.getElementById('caseGallery'),curated=document.getElementById('curatedCaseGallery'),filter=document.getElementById('curatedRuleFilter')?.value||'all';legacy.replaceChildren();curated.replaceChildren();for(const key of [...caseOrder].sort((a,b)=>Number(sampleCatalog[b]?.primaryRule==='red_light')-Number(sampleCatalog[a]?.primaryRule==='red_light'))){const sample=sampleCatalog[key],facts=caseFacts[key];if(!sample||!facts)continue;const isCurated=sample.libraryTier==='task_example';if(isCurated&&filter!=='all'&&sample.primaryRule!==filter)continue;const card=document.createElement('button');card.type='button';card.className='case-card';card.dataset.sampleKey=key;card.setAttribute('aria-pressed',String(key===currentSampleKey));card.setAttribute('aria-label','Load '+sample.name);const media=sample.posterUrl?`<img loading="lazy" src="${escapeHtml(sample.posterUrl)}" alt="Unannotated source frame">`:`<video muted preload="metadata" src="${escapeHtml(sample.url)}"></video>`;card.innerHTML=media+`<div class="case-body"><span class="case-badge">${escapeHtml(facts.badge)}</span><div class="case-title">${escapeHtml(sample.name)}</div>${sample.mediaFacts?`<div class="case-fact">${escapeHtml(sample.mediaFacts)}</div>`:''}<div class="case-fact evidence-note"><strong>Purpose:</strong> ${escapeHtml(facts.expected)}</div><div class="case-fact evidence-note"><strong>Review:</strong> ${escapeHtml(facts.challenge)}</div>${sample.sourceGroup?`<div class="case-fact evidence-note">Group: ${escapeHtml(sample.sourceGroup)}</div>`:''}</div>`;card.onclick=()=>loadSample(key);const thumbnail=card.querySelector('video');if(thumbnail)thumbnail.onloadedmetadata=()=>{if(Number.isFinite(thumbnail.duration)&&thumbnail.duration>.2)thumbnail.currentTime=Math.min(1,thumbnail.duration/4)};(isCurated?curated:legacy).appendChild(card)}appendCollisionCards(curated,filter);const items=Object.values(sampleCatalog).filter(item=>item.libraryTier==='task_example');document.getElementById('curatedLibrarySummary').textContent=['red_light','wrong_way','restricted_zone'].map(rule=>({count:items.filter(item=>item.primaryRule===rule).length,name:{red_light:'red-light',wrong_way:'wrong-way',restricted_zone:'restricted-area'}[rule]})).filter(item=>item.count).map(item=>`${item.count} ${item.name} example${item.count===1?'':'s'}`).join(', ')+(Object.keys(collisionExamples).length?'; '+Object.keys(collisionExamples).length+' collision examples':'')+'. Supporting footage is available below; no accuracy result is implied.'}
async function loadDatasetCandidates(){try{const response=await fetch('/api/dataset-candidates'),data=await response.json();if(!response.ok)throw Error(data.error||'Candidate catalogue unavailable');if(Array.isArray(data.available_builtins))caseOrder=caseOrder.filter(key=>data.available_builtins.includes(key)||key.startsWith('candidate-'));for(const item of [...data.candidates].sort((a,b)=>Number(b.library_tier==='task_example')-Number(a.library_tier==='task_example'))){const key='candidate-'+item.id;if(sampleCatalog[key])continue;const isCurated=item.library_tier==='task_example',name=item.display_name||`Evaluation video: ${item.id}`,reviewedReplay=item.reviewed_replay,role=item.role==='control_candidate'?'Control candidate: review for false positives':'Violation candidate: verify the complete event';sampleCatalog[key]={name,description:reviewedReplay||`${item.reason} ${item.quality_note||''} Review: ${item.review_hint||'inspect the complete clip'}. Complete vehicle-level labels before training or evaluation. Source group: ${item.source_id}.`,url:item.media_url,path:item.path,posterUrl:item.poster_url,libraryTier:item.library_tier||'supplementary',primaryRule:item.rule,sourceGroup:item.source_id,mediaFacts:item.width?`${item.width} x ${item.height} encoded pixels | ${Number(item.duration_seconds).toFixed(1)} s | ${Number(item.fps).toFixed(1)} FPS`:'',backend:'yolo',imageSize:item.image_size||960,confidence:item.confidence??.15,camera:item.camera_id||item.source_id||item.id,evaluationCandidate:true,reviewTime:item.review_time_seconds,points:[],stop:[],signal:[],direction:[],restricted:[],rules:{red_light:item.rule==='red_light',wrong_way:item.rule==='wrong_way',restricted_zone:item.rule==='restricted_zone'}};caseFacts[key]={badge:reviewedReplay?'LABELLED TEST REPLAY':isCurated?(item.role==='control_candidate'?'RED-LIGHT CONTROL CANDIDATE':'RED-LIGHT EVALUATION CANDIDATE'):'SUPPLEMENTARY / REVIEW REQUIRED',expected:reviewedReplay||(isCurated?role:'No accuracy claim until complete manual review'),challenge:item.review_hint||'Draw camera-specific geometry and review the entire clip'};caseOrder.push(key);const option=document.createElement('option');option.value=key;option.textContent=(isCurated?'Task example: ':'Supporting: ')+name;document.getElementById('sampleSelect').appendChild(option)}for(const key of caseOrder.filter(key=>!key.startsWith('candidate-'))){const option=document.createElement('option');option.value=key;option.textContent=(sampleCatalog[key].libraryTier==='task_example'?'Task example: ':'Supporting: ')+sampleCatalog[key].name;document.getElementById('sampleSelect').appendChild(option)}const select=document.getElementById('sampleSelect');if(select.options){const sorted=[...select.options].slice(1).sort((a,b)=>Number(sampleCatalog[b.value]?.libraryTier==='task_example')-Number(sampleCatalog[a.value]?.libraryTier==='task_example'));for(const option of sorted)select.appendChild(option)}renderCaseGallery()}catch(error){document.getElementById('curatedLibrarySummary').textContent='Video catalogue could not be loaded: '+error.message}}
document.getElementById('sampleSelect').onchange=e=>{const sample=sampleCatalog[e.target.value];document.getElementById('sampleDescription').textContent=sample?.description||'Choose a screened video to see its evidence and review status.'};
document.getElementById('curatedRuleFilter').onchange=renderCaseGallery;
document.getElementById('sampleBtn').onclick=()=>loadSample(document.getElementById('sampleSelect').value);
async function saveCurrentSetupToExample(){if(!currentSampleKey||!currentSample){alert('Load an example before saving its setup.');return}try{const profile=cameraProfilePayload();if(!confirm(`Replace the default setup loaded for "${currentSample.name}" with the geometry currently on screen?`))return;const response=await fetch('/api/example-setup',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({example_key:currentSampleKey,profile})}),data=await response.json();if(!response.ok)throw Error(data.error||'Example setup save failed');document.getElementById('defaultBtn').textContent='Restore saved example setup';document.getElementById('sampleDescription').textContent=currentSample.description+' Saved setup will now load automatically for this example.';alert('Saved. Load selected example will now use this setup instead of the original preset.')}catch(error){alert(error.message)}}
document.getElementById('saveExampleSetupBtn').onclick=saveCurrentSetupToExample;
document.getElementById('captureBtn').onclick=()=>{if(!preview.videoWidth){alert('Load a recording before capturing a frame.');return}captureFrame();showWorkspace('geometry-step',true,true)};
document.getElementById('reviewHintBtn').onclick=()=>{if(currentSample?.reviewTime===undefined)return;preview.pause();reviewVideo.pause();preview.currentTime=currentSample.reviewTime;reviewVideo.currentTime=currentSample.reviewTime;showWorkspace('labels-step');updateCurrentTime();reviewVideo.scrollIntoView({behavior:'smooth'})};
document.getElementById('laneMode').onclick=()=>setMode('lane',true);document.getElementById('intersectionAssistBtn').onclick=analyzeIntersection;document.getElementById('forwardLaneBtn').onclick=forwardFitLane;document.getElementById('zoneMode').onclick=()=>setMode('edit');document.getElementById('lineMode').onclick=()=>setMode('line',true);document.getElementById('signalMode').onclick=()=>setMode('signal',true);document.getElementById('directionMode').onclick=()=>setMode('direction',true);document.getElementById('restrictedMode').onclick=()=>setMode('restricted',true);
for(const id of ['redRule','wrongWayRule','restrictedRule'])document.getElementById(id).addEventListener('change',()=>{updateConfigurationHealth();commitActiveLane()});
document.getElementById('laneSelect').onchange=e=>{const targetIndex=Number(e.target.value);commitActiveLane();activeLaneIndex=targetIndex;loadLaneState(laneConfigurations[activeLaneIndex]);renderLaneSelector()};
document.getElementById('laneName').oninput=e=>{if(laneConfigurations[activeLaneIndex]){laneConfigurations[activeLaneIndex].name=e.target.value||`Lane ${activeLaneIndex+1}`;renderLaneSelector()}};
document.getElementById('addLaneBtn').onclick=()=>{if(laneConfigurations.length>=8){alert('A maximum of eight lanes is supported.');return}commitActiveLane();let number=1;while(laneConfigurations.some(lane=>lane.id===`lane-${number}`))number++;activeLaneIndex=laneConfigurations.length;laneConfigurations.push({id:`lane-${number}`,name:`Lane ${number}`,polygon:[],stop_line:[],signal_rectangle:[],direction:[],restricted_polygon:[],restricted_initial_presence:false,lane_assist:{method:'unconfigured'},enabled_rules:{red_light:false,wrong_way:true,restricted_zone:false}});loadLaneState(laneConfigurations[activeLaneIndex]);renderLaneSelector();setMode('lane',true)};
document.getElementById('deleteLaneBtn').onclick=()=>{if(laneConfigurations.length<=1)return;if(!confirm(`Delete ${laneConfigurations[activeLaneIndex].name||'the selected lane'} and all of its rule geometry?`))return;laneConfigurations.splice(activeLaneIndex,1);activeLaneIndex=Math.min(activeLaneIndex,laneConfigurations.length-1);loadLaneState(laneConfigurations[activeLaneIndex]);renderLaneSelector()};
document.getElementById('defaultBtn').onclick=async()=>{if(!frameReady||!currentSample||!currentSampleKey){alert('A preset is available only after loading an example.');return}try{const response=await fetch('/api/example-setup?id='+encodeURIComponent(currentSampleKey));if(response.ok){const data=await response.json();applySampleSetup(sampleWithSavedSetup(currentSample,data.profile));return}}catch(error){console.warn(error)}applySampleSetup(currentSample)};
document.getElementById('undoBtn').onclick=()=>{if(!drawingArmed)return;if(drawMode==='lane'){points.pop();autoLaneLines=[]}else if(drawMode==='restricted')restrictedPoints.pop();else if(drawMode==='line')stopLinePoints.pop();else if(drawMode==='signal')signalPoints.pop();else if(drawMode==='direction')directionPoints.pop();redraw()};document.getElementById('clearBtn').onclick=()=>{if(!confirm('Clear every configured lane and all unsaved geometry?'))return;resetSetup();drawingArmed=false;frameReady=false;drawPlaceholder()};
function sourceFramePresented(version){if(version!==annotationSourceVersion)return;if(!frameReady)captureFrame();if(pendingSample){applySampleSetup(pendingSample);pendingSample=null}loadVideoMetadata()}
function scheduleSourceFrameCapture(){const version=annotationSourceVersion;if(preview.requestVideoFrameCallback){preview.requestVideoFrameCallback(()=>sourceFramePresented(version))}else{preview.addEventListener('loadeddata',()=>requestAnimationFrame(()=>sourceFramePresented(version)),{once:true})}}

function formatTime(seconds){const total=Math.max(0,Math.round((Number(seconds)||0)*1000)),minutes=Math.floor(total/60000),secs=Math.floor(total/1000)%60,millis=total%1000;return `${String(minutes).padStart(2,'0')}:${String(secs).padStart(2,'0')}.${String(millis).padStart(3,'0')}`}
function escapeHtml(value){return String(value).replace(/[&<>'"]/g,ch=>({'&':'&amp;','<':'&lt;','>':'&gt;',"'":'&#39;','"':'&quot;'}[ch]))}
async function loadVideoMetadata(){const target=document.getElementById('sourceFps'),version=annotationSourceVersion,path=serverVideoPath;if(!path){sourceVideoFps=30;target.textContent='Source FPS: 30 estimated (upload to verify)';return}try{const response=await fetch('/api/video-metadata?video_path='+encodeURIComponent(path)),data=await response.json();if(version!==annotationSourceVersion||path!==serverVideoPath)return;if(!response.ok)throw Error(data.error||'Video metadata unavailable');sourceVideoFps=Number(data.fps)||30;target.textContent=`Source FPS: ${sourceVideoFps.toFixed(3)} ? ${data.frames} frames`}catch(error){if(version!==annotationSourceVersion)return;sourceVideoFps=30;target.textContent='Source FPS: unavailable ? using 30 FPS estimate'}}
function updateCurrentTime(){document.getElementById('currentTime').textContent=formatTime(reviewVideo.currentTime);renderAnnotationTimeline()}
function drawLabelPlaceholder(){labelCtx.fillStyle='#24282d';labelCtx.fillRect(0,0,labelCanvas.width,labelCanvas.height);labelCtx.fillStyle='#d4dbe2';labelCtx.font='20px Segoe UI';labelCtx.textAlign='center';labelCtx.fillText('Capture a frame, then select one specific vehicle',labelCanvas.width/2,labelCanvas.height/2)}
async function captureLabelFrame(){if(!reviewVideo.videoWidth){alert('The review video is not ready yet.');return}reviewVideo.pause();const scale=Math.min(960/reviewVideo.videoWidth,540/reviewVideo.videoHeight,1),temporary=document.createElement('canvas');temporary.width=Math.round(reviewVideo.videoWidth*scale);temporary.height=Math.round(reviewVideo.videoHeight*scale);temporary.getContext('2d').drawImage(reviewVideo,0,0,temporary.width,temporary.height);labelCanvas.width=temporary.width;labelCanvas.height=temporary.height;labelCtx.drawImage(temporary,0,0);labelBaseImage=labelCtx.getImageData(0,0,labelCanvas.width,labelCanvas.height);labelFrameDataUrl=temporary.toDataURL('image/jpeg',.9);labelFrameTime=Math.round(reviewVideo.currentTime*1000)/1000;labelBoxPoints=[];vehicleProposals=[];selectedProposalIndex=null;manualBoxMode=false;labelFrameReady=true;redrawLabelFrame();document.getElementById('labelBoxInfo').textContent='Finding selectable cars, buses and trucks...';try{const response=await fetch('/api/detect-label-frame',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({image_data_url:labelFrameDataUrl,model_name:document.getElementById('modelName').value,confidence:Math.min(.2,Number(document.getElementById('confidence').value)||.15)})}),data=await response.json();if(!response.ok)throw Error(data.error||'Vehicle proposal detection failed');vehicleProposals=data.proposals||[];redrawLabelFrame()}catch(error){manualBoxMode=true;document.getElementById('labelBoxInfo').textContent=`Vehicle proposals unavailable: ${error.message}. Manual box mode is active.`}}
function redrawLabelFrame(){if(!labelFrameReady||!labelBaseImage){drawLabelPlaceholder();return}labelCtx.putImageData(labelBaseImage,0,0);vehicleProposals.forEach((proposal,index)=>{const b=proposal.bbox_normalized,x=b[0]*labelCanvas.width,y=b[1]*labelCanvas.height,w=(b[2]-b[0])*labelCanvas.width,h=(b[3]-b[1])*labelCanvas.height;labelCtx.strokeStyle=index===selectedProposalIndex?'#00e5ff':'#2d7ff0';labelCtx.lineWidth=index===selectedProposalIndex?4:2;labelCtx.strokeRect(x,y,w,h);labelCtx.fillStyle=index===selectedProposalIndex?'#00e5ff':'#2d7ff0';labelCtx.font='bold 12px Segoe UI';labelCtx.fillText(`${index+1} ${proposal.class_name} ${(proposal.confidence*100).toFixed(0)}%`,x+3,Math.max(13,y-4))});if(labelBoxPoints.length===2){const a=labelBoxPoints[0],b=labelBoxPoints[1];labelCtx.strokeStyle='#00e5ff';labelCtx.lineWidth=4;labelCtx.strokeRect(Math.min(a[0],b[0])*labelCanvas.width,Math.min(a[1],b[1])*labelCanvas.height,Math.abs(a[0]-b[0])*labelCanvas.width,Math.abs(a[1]-b[1])*labelCanvas.height)}else if(labelBoxPoints.length===1){const p=labelBoxPoints[0];labelCtx.fillStyle='#00e5ff';labelCtx.beginPath();labelCtx.arc(p[0]*labelCanvas.width,p[1]*labelCanvas.height,6,0,Math.PI*2);labelCtx.fill()}const state=labelBoxPoints.length===2?'vehicle selected':manualBoxMode?'click two opposite corners':vehicleProposals.length?`click one of ${vehicleProposals.length} proposed vehicles`:'no proposal found; use manual box';document.getElementById('labelBoxInfo').textContent=`Captured time: ${formatTime(labelFrameTime)}. ${state}.`}
labelCanvas.addEventListener('click',e=>{if(!labelFrameReady)return;const r=labelCanvas.getBoundingClientRect(),p=[(e.clientX-r.left)/r.width,(e.clientY-r.top)/r.height];if(!manualBoxMode&&vehicleProposals.length){const matches=vehicleProposals.map((proposal,index)=>({proposal,index,area:(proposal.bbox_normalized[2]-proposal.bbox_normalized[0])*(proposal.bbox_normalized[3]-proposal.bbox_normalized[1])})).filter(item=>{const b=item.proposal.bbox_normalized;return p[0]>=b[0]&&p[0]<=b[2]&&p[1]>=b[1]&&p[1]<=b[3]}).sort((a,b)=>a.area-b.area);if(matches.length){selectedProposalIndex=matches[0].index;const b=matches[0].proposal.bbox_normalized;labelBoxPoints=[[b[0],b[1]],[b[2],b[3]]];redrawLabelFrame();return}}if(labelBoxPoints.length===2)labelBoxPoints=[];labelBoxPoints.push(p);redrawLabelFrame()});
document.getElementById('manualBoxBtn').onclick=()=>{manualBoxMode=true;selectedProposalIndex=null;labelBoxPoints=[];redrawLabelFrame()};
const eventNames={candidate_red_signal_violation:'Red-light running',candidate_wrong_way_driving:'Wrong-way driving',candidate_restricted_zone_entry:'Restricted-zone entry'};
const eventDescriptions={candidate_red_signal_violation:'Vehicle crosses the stop line while the controlled signal is red',candidate_wrong_way_driving:'Vehicle persistently travels opposite to the configured legal direction',candidate_restricted_zone_entry:'Vehicle enters the configured restricted zone'};
const eventRuleKeys={candidate_red_signal_violation:'red_light',candidate_wrong_way_driving:'wrong_way',candidate_restricted_zone_entry:'restricted_zone'};
function syncLabelRuleOptions(preserveSelection=false){const laneSelect=document.getElementById('labelLaneId'),eventSelect=document.getElementById('labelEventType');if(!laneSelect||!eventSelect)return;const lane=laneConfigurations.find(item=>item.id===laneSelect.value);if(!lane)return;const enabled=lane.enabled_rules||{};for(const option of eventSelect.options)option.disabled=!Boolean(enabled[eventRuleKeys[option.value]]);const selected=eventSelect.options[eventSelect.selectedIndex];if(!preserveSelection&&(!selected||selected.disabled)){const replacement=[...eventSelect.options].find(option=>!option.disabled);if(replacement){eventSelect.value=replacement.value;document.getElementById('labelDescription').value=eventDescriptions[replacement.value]}}}
document.getElementById('labelLaneId').onchange=()=>syncLabelRuleOptions();
document.getElementById('labelEventType').onchange=e=>{document.getElementById('labelDescription').value=eventDescriptions[e.target.value]};
function labelThumbnail(bbox){if(!labelFrameDataUrl)return'';const image=document.createElement('canvas'),source=document.createElement('canvas');source.width=labelCanvas.width;source.height=labelCanvas.height;source.getContext('2d').putImageData(labelBaseImage,0,0);const x=Math.max(0,Math.floor(bbox[0]*source.width)),y=Math.max(0,Math.floor(bbox[1]*source.height)),w=Math.max(1,Math.ceil((bbox[2]-bbox[0])*source.width)),h=Math.max(1,Math.ceil((bbox[3]-bbox[1])*source.height));image.width=160;image.height=100;image.getContext('2d').drawImage(source,x,y,w,h,0,0,160,100);return image.toDataURL('image/jpeg',.72)}
function renderLabels(){const host=document.getElementById('labelList');updateAnnotationStatus();renderAnnotationTimeline();if(!labels.length){host.innerHTML='<p class="empty-labels">No event labels yet. Continue reviewing the complete original video.</p>';return}host.innerHTML=labels.map((item,index)=>`<div class="label-card">${item.thumbnail_data_url?`<img src="${item.thumbnail_data_url}" alt="Selected vehicle">`:'<div class="hint">No thumbnail</div>'}<div><strong>${formatTime(item.timestamp_seconds)} · ${eventNames[item.candidate_event]||escapeHtml(item.candidate_event)}</strong><p class="hint">${escapeHtml(item.lane_name||item.lane_id||'Lane 1')} · <span class="${item.status}">${item.status}</span> · ${escapeHtml(item.vehicle_class||'vehicle')}</p><p>${escapeHtml(item.description)}</p></div><div><button data-seek="${index}">Go to</button> <button data-edit="${index}">Edit</button> <button data-delete="${index}" class="danger">Delete</button></div></div>`).join('');host.querySelectorAll('[data-seek]').forEach(button=>button.onclick=()=>{reviewVideo.currentTime=labels[Number(button.dataset.seek)].timestamp_seconds;reviewVideo.pause();updateCurrentTime()});host.querySelectorAll('[data-edit]').forEach(button=>button.onclick=()=>editLabel(Number(button.dataset.edit)));host.querySelectorAll('[data-delete]').forEach(button=>button.onclick=()=>{if(confirm('Delete this event label?')){labels.splice(Number(button.dataset.delete),1);scheduleDraftSave();renderLabels()}})}
function editLabel(index){const item=labels[index];editingLabelIndex=index;reviewVideo.pause();reviewVideo.currentTime=item.timestamp_seconds;document.getElementById('labelLaneId').value=item.lane_id;document.getElementById('labelEventType').value=item.candidate_event;syncLabelRuleOptions(true);document.getElementById('labelDescription').value=item.description;labelFrameTime=item.timestamp_seconds;labelBoxPoints=[[item.vehicle_bbox_normalized[0],item.vehicle_bbox_normalized[1]],[item.vehicle_bbox_normalized[2],item.vehicle_bbox_normalized[3]]];document.getElementById('markBtn').textContent='Update confirmed label';document.getElementById('labels-step').open=true;captureLabelFrame().then(()=>{labelBoxPoints=[[item.vehicle_bbox_normalized[0],item.vehicle_bbox_normalized[1]],[item.vehicle_bbox_normalized[2],item.vehicle_bbox_normalized[3]]];manualBoxMode=true;redrawLabelFrame()})}
function addLabel(status){if(!labelFrameReady||labelBoxPoints.length!==2){alert('Capture the event frame and select one specific vehicle first.');return}commitActiveLane();const candidateEvent=document.getElementById('labelEventType').value,laneId=document.getElementById('labelLaneId').value,lane=laneConfigurations.find(item=>item.id===laneId);if(!lane?.enabled_rules?.[eventRuleKeys[candidateEvent]]){alert(`${eventNames[candidateEvent]} is not enabled for ${lane?.name||laneId}. Select one of the available violation types for this lane.`);return}const xs=labelBoxPoints.map(p=>p[0]),ys=labelBoxPoints.map(p=>p[1]),bbox=[Math.min(...xs),Math.min(...ys),Math.max(...xs),Math.max(...ys)],vehiclePoint=[(bbox[0]+bbox[2])/2,bbox[3]],description=document.getElementById('labelDescription').value.trim()||eventNames[candidateEvent],proposal=selectedProposalIndex===null?null:vehicleProposals[selectedProposalIndex],item={timestamp_seconds:labelFrameTime,lane_id:laneId,lane_name:lane?.name||laneId,candidate_event:candidateEvent,status,vehicle_class:proposal?.class_name||'vehicle',selection_method:proposal?'detector_proposal_confirmed_by_human':'manual_box',vehicle_bbox_normalized:bbox.map(v=>Math.round(v*1000000)/1000000),vehicle_point_normalized:vehiclePoint.map(v=>Math.round(v*1000000)/1000000),description,thumbnail_data_url:labelThumbnail(bbox)};if(editingLabelIndex===null)labels.push(item);else{labels[editingLabelIndex]=item;editingLabelIndex=null;document.getElementById('markBtn').textContent='Mark selected vehicle'}labels.sort((a,b)=>a.timestamp_seconds-b.timestamp_seconds);labelBoxPoints=[];selectedProposalIndex=null;redrawLabelFrame();scheduleDraftSave();renderLabels()}
function mergeReviewedIntervals(){const duration=Number(reviewVideo.duration)||0,sorted=reviewedIntervals.map(x=>[Math.max(0,Math.min(...x)),Math.min(duration,Math.max(...x))]).filter(x=>x[1]>x[0]).sort((a,b)=>a[0]-b[0]),merged=[];for(const interval of sorted){if(merged.length&&interval[0]<=merged[merged.length-1][1])merged[merged.length-1][1]=Math.max(merged[merged.length-1][1],interval[1]);else merged.push(interval)}reviewedIntervals=merged;return merged}
function reviewCoverageFraction(){const duration=Number(reviewVideo.duration)||0;if(!duration)return 0;return Math.min(1,mergeReviewedIntervals().reduce((sum,x)=>sum+x[1]-x[0],0)/duration)}
function recordReviewProgress(){const now=reviewVideo.currentTime;if(!reviewVideo.paused&&!reviewVideo.seeking&&lastReviewedTime!==null){const delta=now-lastReviewedTime;if(delta>0&&delta<=Math.max(1.2,reviewVideo.playbackRate*1.2))reviewedIntervals.push([lastReviewedTime,now])}lastReviewedTime=reviewVideo.paused?null:now;updateCurrentTime();updateAnnotationStatus();if(!reviewVideo.paused)scheduleDraftSave()}
function renderAnnotationTimeline(){const host=document.getElementById('annotationTimeline');if(!host)return;const duration=Number(reviewVideo.duration)||0;host.querySelectorAll('.timeline-reviewed,.timeline-marker').forEach(node=>node.remove());if(duration){for(const interval of mergeReviewedIntervals()){const bar=document.createElement('div');bar.className='timeline-reviewed';bar.style.left=(interval[0]/duration*100)+'%';bar.style.width=((interval[1]-interval[0])/duration*100)+'%';host.appendChild(bar)}for(const item of labels){const marker=document.createElement('div');marker.className='timeline-marker '+(item.status==='uncertain'?'uncertain':'');marker.style.left=(item.timestamp_seconds/duration*100)+'%';marker.title=`${formatTime(item.timestamp_seconds)} ${eventNames[item.candidate_event]||item.candidate_event}`;host.appendChild(marker)}if(document.getElementById('annotationMode').value==='error_analysis')for(const event of predictionEvents){const marker=document.createElement('div');marker.className='timeline-marker prediction';marker.style.left=(event.timestamp_seconds/duration*100)+'%';marker.title=`Model candidate ${formatTime(event.timestamp_seconds)}`;host.appendChild(marker)}document.getElementById('timelinePlayhead').style.left=(reviewVideo.currentTime/duration*100)+'%'}}
function updateAnnotationStatus(){const coverage=reviewCoverageFraction()*100;document.getElementById('reviewCoverage').textContent=coverage.toFixed(1)+'%';document.getElementById('confirmedCount').textContent=labels.filter(x=>x.status==='confirmed').length;document.getElementById('uncertainCount').textContent=labels.filter(x=>x.status==='uncertain').length;const ready=coverage>=98;document.getElementById('coverageHelp').className=ready?'health-good':'hint';document.getElementById('coverageHelp').textContent=ready?'Full-review coverage threshold reached. Resolve uncertain labels before completion.':'Play the original video normally to record reviewed intervals. At least 98% measured coverage is required before final completion.'}
function updateReviewPlaybackButton(){document.getElementById('reviewPlayPauseBtn').textContent=reviewVideo.paused?'Play review video':'Pause review video'}
async function toggleReviewPlayback(){if(!reviewVideo.src)return;if(reviewVideo.ended)reviewVideo.currentTime=0;if(reviewVideo.paused){try{await reviewVideo.play()}catch(error){alert('The review video could not start: '+error.message)}}else reviewVideo.pause()}
reviewVideo.addEventListener('loadedmetadata',()=>{updateCurrentTime();restoreAnnotationDraft()});reviewVideo.addEventListener('timeupdate',recordReviewProgress);reviewVideo.addEventListener('seeked',()=>{lastReviewedTime=null;updateCurrentTime()});reviewVideo.addEventListener('play',()=>{lastReviewedTime=reviewVideo.currentTime;updateReviewPlaybackButton()});reviewVideo.addEventListener('pause',()=>{lastReviewedTime=null;updateReviewPlaybackButton()});reviewVideo.addEventListener('ended',updateReviewPlaybackButton);
document.getElementById('reviewPlayPauseBtn').onclick=toggleReviewPlayback;
document.addEventListener('keydown',event=>{const tag=event.target?.tagName;if(event.code!=='Space'||!document.getElementById('labels-step').open||['INPUT','SELECT','TEXTAREA','BUTTON'].includes(tag))return;event.preventDefault();toggleReviewPlayback()});
document.getElementById('annotationTimeline').onclick=e=>{if(!reviewVideo.duration)return;const r=e.currentTarget.getBoundingClientRect();reviewVideo.currentTime=Math.max(0,Math.min(reviewVideo.duration,(e.clientX-r.left)/r.width*reviewVideo.duration));reviewVideo.pause()};
function seekRelative(seconds){reviewVideo.pause();reviewVideo.currentTime=Math.max(0,Math.min(reviewVideo.duration||Infinity,reviewVideo.currentTime+seconds))}
document.getElementById('backFiveBtn').onclick=()=>seekRelative(-5);document.getElementById('backOneBtn').onclick=()=>seekRelative(-1);document.getElementById('backFrameBtn').onclick=()=>seekRelative(-1/sourceVideoFps);
document.getElementById('forwardFrameBtn').onclick=()=>seekRelative(1/sourceVideoFps);document.getElementById('forwardOneBtn').onclick=()=>seekRelative(1);document.getElementById('forwardFiveBtn').onclick=()=>seekRelative(5);
document.getElementById('playbackRate').onchange=e=>{reviewVideo.playbackRate=Number(e.target.value)};
document.getElementById('nextGapBtn').onclick=()=>{const duration=Number(reviewVideo.duration)||0,intervals=mergeReviewedIntervals();let cursor=0,gaps=[];for(const interval of intervals){if(interval[0]-cursor>.5)gaps.push([cursor,interval[0]]);cursor=Math.max(cursor,interval[1])}if(duration-cursor>.5)gaps.push([cursor,duration]);if(!gaps.length){alert('No unreviewed gap longer than 0.5 seconds remains.');return}const next=gaps.find(gap=>gap[0]>reviewVideo.currentTime+.1)||gaps[0];reviewVideo.pause();reviewVideo.currentTime=Math.min(duration,next[0]+.02)};
document.getElementById('captureLabelBtn').onclick=captureLabelFrame;
document.getElementById('markBtn').onclick=()=>addLabel('confirmed');document.getElementById('uncertainBtn').onclick=()=>addLabel('uncertain');
document.getElementById('undoLabelBtn').onclick=()=>{labels.pop();scheduleDraftSave();renderLabels()};document.getElementById('clearLabelsBtn').onclick=()=>{if(labels.length&&confirm('Clear all event labels for this video?')){labels=[];scheduleDraftSave();renderLabels()}};
function confirmedLabels(){return labels.filter(item=>item.status==='confirmed').map(item=>({timestamp_seconds:item.timestamp_seconds,lane_id:item.lane_id,lane_name:item.lane_name,candidate_event:item.candidate_event,vehicle_class:item.vehicle_class,selection_method:item.selection_method,vehicle_bbox_normalized:item.vehicle_bbox_normalized,vehicle_point_normalized:item.vehicle_point_normalized,description:item.description,annotation_status:'confirmed'}))}
async function ensureAnnotationVideo(){const version=annotationSourceVersion;if(serverVideoPath)return serverVideoPath;if(!selectedFile)throw Error('Select a video first.');document.getElementById('draftState').textContent='Uploading';if(!annotationUpload||annotationUpload.version!==version)annotationUpload={version,promise:uploadFile(selectedFile)};const pending=annotationUpload;let path;try{path=await pending.promise}catch(error){if(annotationUpload===pending)annotationUpload=null;throw error}if(version!==annotationSourceVersion)throw Error('Video changed during upload.');serverVideoPath=path;await loadVideoMetadata();if(version!==annotationSourceVersion)throw Error('Video changed during upload.');return path}
function draftPayload(){return{video_name:fileName.textContent,video_path:serverVideoPath,reviewer:document.getElementById('reviewerName').value.trim(),mode:document.getElementById('annotationMode').value,labels,reviewed_intervals:mergeReviewedIntervals(),duration_seconds:Number(reviewVideo.duration)||0,current_time_seconds:reviewVideo.currentTime}}
async function saveAnnotationDraft(){const version=annotationSourceVersion,revision=annotationRevision,payload=JSON.parse(JSON.stringify(draftPayload()));const persist=async()=>{try{if(!payload.video_path){if(version!==annotationSourceVersion)return;payload.video_path=await ensureAnnotationVideo()}const response=await fetch('/api/annotation-draft',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(payload)}),data=await response.json();if(!response.ok)throw Error(data.error||'Draft save failed');if(version!==annotationSourceVersion||revision!==annotationRevision)return;groundTruthFinalized=Boolean(data.review_complete);document.getElementById('draftState').textContent=groundTruthFinalized?'Finalized':'Saved';document.getElementById('labelSaveStatus').textContent='Draft saved: '+data.path}catch(error){if(version!==annotationSourceVersion)return;document.getElementById('draftState').textContent='Save failed';document.getElementById('labelSaveStatus').textContent=error.message}};annotationSaveQueue=annotationSaveQueue.catch(()=>{}).then(persist);return annotationSaveQueue}
function scheduleDraftSave(){annotationRevision++;document.getElementById('draftState').textContent='Unsaved';if(annotationDraftTimer!==null)return;annotationDraftTimer=setTimeout(()=>{annotationDraftTimer=null;void saveAnnotationDraft()},1500)}
async function restoreAnnotationDraft(){const path=serverVideoPath,version=annotationSourceVersion,revision=annotationRevision;if(!path||annotationRestoredFor===path)return;annotationRestoredFor=path;try{const response=await fetch('/api/annotation-draft?video_path='+encodeURIComponent(path)),data=await response.json();if(version!==annotationSourceVersion||path!==serverVideoPath||revision!==annotationRevision)return;if(!response.ok)throw Error(data.error||'Draft load failed');if(!data.draft)return;labels=data.draft.labels||[];reviewedIntervals=data.draft.reviewed_intervals||[];groundTruthFinalized=Boolean(data.draft.review_complete);document.getElementById('reviewerName').value=data.draft.reviewer||'';if(Number.isFinite(data.draft.current_time_seconds))reviewVideo.currentTime=data.draft.current_time_seconds;document.getElementById('draftState').textContent=groundTruthFinalized?'Finalized':'Restored';renderLabels()}catch(error){if(version!==annotationSourceVersion)return;annotationRestoredFor=null;document.getElementById('draftState').textContent='Not saved'}}
async function runQualityChecks(items=labels){commitActiveLane();const host=document.getElementById('qualityCheckResult');host.className='hint';host.textContent='Running annotation quality checks...';try{const response=await fetch('/api/annotation-quality',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({labels:items,duration_seconds:Number(reviewVideo.duration)||0,lanes:laneConfigurations})}),data=await response.json();if(!response.ok)throw Error(data.error||'Quality check failed');if(!data.issues.length){host.className='health-good';host.textContent='Quality checks passed: no duplicate, timing, vehicle-box or lane/rule conflicts found.'}else{host.className=data.passed?'health-warn':'health-bad';host.innerHTML=`<strong>${data.error_count} error(s), ${data.warning_count} warning(s)</strong><ul>${data.issues.map(issue=>`<li><strong>${escapeHtml(issue.severity.toUpperCase())}:</strong> ${escapeHtml(issue.message)}</li>`).join('')}</ul>`}return data}catch(error){host.className='health-bad';host.textContent=error.message;return{passed:false,error_count:1,warning_count:0,issues:[]}}}
document.getElementById('qualityCheckBtn').onclick=()=>runQualityChecks();
async function submitGroundTruth(negative){const sourceVersion=annotationSourceVersion,reviewRevision=annotationRevision;const reviewer=document.getElementById('reviewerName').value.trim();if(!reviewer){alert('Enter the reviewer name or ID.');return}if(reviewCoverageFraction()<.98){alert('Complete at least 98% measured full-video review before finalizing ground truth.');return}if(labels.some(item=>item.status==='uncertain')){alert('Resolve or delete every uncertain label before final completion.');return}const confirmed=confirmedLabels();if(negative&&labels.length){alert('A negative review cannot contain event labels.');return}if(!negative&&!confirmed.length){alert('Add at least one confirmed event, or save a fully reviewed negative.');return}const quality=await runQualityChecks(negative?[]:confirmed);if(!quality.passed){alert('Resolve the annotation quality errors before final completion.');return}reviewVideo.pause();clearTimeout(annotationDraftTimer);annotationDraftTimer=null;await ensureAnnotationVideo();await saveAnnotationDraft();if(sourceVersion!==annotationSourceVersion||reviewRevision!==annotationRevision){alert('The video or review changed while checks were running. Review the current labels and finalize again.');return}if(!confirm('Finalize this independent full-video review? Ground truth should be completed before model error analysis.'))return;const response=await fetch('/api/save-ground-truth',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({video_name:fileName.textContent,video_path:serverVideoPath,annotations:negative?[]:confirmed,reviewer,reviewed_intervals:mergeReviewedIntervals(),duration_seconds:Number(reviewVideo.duration)||0,review_complete:true,review_note:negative?'Full-video review completed; no eligible events found.':'Full-video review completed with confirmed vehicle events.',register_campaign:Boolean(currentSample?.evaluationCandidate),job_id:jobId,camera_id:document.getElementById('cameraId').value,lanes:laneConfigurations})}),data=await response.json();if(!response.ok){alert(data.error||'Could not finalize ground truth');return}if(sourceVersion!==annotationSourceVersion||reviewRevision!==annotationRevision)return;groundTruthFinalized=true;document.getElementById('draftState').textContent='Finalized';document.getElementById('labelSaveStatus').textContent=`Finalized ${data.review_coverage_percent.toFixed(1)}% review: ${data.path}`+(data.campaign_registration?' · Campaign: '+data.campaign_registration:'');loadResearchReadiness()}
document.getElementById('saveDraftBtn').onclick=saveAnnotationDraft;document.getElementById('saveLabelsBtn').onclick=()=>submitGroundTruth(false);document.getElementById('saveNegativeBtn').onclick=()=>submitGroundTruth(true);
document.getElementById('exportLabelsBtn').onclick=()=>{const confirmed=confirmedLabels();if(!confirmed.length){alert('Add at least one confirmed violation before exporting.');return}const stem=(fileName.textContent||'video').replace(/\.[^.]+$/,'').replace(/[^A-Za-z0-9_-]+/g,'-').replace(/^-|-$/g,'')||'video',blob=new Blob([JSON.stringify(confirmed,null,2)+'\n'],{type:'application/json'}),url=URL.createObjectURL(blob),link=document.createElement('a');link.href=url;link.download=stem+'_ground_truth.json';document.body.appendChild(link);link.click();link.remove();setTimeout(()=>URL.revokeObjectURL(url),1000)};
renderLabels();updateCurrentTime();drawLabelPlaceholder();
document.getElementById('annotationMode').onchange=e=>{const result=document.getElementById('result');if(e.target.value==='error_analysis'&&!groundTruthFinalized){alert('Finalize independent ground truth before opening model error analysis.');e.target.value='ground_truth';return}if(e.target.value==='ground_truth'){result.style.display='none'}else{result.style.removeProperty('display');if(jobId)loadEventReview(jobId)}renderAnnotationTimeline()};

function uploadFile(file){return new Promise((resolve,reject)=>{const xhr=new XMLHttpRequest();xhr.open('POST','/api/upload?filename='+encodeURIComponent(file.name));xhr.upload.onprogress=e=>{if(e.lengthComputable){const p=e.loaded/e.total*100;progress.style.width=p+'%';statusEl.textContent='Copying video: '+p.toFixed(0)+'%'}};xhr.onload=()=>{if(xhr.status===200)resolve(JSON.parse(xhr.responseText).video_path);else reject(new Error(xhr.responseText))};xhr.onerror=()=>reject(new Error('Video copy failed'));xhr.setRequestHeader('Content-Type','application/octet-stream');xhr.send(file)})}
function laneValidationIssue(lane){const enabled=lane.enabled_rules||{},prefix=lane.name||lane.id;if(!Object.values(enabled).some(Boolean))return `${prefix}: enable at least one rule`;if((enabled.red_light||enabled.wrong_way)&&(lane.polygon||[]).length<3)return `${prefix}: controlled lane is missing`;if((enabled.red_light)&&(lane.stop_line||[]).length!==2)return `${prefix}: stop line is missing`;if(enabled.red_light&&(lane.signal_rectangle||[]).length!==2)return `${prefix}: traffic-light box is missing`;if((enabled.red_light||enabled.wrong_way)&&(lane.direction||[]).length!==2)return `${prefix}: legal direction is missing`;if(enabled.restricted_zone&&(lane.restricted_polygon||[]).length<3)return `${prefix}: restricted zone is missing`;return null}
async function validateRoadGeometry(lanes){
  if(!setupFrameData)throw Error('Freeze a setup frame before analysis.');
  statusEl.textContent='Checking road geometry...';
  const response=await fetch('/api/validate-road-geometry',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({image_data_url:setupFrameDataUrl(),lanes})}),data=await response.json();
  if(!response.ok)throw Error(data.error||'Road geometry validation failed');
  applyRoadUnderstanding(data);
  for(const check of data.geometry_checks||[]){const lane=lanes.find(item=>item.id===check.lane_id);if(!lane)continue;lane.lane_assist={...(lane.lane_assist||{}),road_model:data.road_understanding.model,road_task:data.road_understanding.task,road_device:data.road_understanding.device,semantic_road_overlap:check.coverage,road_geometry_status:check.classification,review_required:check.classification!=='pass'}}
  const checks=data.geometry_checks||[],flagged=checks.filter(check=>check.classification!=='pass');
  if(flagged.length){const detail=flagged.map(check=>`${check.lane_name} ${check.geometry.replaceAll('_',' ')}: ${Math.round(check.coverage*100)}% road overlap (${check.classification})`).join('\n');if(!confirm(`The road model found geometry that needs human review:\n\n${detail}\n\nContinue using your manual geometry?`))throw Error('Analysis cancelled so the lane geometry can be corrected.')}
  else document.getElementById('modeHelp').textContent=checks.length?'Road-surface sanity check passed for every monitored vehicle lane.':'Road-surface overlap is not applicable to restricted-zone-only rules.';
}
async function startRun(){
  if(!serverVideoPath&&!selectedFile){alert('Select a video first.');return}
  if(!laneConfigurations.length)initialiseSingleLane('Lane 1');
  commitActiveLane();
  const issue=laneConfigurations.map(laneValidationIssue).find(Boolean);if(issue){alert(issue);return}
  const enabled=laneConfigurations[0].enabled_rules;
  const runBtn=document.getElementById('runBtn');runBtn.disabled=true;document.getElementById('result').classList.remove('show');progress.style.width='0%';
  try{await validateRoadGeometry(laneConfigurations);let path=serverVideoPath;if(!path){path=await uploadFile(selectedFile);serverVideoPath=path}statusEl.textContent='Starting model...';const response=await fetch('/api/run',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({video_path:path,backend:selectedBackend,polygon:laneConfigurations[0].polygon,stop_line:laneConfigurations[0].stop_line,signal_rectangle:laneConfigurations[0].signal_rectangle,direction:laneConfigurations[0].direction,restricted_polygon:laneConfigurations[0].restricted_polygon,restricted_initial_presence:laneConfigurations[0].restricted_initial_presence,lane_assist:laneConfigurations[0].lane_assist,enabled_rules:enabled,lanes:laneConfigurations,camera_id:document.getElementById('cameraId').value,confidence:Number(document.getElementById('confidence').value),image_size:Number(document.getElementById('imageSize').value),model_name:document.getElementById('modelName').value,tracker_name:document.getElementById('trackerName').value,device:document.getElementById('computeDevice').value})});const data=await response.json();if(!response.ok)throw new Error(data.error||'Could not start analysis');jobId=data.job_id;pollJob()}catch(err){runBtn.disabled=false;statusEl.textContent='Start failed';alert(err.message)}
}
async function registerFinalizedReviewAfterRun(){if(!currentSample?.evaluationCandidate||!groundTruthFinalized||!jobId)return;const status=document.getElementById('labelSaveStatus');status.textContent='Registering the finalized review and frozen configuration in the accuracy campaign...';try{const response=await fetch('/api/register-finalized-ground-truth',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({video_path:serverVideoPath,job_id:jobId,camera_id:document.getElementById('cameraId').value})}),data=await response.json();if(!response.ok)throw Error(data.error||'Campaign registration failed');status.textContent='Finalized review registered: '+data.campaign_registration;loadResearchReadiness()}catch(error){status.textContent='Review remains finalized, but campaign registration needs attention: '+error.message}}
async function pollJob(){const response=await fetch('/api/status?id='+encodeURIComponent(jobId));const job=await response.json();if(job.status==='error'){document.getElementById('runBtn').disabled=false;statusEl.textContent='Analysis failed';alert(job.error);return}const total=job.total_frames||0,current=job.current_frame||0,p=total?Math.min(100,current/total*100):0;progress.style.width=p+'%';statusEl.textContent=job.status==='queued'?'Waiting...':`Analyzing: ${current}/${total||'?'} frames (${p.toFixed(0)}%)`;if(job.status!=='complete'){setTimeout(pollJob,500);return}progress.style.width='100%';statusEl.textContent='Analysis complete';document.getElementById('runBtn').disabled=false;showResult(job.summary);await registerFinalizedReviewAfterRun()}
function showResult(s){const reduction=s.metadata_plus_evidence_reduction_percent_vs_video,counts=s.candidate_events_by_type||{},breakdown=Object.entries(counts).map(([type,count])=>`${eventNames[type]||type}: <span class="metric">${count}</span>`).join('; '),laneCounts=s.candidate_events_by_lane||{},laneBreakdown=Object.entries(laneCounts).map(([lane,count])=>`${lane}: <span class="metric">${count}</span>`).join('; '),diagnostics=Object.values(s.tracking_diagnostics_by_lane||{}),recovered=diagnostics.reduce((sum,item)=>sum+item.short_occlusion_recoveries,0),resets=diagnostics.reduce((sum,item)=>sum+item.long_gap_resets,0),stitches=diagnostics.reduce((sum,item)=>sum+(item.tracklet_stitches||0),0),tracker=s.tracker==='botsort'?'BoT-SORT':s.tracker==='bytetrack'?'ByteTrack':s.tracker,throughput=Number(s.processing_fps||0).toFixed(2),realTimeFactor=Number((s.real_time_factor??(s.processing_fps/s.source_fps))||0),realTimeStatus=realTimeFactor>=1.25?'meets 25% headroom':realTimeFactor>=1?'meets source rate without 25% headroom':'does not meet source rate',incidents=s.candidate_incidents??s.candidate_events,multi=s.multi_rule_incidents||0,fallbackExcluded=s.fallback_observations_excluded||0,fallbackWrongWay=s.fallback_wrong_way_observations_evaluated||0,ambiguous=s.lane_assignment_diagnostics?.ambiguous_lane_overlap||0;document.getElementById('metrics').innerHTML=`<div class="metric-grid"><div class="metric-card"><div class="metric-value">${s.frames_processed}</div><div class="metric-label">Frames processed</div></div><div class="metric-card"><div class="metric-value">${throughput}</div><div class="metric-label">Processing FPS</div></div><div class="metric-card"><div class="metric-value">${incidents}</div><div class="metric-label">Vehicle incidents</div></div><div class="metric-card"><div class="metric-value">${s.lane_count||1}</div><div class="metric-label">Independent lanes</div></div></div><div class="result-detail"><strong>Detector:</strong> ${s.model||'unknown'}. <strong>Tracker:</strong> ${tracker}.<br><strong>Event consolidation:</strong> ${s.candidate_events} rule events became <span class="metric">${incidents}</span> vehicle incidents; ${multi} incident(s) contain multiple rule types.<br><strong>Real-time:</strong> ${realTimeFactor.toFixed(2)}× source rate — ${realTimeStatus}.${breakdown?`<br><strong>By rule:</strong> ${breakdown}.`:''}${laneBreakdown?`<br><strong>By lane:</strong> ${laneBreakdown}.`:''}${diagnostics.length?`<br><strong>Tracking continuity:</strong> <span class="metric">${recovered}</span> short same-ID gaps preserved; <span class="metric">${stitches}</span> conservative ID-switch stitches; <span class="metric">${resets}</span> long gaps reset safely.`:''}<br><strong>Reliability policy:</strong> ${fallbackExcluded} provisional observations excluded from non-motion rules; ${fallbackWrongWay} eligible only for persistent wrong-way analysis; ${ambiguous} overlapping-lane observations rejected as ambiguous.<br><strong>Edge-transfer proxy:</strong> metadata plus evidence is <span class="metric">${reduction.toFixed(2)}%</span> smaller than the source video.</div>`;document.getElementById('result').classList.add('show');document.getElementById('result').scrollIntoView({behavior:'smooth'})}
document.getElementById('runBtn').onclick=startRun;
document.getElementById('folderBtn').onclick=()=>fetch('/api/open-folder?id='+encodeURIComponent(jobId),{method:'POST'});
document.getElementById('playBtn').onclick=()=>{const v=document.getElementById('resultVideo'),message=document.getElementById('resultVideoStatus');message.textContent='Loading result video...';v.onerror=()=>message.textContent='The browser could not decode the result video.';v.oncanplay=()=>message.textContent='Result video is ready.';v.src='/api/result-video?id='+encodeURIComponent(jobId)+'&t='+Date.now();v.style.display='block';v.load();v.play().catch(()=>{})};
document.getElementById('reportBtn').onclick=()=>{if(!jobId){alert('Complete an analysis first.');return}window.open('/api/run-report?id='+encodeURIComponent(jobId),'_blank','noopener')};
document.getElementById('bundleBtn').onclick=()=>{if(!jobId){alert('Complete an analysis first.');return}window.location='/api/evidence-bundle?id='+encodeURIComponent(jobId)};
async function loadResearchReadiness(){const metrics=document.getElementById('readinessMetrics'),status=document.getElementById('readinessStatus');try{const [readinessResponse,historyResponse]=await Promise.all([fetch('/api/research-readiness'),fetch('/api/run-history')]),data=await readinessResponse.json(),history=await historyResponse.json();if(!readinessResponse.ok)throw Error(data.error||'Readiness check failed');const campaign=data.campaign||{};metrics.innerHTML=`<div class="readiness-item"><b>${data.accepted_candidate_videos}</b><span>Candidate videos</span></div><div class="readiness-item"><b>${data.completed_full_video_reviews}</b><span>Completed reviews</span></div><div class="readiness-item"><b>${campaign.validation_case_count||0}</b><span>Validation videos</span></div><div class="readiness-item"><b>${campaign.test_case_count||0}</b><span>Test videos</span></div>`;const messages=[...(campaign.blockers||[]),...(campaign.warnings||[])];status.className=campaign.ready?'health-good':'health-warn';status.textContent=campaign.ready?`Ready: ${data.campaign_source} can run as a frozen accuracy study.`:`Not ready yet: ${messages.join(' ')||'complete labels and freeze the validation/test inputs.'}`;renderRunHistory(history.runs||[])}catch(error){status.className='health-warn';status.textContent=error.message}}
function renderRunHistory(runs){const host=document.getElementById('runHistory');if(!runs.length){host.innerHTML='<p class="hint">No completed runs with summary files yet.</p>';return}host.innerHTML=`<table class="history-table"><thead><tr><th>Run</th><th>Model + tracker</th><th>Events / incidents</th><th>FPS</th><th>Real-time</th></tr></thead><tbody>${runs.map(run=>`<tr><td>${escapeHtml(run.run_name)}</td><td>${escapeHtml(run.model)} + ${escapeHtml(run.tracker)}</td><td>${run.candidate_events} / ${run.candidate_incidents}</td><td>${Number(run.processing_fps||0).toFixed(2)}</td><td>${Number(run.real_time_factor||0).toFixed(2)}×</td></tr>`).join('')}</tbody></table>`}
document.getElementById('refreshReadinessBtn').onclick=loadResearchReadiness;
async function loadComputeStatus(){
  try{
    const response=await fetch('/api/compute'),data=await response.json();
    if(!response.ok)throw Error(data.error||'Hardware check failed');
    const message=data.cuda_available?'NVIDIA available: '+data.gpu_name:'CPU available. NVIDIA acceleration requires a CUDA-enabled PyTorch installation and an NVIDIA GPU.';
    document.querySelectorAll('.compute-status').forEach(el=>el.textContent=message);
    document.querySelectorAll('.compute-device option[value="cuda"]').forEach(el=>el.disabled=!data.cuda_available);
  }catch(error){document.querySelectorAll('.compute-status').forEach(el=>el.textContent=error.message)}
}
loadComputeStatus();loadDatasetCandidates();loadResearchReadiness();loadCameraProfiles();initialiseSingleLane('Lane 1');updateRuleToolVisibility();drawPlaceholder();
// Event review and reproducible setup controls.
const reviewHost=document.createElement('div');
document.getElementById('result').appendChild(reviewHost);
const originalSetVideoSource=setVideoSource;
setVideoSource=function(...args){
  jobId=null;document.getElementById('result').classList.remove('show');
  const video=document.getElementById('resultVideo');video.pause();video.removeAttribute('src');video.load();video.style.display='none';
  document.getElementById('runBtn').disabled=false;reviewHost.textContent='';
  originalSetVideoSource(...args);
};
pollJob=async function(){
  const id=jobId;if(!id)return;
  try{
    const response=await fetch('/api/status?id='+encodeURIComponent(id));
    const job=await response.json();if(id!==jobId)return;
    if(!response.ok||job.status==='error')throw Error(job.error||'Could not read analysis status');
    const total=job.total_frames||0,current=job.current_frame||0;
    progress.style.width=(total?Math.min(100,current/total*100):0)+'%';
    statusEl.textContent=job.status==='queued'?'Waiting for the current analysis to finish...':`Analyzing: ${current}/${total||'?'} frames`;
    if(job.status!=='complete'){setTimeout(()=>{if(id===jobId)pollJob()},500);return}
    progress.style.width='100%';statusEl.textContent='Analysis complete';document.getElementById('runBtn').disabled=false;showResult(job.summary);
  }catch(error){if(id===jobId){statusEl.textContent=error.message;document.getElementById('runBtn').disabled=false}}
};
const originalShowResult=showResult;
showResult=function(s){originalShowResult(s);loadEventReview(jobId);if(document.getElementById('annotationMode').value==='ground_truth')document.getElementById('result').style.display='none'};
async function loadEventReview(id){
  reviewHost.textContent='Loading event evidence...';
  try{
    const response=await fetch('/api/events?id='+encodeURIComponent(id));
    if(!response.ok)throw Error('Could not load event evidence');
    const data=await response.json();
    if(id!==jobId)return;
    predictionEvents=data.events||[];renderAnnotationTimeline();
    if(document.getElementById('annotationMode').value==='ground_truth'){reviewHost.innerHTML='<p class="warning">Model candidates are hidden during blind ground-truth review. Finalize ground truth, then switch the annotation workspace to Model error analysis.</p>';return}
    const incidents=data.incidents||[],multi=incidents.filter(item=>item.multi_rule).length;
    reviewHost.innerHTML=`<h3>Candidate incident review</h3><div class="incident-summary"><strong>${incidents.length} consolidated vehicle incident(s)</strong> from ${data.events.length} rule event(s). ${multi} incident(s) combine multiple rule types for the same tracked vehicle.</div><p class="hint">The cards below show rule-level evidence. Select an event to inspect the original video. Vehicle crops contain no added analysis overlay.</p>`;
    if(!data.events.length){reviewHost.insertAdjacentHTML('beforeend','<p>No candidate events. This does not prove that no violation occurred.</p>');return}
    for(const event of data.events){
      const card=document.createElement('div');card.className='card';
      const heading=document.createElement('p');heading.textContent=`${formatTime(event.timestamp_seconds)} · ${event.lane_name||'Lane 1'} · ${eventNames[event.candidate_event]||event.candidate_event} · Vehicle ${event.track_id}`;
      card.appendChild(heading);
      const explanation=event.rule_explanation||{},details=document.createElement('p');details.className='hint';
      const parts=[`Detector confidence ${Number(event.detector_confidence||0).toFixed(2)}`,`Direction alignment ${Number(event.direction_alignment||0).toFixed(2)}`];
      if(event.signal_confidence!==null&&event.signal_confidence!==undefined)parts.push(`Signal confidence ${Number(event.signal_confidence).toFixed(2)}`);
      if(event.evidence_quality_score!==null&&event.evidence_quality_score!==undefined)parts.push(`Evidence strength ${Number(event.evidence_quality_score).toFixed(2)} (not a probability)`);
      if(explanation.observations_used)parts.push(`${explanation.observations_used} trajectory observations`);
      if(explanation.opposite_step_ratio!==null&&explanation.opposite_step_ratio!==undefined)parts.push(`${(explanation.opposite_step_ratio*100).toFixed(0)}% opposite-direction steps`);
      if(explanation.tracklet_stitched)parts.push(`continued from Track ${explanation.previous_track_id}`);
      details.textContent='Why this candidate fired: '+parts.join(' · ');card.appendChild(details);
      const flags=event.review_flags||[];if(flags.length){const warning=document.createElement('p');warning.className='warning';warning.textContent='Human review flags: '+flags.map(flag=>flag.replaceAll('_',' ')).join(', ');card.appendChild(warning)}
      const button=document.createElement('button');button.textContent='View original at event';
      button.onclick=()=>{showWorkspace('video-step');preview.pause();preview.currentTime=event.timestamp_seconds;preview.scrollIntoView({behavior:'smooth'})};card.appendChild(button);
      const img=document.createElement('img');img.alt='Vehicle evidence';img.style.cssText='display:block;max-width:100%;max-height:160px;margin-top:10px';
      img.src='/api/evidence?id='+encodeURIComponent(id)+'&name='+encodeURIComponent(event.vehicle_evidence_pointer.split('/').pop());card.appendChild(img);
      if(event.evidence_clip_pointer){const clip=document.createElement('video');clip.controls=true;clip.preload='metadata';clip.style.cssText='display:block;width:100%;max-width:720px;margin-top:10px;border-radius:10px';clip.src='/api/evidence?id='+encodeURIComponent(id)+'&name='+encodeURIComponent(event.evidence_clip_pointer.split('/').pop());card.appendChild(clip)}
      reviewHost.appendChild(card);
    }
  }catch(error){reviewHost.textContent=error.message}
}
const evaluateButton=document.createElement('button');evaluateButton.textContent='Evaluate against confirmed labels';
document.getElementById('saveLabelsBtn').parentElement.appendChild(evaluateButton);
const evaluationText=document.createElement('pre');evaluationText.style.whiteSpace='pre-wrap';evaluateButton.parentElement.after(evaluationText);
evaluateButton.onclick=async()=>{
  if(!jobId){alert('Complete an analysis first.');return}
  if(!confirm('Have you labelled every relevant event in this video? Incomplete labels make precision and recall misleading.'))return;
  try{
    const response=await fetch('/api/evaluate',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({job_id:jobId,labels:confirmedLabels(),video_path:serverVideoPath})});
    const data=await response.json();if(!response.ok)throw Error(data.error);
    evaluationText.textContent=`TP: ${data.true_positives}  FP: ${data.false_positives}  FN: ${data.false_negatives}\nPrecision: ${data.precision}  Recall: ${data.recall}  F1: ${data.f1}\nTime tolerance: ${data.tolerance_seconds}s; spatial tolerance: ${data.spatial_tolerance_normalized}\n`+Object.entries(data.by_type||{}).map(([k,v])=>`${eventNames[k]||k}: TP ${v.true_positives}, FP ${v.false_positives}, FN ${v.false_negatives}, F1 ${v.f1}`).join('\n');
  }catch(error){evaluationText.textContent=error.message}
};
async function loadCameraProfiles(selected=''){
  const select=document.getElementById('cameraProfileSelect'),status=document.getElementById('cameraProfileStatus');
  try{const response=await fetch('/api/camera-profiles'),data=await response.json();if(!response.ok)throw Error(data.error||'Could not list camera profiles');select.innerHTML='<option value="">No saved profile selected</option>';for(const profile of data.profiles){const option=document.createElement('option');option.value=profile.profile_id;option.textContent=`${profile.display_name} · ${profile.lane_count} lane(s) · ${profile.source_signature.width||'?'}×${profile.source_signature.height||'?'}`;select.appendChild(option)}if(selected)select.value=selected;status.textContent=data.profiles.length?`${data.profiles.length} reusable camera profile(s) available.`:'No profiles saved yet. Calibrate a new camera and save it once.'}catch(error){status.textContent=error.message}
}
function cameraProfilePayload(){
  if(!frameReady||!preview.videoWidth||!preview.videoHeight)throw Error('Freeze a setup frame from the selected video first.');
  commitActiveLane();const issue=laneConfigurations.map(laneValidationIssue).find(Boolean);if(issue)throw Error(issue);
  const cameraId=document.getElementById('cameraId').value.trim();if(!cameraId)throw Error('Enter a stable camera profile ID.');
  return{camera_id:cameraId,display_name:cameraId,source_signature:{width:preview.videoWidth,height:preview.videoHeight,aspect_ratio:preview.videoWidth/preview.videoHeight},calibration_time_seconds:setupFrameTime,lanes:JSON.parse(JSON.stringify(laneConfigurations)),inference:{confidence:Number(document.getElementById('confidence').value),image_size:Number(document.getElementById('imageSize').value),model_name:document.getElementById('modelName').value,tracker_name:document.getElementById('trackerName').value}};
}
async function saveCameraProfile(){const status=document.getElementById('cameraProfileStatus');try{const payload=cameraProfilePayload();if(!confirm(`Save this geometry for camera "${payload.camera_id}"? Reuse it only while the physical camera position and lens remain unchanged.`))return;status.textContent='Validating and saving camera profile...';const response=await fetch('/api/camera-profile',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(payload)}),data=await response.json();if(!response.ok)throw Error(data.error||'Camera profile save failed');await loadCameraProfiles(data.profile.profile_id);status.textContent=`Saved ${data.profile.display_name}. Future clips from this unchanged camera can reuse ${data.profile.lanes.length} lane(s).`}catch(error){status.textContent=error.message;alert(error.message)}}
async function applyCameraProfile(){const select=document.getElementById('cameraProfileSelect'),status=document.getElementById('cameraProfileStatus');if(!select.value){alert('Select a saved camera profile first.');return}if(!frameReady||!preview.videoWidth){alert('Select the new video and freeze a setup frame first.');return}try{const response=await fetch('/api/camera-profile?id='+encodeURIComponent(select.value)),data=await response.json();if(!response.ok)throw Error(data.error||'Camera profile load failed');const profile=data.profile,currentRatio=preview.videoWidth/preview.videoHeight,savedRatio=Number(profile.source_signature.aspect_ratio),difference=Math.abs(currentRatio-savedRatio)/savedRatio;if(difference>.03)throw Error(`Profile rejected: this video is ${preview.videoWidth}×${preview.videoHeight}, but the saved camera is ${profile.source_signature.width}×${profile.source_signature.height}. The aspect ratios differ by ${(difference*100).toFixed(1)}%.`);if(!confirm(`Confirm that this clip comes from the same unchanged physical camera as "${profile.display_name}". Resolution may change, but viewpoint, crop and lens must match.`))return;const first=profile.lanes[0],settings=profile.inference||{};applySampleSetup({camera:profile.camera_id,lanes:profile.lanes,rules:first.enabled_rules,imageSize:settings.image_size,confidence:settings.confidence,modelName:settings.model_name,trackerName:settings.tracker_name});status.textContent=`Loaded ${profile.display_name}. Aspect ratio check passed; visually verify every red lane and rule marker before running.`}catch(error){status.textContent=error.message;alert(error.message)}}
document.getElementById('saveProfileBtn').onclick=saveCameraProfile;document.getElementById('loadProfileBtn').onclick=applyCameraProfile;document.getElementById('refreshProfilesBtn').onclick=()=>loadCameraProfiles();

// Task navigation changes visibility only: source, geometry and review state stay in place.
function showWorkspace(requested,updateHash=true,focus=false){const alias={'analysis-step':'review-step','impact-step':'video-step'},id=alias[requested]||requested,panels=[...document.querySelectorAll('[data-workspace]')],selected=panels.find(panel=>panel.id===id)||panels[0];for(const panel of panels){const active=panel===selected;panel.hidden=!active;if(panel.tagName==='DETAILS')panel.open=active}for(const link of document.querySelectorAll('[data-workspace-link]')){if(link.dataset.workspaceLink===selected.id)link.setAttribute('aria-current','page');else link.removeAttribute('aria-current')}if(updateHash&&location.hash!=='#'+selected.id)history.pushState(null,'','#'+selected.id);if(focus){const title=selected.querySelector('h1');if(title){title.tabIndex=-1;title.focus({preventScroll:true})}window.scrollTo({top:0,behavior:'instant'})}document.dispatchEvent?.(new CustomEvent('workspacechange',{detail:selected.id}));return selected.id}
function syncWorkspaceSource(){canvas.style.setProperty('--frame-ratio',String(canvas.width/canvas.height));const loaded=Boolean(preview.getAttribute('src'));document.body.classList.toggle('has-source',loaded);document.getElementById('sessionSource').textContent=fileName.textContent;const duration=Number(preview.duration),dimensions=preview.videoWidth?`${preview.videoWidth} x ${preview.videoHeight}`:'';document.getElementById('sourceDuration').textContent=Number.isFinite(duration)?formatTime(duration):'--:--';document.getElementById('sessionMedia').textContent=loaded?[dimensions,Number.isFinite(duration)?duration.toFixed(1)+' s':'Loading source'].filter(Boolean).join(' / '):'Fixed-camera video';for(const card of document.querySelectorAll('[data-sample-key]'))card.setAttribute('aria-pressed',String(card.dataset.sampleKey===currentSampleKey));syncAnalysisWorkspace()}
function syncAnalysisWorkspace(){const result=document.getElementById('result'),complete=result.classList.contains('show'),blind=complete&&document.getElementById('annotationMode').value==='ground_truth';document.getElementById('analysisEmpty').hidden=complete||document.getElementById('runBtn').disabled;document.getElementById('blindReviewNotice').hidden=!blind;document.getElementById('sessionState').textContent=statusEl.textContent}
document.querySelectorAll('[data-workspace-link]').forEach(link=>link.addEventListener('click',event=>{event.preventDefault();showWorkspace(link.dataset.workspaceLink,true,true)}));
document.querySelectorAll('[data-go]').forEach(button=>button.addEventListener('click',()=>showWorkspace(button.dataset.go,true,true)));
window.addEventListener('hashchange',()=>showWorkspace(location.hash.slice(1),false));
window.addEventListener('popstate',()=>showWorkspace(location.hash.slice(1),false));
preview.addEventListener('loadedmetadata',syncWorkspaceSource);
preview.addEventListener('emptied',syncWorkspaceSource);
new MutationObserver(()=>canvas.style.setProperty('--frame-ratio',String(canvas.width/canvas.height))).observe(canvas,{attributes:true,attributeFilter:['width','height']});
new MutationObserver(syncWorkspaceSource).observe(fileName,{childList:true,characterData:true,subtree:true});
new MutationObserver(syncAnalysisWorkspace).observe(statusEl,{childList:true,characterData:true,subtree:true});
new MutationObserver(syncAnalysisWorkspace).observe(document.getElementById('result'),{attributes:true,attributeFilter:['class','style']});
document.getElementById('annotationMode').addEventListener('change',syncAnalysisWorkspace);
showWorkspace(location.hash.slice(1),false);syncWorkspaceSource();
// Optional scene-level review stays separate from lane-rule jobs and blind labels.
let collisionJobId=null,collisionModelReady=false;
const collisionPreview=document.getElementById('collisionPreview'),collisionStatus=document.getElementById('collisionStatus');
function syncCollisionSource(){const url=preview.getAttribute('src');if(url!==collisionPreview.getAttribute('src')){collisionPreview.pause();if(url)collisionPreview.src=url;else collisionPreview.removeAttribute('src');collisionPreview.load();document.getElementById('collisionStart').value='0';document.getElementById('collisionEnd').value='8'}document.getElementById('collisionSourceName').textContent=fileName.textContent}
function resetCollisionReview(){collisionJobId=null;document.getElementById('collisionResults').replaceChildren();document.getElementById('collisionRunBtn').disabled=!collisionModelReady;document.getElementById('collisionProgress').style.width='0%';collisionStatus.textContent='Select an interval, then start offline review.'}
const collisionSetVideoSource=setVideoSource;
setVideoSource=function(...args){resetCollisionReview();collisionSetVideoSource(...args);syncCollisionSource()};
collisionPreview.addEventListener('loadedmetadata',()=>{document.getElementById('collisionEnd').value=String(Math.min(Number(document.getElementById('collisionEnd').value)||8,Number(collisionPreview.duration)||8));document.getElementById('collisionEnd').max=String(collisionPreview.duration)});
document.addEventListener('workspacechange',event=>{if(event.detail!=='collision-step')collisionPreview.pause();if(event.detail!=='labels-step')reviewVideo.pause();if(event.detail!=='video-step')preview.pause()});
async function loadCollisionExamples(){try{const response=await fetch('/api/collision-examples'),data=await response.json();if(!response.ok)throw Error(data.error||'Collision examples unavailable');collisionModelReady=Boolean(data.model_available);document.getElementById('collisionModelStatus').textContent=collisionModelReady?'Local video model available. Processing stays on this computer.':'Local video model unavailable; prepare it before using this experiment.';document.getElementById('collisionRunBtn').disabled=!collisionModelReady;for(const item of data.examples){collisionExamples[item.id]=item;const option=document.createElement('option');option.value=item.id;option.textContent=item.no_collision_control?'No-collision control: '+item.name:item.name;document.getElementById('collisionExampleSelect').appendChild(option)}renderCaseGallery()}catch(error){document.getElementById('collisionModelStatus').textContent=error.message;document.getElementById('collisionRunBtn').disabled=true}}
function appendCollisionCards(gallery,filter){
  if(filter!=='all'&&filter!=='collision')return;
  for(const sample of Object.values(collisionExamples)){
    const card=document.createElement('button');
    card.type='button';
    card.className='case-card';
    card.setAttribute('aria-label','Open '+sample.name+' in Collision review');
    const badge=sample.no_collision_control?'NO-COLLISION CONTROL':'COLLISION EXAMPLE';
    card.innerHTML=`<video muted preload="metadata" src="${escapeHtml(sample.media_url)}"></video><div class="case-body"><span class="case-badge">${badge}</span><div class="case-title">${escapeHtml(sample.name)}</div><div class="case-fact">Open in Collision review</div></div>`;
    const thumbnail=card.querySelector('video');
    thumbnail.onloadedmetadata=()=>{thumbnail.currentTime=sample.start_seconds};
    card.onclick=()=>openCollisionExample(sample);
    gallery.appendChild(card);
  }
}
function openCollisionExample(sample){
  if(!sample){collisionStatus.textContent='Select a collision example first.';return}
  sourceSelectionVersion++;
  selectedFile=null;
  currentSample=null;
  currentSampleKey=null;
  pendingSample=null;
  if(objectUrl){URL.revokeObjectURL(objectUrl);objectUrl=null}
  document.getElementById('sampleSelect').value='';
  document.getElementById('saveExampleSetupBtn').disabled=true;
  document.getElementById('reviewHintBtn').style.display='none';
  document.getElementById('sampleDescription').textContent=sample.no_collision_control
    ? 'No-collision control. Check whether the model flags the selected interval.'
    : 'Example for collision review.';
  setVideoSource(sample.media_url,sample.name,sample.path,'yolo');
  document.getElementById('collisionExampleSelect').value=sample.id;
  document.getElementById('collisionStart').value=String(sample.start_seconds);
  document.getElementById('collisionEnd').value=String(sample.end_seconds);
  collisionPreview.currentTime=sample.start_seconds;
  syncWorkspaceSource();
  showWorkspace('collision-step',true,true);
}
document.getElementById('collisionExampleBtn').onclick=()=>openCollisionExample(collisionExamples[document.getElementById('collisionExampleSelect').value]);
async function startCollisionReview(){const version=annotationSourceVersion,file=selectedFile;if(!serverVideoPath&&!file){collisionStatus.textContent='Load a source video first.';return}if(currentSample?.evaluationCandidate&&!groundTruthFinalized){collisionStatus.textContent='Complete the independent annotation of this evaluation video before viewing model predictions.';return}const start=Number(document.getElementById('collisionStart').value),end=Number(document.getElementById('collisionEnd').value);if(!Number.isFinite(start)||!Number.isFinite(end)||start<0||end-start<2||end-start>30){collisionStatus.textContent='Choose an interval of 2 to 30 seconds inside the recording.';return}document.getElementById('collisionRunBtn').disabled=true;document.getElementById('collisionResults').replaceChildren();document.getElementById('collisionProgress').style.width='0%';try{let path=serverVideoPath;if(!path){collisionStatus.textContent='Importing source video...';path=await uploadFile(file);if(version!==annotationSourceVersion)return;serverVideoPath=path}const response=await fetch('/api/review-collisions',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({video_path:path,start_seconds:start,end_seconds:end,device:document.getElementById('collisionDevice').value})}),data=await response.json();if(version!==annotationSourceVersion)return;if(!response.ok)throw Error(data.error||'Could not start collision review');collisionJobId=data.job_id;pollCollisionReview(collisionJobId,version)}catch(error){if(version===annotationSourceVersion){collisionStatus.textContent=error.message;document.getElementById('collisionRunBtn').disabled=false}}}
async function pollCollisionReview(id,version){if(id!==collisionJobId||version!==annotationSourceVersion)return;try{const response=await fetch('/api/status?id='+encodeURIComponent(id)),job=await response.json();if(id!==collisionJobId||version!==annotationSourceVersion)return;if(!response.ok||job.status==='error'||job.kind!=='collision_review')throw Error(job.error||'Could not read collision-review status');collisionStatus.textContent=job.status==='queued'?'Queued behind the current analysis.':`${job.stage||'Reviewing'} · ${job.current_window||0}/${job.total_windows||'?'} windows`;document.getElementById('collisionProgress').style.width=(job.total_windows?100*job.current_window/job.total_windows:0)+'%';if(job.status!=='complete'){setTimeout(()=>pollCollisionReview(id,version),800);return}document.getElementById('collisionRunBtn').disabled=false;document.getElementById('collisionProgress').style.width='100%';collisionStatus.textContent='Offline collision review complete.';renderCollisionResults(job.summary,id)}catch(error){if(id===collisionJobId&&version===annotationSourceVersion){collisionStatus.textContent=error.message;document.getElementById('collisionRunBtn').disabled=false}}}
function renderCollisionResults(summary,id){const host=document.getElementById('collisionResults');host.replaceChildren();const info=document.createElement('p');info.className='collision-summary';info.textContent=`${summary.candidate_window_count} candidate window(s), ${summary.uncertain_window_count} uncertain window(s). Processing: ${Number(summary.processing_seconds).toFixed(1)} s. Overlapping windows can refer to the same incident.`;host.appendChild(info);const labels={A:'normal movement',B:'visible impact candidate',C:'insufficient evidence'};for(const item of summary.windows){if(item.decision==='no_collision_observed')continue;const card=document.createElement('article');card.className='collision-window '+(item.decision==='collision_review_candidate'?'candidate':'uncertain');const title=document.createElement('h3');title.textContent=`${item.decision==='collision_review_candidate'?'Possible collision':'Uncertain interaction'} · ${formatTime(item.start_seconds)} – ${formatTime(item.end_seconds)}`;card.appendChild(title);const detail=document.createElement('p');detail.textContent=`Full scene: ${labels[item.full.label]||'unresolved'}. Local interaction: ${labels[item.local.label]||'unresolved'}. No exact impact frame or legal fault is inferred.${item.quality_flags?.length?' Source quality: possible camera movement or scene cut.':''}`;card.appendChild(detail);const button=document.createElement('button');button.textContent='Seek to interval start';button.onclick=()=>{collisionPreview.pause();collisionPreview.currentTime=item.start_seconds;collisionPreview.scrollIntoView({behavior:'smooth',block:'center'})};card.appendChild(button);if(item.storyboard){const image=document.createElement('img');image.alt='Original source frames with timestamps for human collision review';image.src='/api/collision-review-asset?id='+encodeURIComponent(id)+'&name='+encodeURIComponent(item.storyboard);card.appendChild(image)}host.appendChild(card)}if(!summary.candidate_window_count&&!summary.uncertain_window_count){const empty=document.createElement('p');empty.className='small-note';empty.textContent='No collision was identified in the sampled frames. This does not prove that no collision occurred.';host.appendChild(empty)}const link=document.createElement('a');link.className='button';link.textContent='Download review JSON';link.download='collision-review.json';link.href='/api/collision-review-asset?id='+encodeURIComponent(id)+'&name=collision_review.json';host.appendChild(link)}
document.getElementById('collisionRunBtn').onclick=startCollisionReview;
new MutationObserver(syncCollisionSource).observe(fileName,{childList:true,characterData:true,subtree:true});
loadCollisionExamples();syncCollisionSource();
</script></body></html>"""


HTML = HTML.replace("</style>", STUDY_STYLE + "\n</style>", 1)
HTML = HTML.replace('<div class="readiness-grid" id="readinessMetrics">',
                    STUDY_HTML + '<details class="advanced-panel"><summary>Advanced detector comparison</summary>'
                    '<div class="readiness-grid" id="readinessMetrics">', 1)
HTML = HTML.replace("<h3>Recent experiment history</h3>", "</details><h3>Recent experiment history</h3>", 1)
HTML = HTML.replace("</script>", STUDY_SCRIPT + "\n</script>", 1)


class RequestHandler(BaseHTTPRequestHandler):
    state: AppState

    def log_message(self, format: str, *args) -> None:
        return

    def send_json(self, payload: dict, status: int = 200) -> None:
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path.startswith("/api/study/"):
            self.handle_study_get(parsed)
            return
        if parsed.path == "/api/compute":
            self.send_json(compute_status())
            return
        if parsed.path == "/":
            data = HTML.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
            return
        if parsed.path == "/media/sample":
            self.serve_file(SAMPLE_VIDEO)
            return
        if parsed.path.startswith('/media/collision-example/'):
            item=COLLISION_EXAMPLES.get(parsed.path.removeprefix('/media/collision-example/'))
            if item is None:
                self.send_error(HTTPStatus.NOT_FOUND)
            else:
                self.serve_file(item[1])
            return
        if parsed.path == '/api/collision-examples':
            from .collision_review import model_available
            self.send_json({'model_available':model_available(),'examples':[
                {'id':key,'name':name,'path':path.relative_to(PROJECT_ROOT).as_posix(),
                 'media_url':'/media/collision-example/'+quote(key),
                 'start_seconds':start,'end_seconds':end,'no_collision_control':control}
                for key,(name,path,start,end,control) in COLLISION_EXAMPLES.items() if path.is_file()]})
            return
        if parsed.path == '/api/collision-review-asset':
            query=parse_qs(parsed.query)
            job=self.state.get_job(query.get('id',[''])[0]);name=query.get('name',[''])[0]
            if not job or job.get('kind')!='collision_review' or not re.fullmatch(r'(?:window-\d{3}\.jpg|collision_review\.json)',name):
                self.send_error(HTTPStatus.NOT_FOUND)
            else:
                self.serve_file(Path(job['output_dir'])/name)
            return
        if parsed.path.startswith("/media/example/"):
            example_id = parsed.path.removeprefix("/media/example/")
            example_path = REAL_EXAMPLE_VIDEOS.get(example_id)
            if example_path is None:
                self.send_error(HTTPStatus.NOT_FOUND)
            else:
                self.serve_file(example_path)
            return
        if parsed.path == "/api/camera-profiles":
            self.send_json({"profiles": load_camera_profiles()})
            return
        if parsed.path == "/api/camera-profile":
            try:
                profile_id = safe_name(parse_qs(parsed.query).get("id", [""])[0]).lower()
                if not profile_id or profile_id == "video":
                    raise ValueError("Select a camera profile")
                path = CAMERA_PROFILE_DIR / f"{profile_id}.json"
                if not path.is_file():
                    self.send_json({"error": "Camera profile not found"}, 404)
                    return
                self.send_json({"profile": json.loads(path.read_text(encoding="utf-8"))})
            except (ValueError, OSError, json.JSONDecodeError) as exc:
                self.send_json({"error": str(exc)}, 400)
            return
        if parsed.path == "/api/example-setup":
            try:
                example_key = safe_name(parse_qs(parsed.query).get("id", [""])[0]).lower()
                if not example_key or example_key == "video":
                    raise ValueError("Select an example")
                path = EXAMPLE_SETUP_DIR / f"{example_key}.json"
                if not path.is_file():
                    self.send_json({"error": "No saved setup for this example"}, 404)
                    return
                self.send_json({"profile": json.loads(path.read_text(encoding="utf-8"))})
            except (ValueError, OSError, json.JSONDecodeError) as exc:
                self.send_json({"error": str(exc)}, 400)
            return
        if parsed.path == "/api/dataset-candidates":
            candidates = []
            for item in load_evaluation_candidates():
                candidates.append({
                    key: value for key, value in item.items() if key != "resolved_path"
                } | {"media_url": f"/media/candidate/{quote(str(item['id']))}",
                     "poster_url": f"/media/candidate-poster/{quote(str(item['id']))}" if item.get('poster_path') else None})
            builtins = {"real": SAMPLE_VIDEO, **REAL_EXAMPLE_VIDEOS}
            self.send_json({
                "candidates": candidates,
                "available_builtins": [key for key, path in builtins.items() if path.is_file()],
            })
            return
        if parsed.path == "/api/research-readiness":
            active_campaign = DATA_DIR / "accuracy_campaign.active.json"
            campaign_path = active_campaign if active_campaign.is_file() else DATA_DIR / "accuracy_campaign.template.json"
            try:
                campaign = validate_campaign(campaign_path)
                campaign_summary = {key: value for key, value in campaign.items() if key != "manifest"}
            except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
                campaign_summary = {"ready": False, "blockers": [str(exc)], "warnings": []}
            review_files = list(ANNOTATION_DIR.glob("*.review.json")) if ANNOTATION_DIR.is_dir() else []
            completed_sources = {}
            for review_file in sorted(review_files):
                try:
                    review = json.loads(review_file.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError):
                    continue
                identity = review.get('source_sha256') or review.get('video_path')
                if review.get("review_complete") and identity:
                    completed_sources[identity] = review
            self.send_json({
                "release": RELEASE_VERSION,
                "accepted_candidate_videos": len(load_evaluation_candidates()),
                "completed_full_video_reviews": len(completed_sources),
                "completed_negative_reviews": sum(bool(review.get('negative_video')) for review in completed_sources.values()),
                "campaign_source": campaign_path.resolve().relative_to(PROJECT_ROOT.resolve()).as_posix(),
                "campaign": campaign_summary,
            })
            return
        if parsed.path == "/api/annotation-draft":
            try:
                relative = parse_qs(parsed.query).get("video_path", [""])[0]
                video_path = (PROJECT_ROOT / relative).resolve()
                if not relative or not is_within(video_path, DATA_DIR) or not video_path.is_file():
                    raise ValueError("The draft video must be inside the project data directory")
                _source_hash, identity = _annotation_identity(video_path)
                draft_path = ANNOTATION_DRAFT_DIR / f"{identity}.draft.json"
                draft = json.loads(draft_path.read_text(encoding="utf-8")) if draft_path.is_file() else None
                self.send_json({"draft": draft})
            except (ValueError, OSError, json.JSONDecodeError) as exc:
                self.send_json({"error": str(exc)}, 400)
            return
        if parsed.path == "/api/video-metadata":
            try:
                relative = parse_qs(parsed.query).get("video_path", [""])[0]
                video_path = (PROJECT_ROOT / relative).resolve()
                if not relative or not is_within(video_path, DATA_DIR) or not video_path.is_file():
                    raise ValueError("The video must be inside the project data directory")
                capture = cv2.VideoCapture(str(video_path))
                fps = float(capture.get(cv2.CAP_PROP_FPS) or 0)
                frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
                capture.release()
                if not math.isfinite(fps) or fps <= 0:
                    raise ValueError("Video frame rate is unavailable")
                self.send_json({
                    "fps": round(fps, 6), "frames": frames,
                    "duration_seconds": round(frames / fps, 6) if frames else 0,
                })
            except (ValueError, OSError) as exc:
                self.send_json({"error": str(exc)}, 400)
            return
        if parsed.path == "/api/run-history":
            runs = []
            runs_root = PROJECT_ROOT / "runs"
            for summary_path in sorted(
                runs_root.glob("*/summary.json") if runs_root.is_dir() else [],
                key=lambda path: path.stat().st_mtime,
                reverse=True,
            )[:20]:
                try:
                    summary = json.loads(summary_path.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError):
                    continue
                runs.append({
                    "run_name": summary_path.parent.name,
                    "input_video": Path(summary.get("input_video", "unknown")).name,
                    "model": str(summary.get("model", "unknown")),
                    "tracker": str(summary.get("tracker", "unknown")),
                    "candidate_events": int(summary.get("candidate_events", 0)),
                    "candidate_incidents": int(summary.get("candidate_incidents", summary.get("candidate_events", 0))),
                    "processing_fps": summary.get("processing_fps"),
                    "real_time_factor": summary.get("real_time_factor"),
                    "completed_at": datetime.fromtimestamp(summary_path.stat().st_mtime).isoformat(timespec="seconds"),
                })
            self.send_json({"runs": runs})
            return
        if parsed.path.startswith('/media/candidate-poster/'):
            candidate_id = parsed.path.removeprefix('/media/candidate-poster/')
            item = next((entry for entry in load_evaluation_candidates() if entry['id'] == candidate_id), None)
            poster = (PROJECT_ROOT/str(item.get('poster_path', ''))).resolve() if item else None
            if poster is None or not is_within(poster, DATA_DIR) or not poster.is_file():
                self.send_error(HTTPStatus.NOT_FOUND)
            else:
                self.serve_file(poster)
            return
        if parsed.path.startswith("/media/candidate/"):
            candidate_id = parsed.path.removeprefix("/media/candidate/")
            item = next((entry for entry in load_evaluation_candidates() if entry["id"] == candidate_id), None)
            if item is None:
                self.send_error(HTTPStatus.NOT_FOUND)
            else:
                self.serve_file(item["resolved_path"])
            return
        if parsed.path == "/api/status":
            job_id = parse_qs(parsed.query).get("id", [""])[0]
            job = self.state.get_job(job_id)
            if not job:
                self.send_json({"error": "Job not found"}, 404)
            else:
                self.send_json(job)
            return
        if parsed.path in {"/api/events", "/api/evidence"}:
            query = parse_qs(parsed.query)
            job = self.state.get_job(query.get("id", [""])[0])
            if not job or job["status"] != "complete":
                self.send_json({"error": "Completed run required"}, 404)
                return
            if parsed.path == "/api/events":
                incidents_path = Path(job["summary"]["outputs"].get("incidents", ""))
                incidents = []
                if incidents_path.is_file() and is_within(incidents_path, Path(job["output_dir"])):
                    incidents = json.loads(incidents_path.read_text(encoding="utf-8"))
                self.send_json({
                    "events": read_jsonl(Path(job["summary"]["outputs"]["events"])),
                    "incidents": incidents,
                })
            else:
                root = Path(job["output_dir"]) / "evidence"
                path = root / query.get("name", [""])[0]
                if not is_within(path, root) or path.suffix.lower() not in {".jpg", ".webm"}:
                    self.send_json({"error": "Invalid evidence path"}, 400)
                else:
                    self.serve_file(path)
            return
        if parsed.path == "/api/evidence-bundle":
            job_id = parse_qs(parsed.query).get("id", [""])[0]
            job = self.state.get_job(job_id)
            if not job or job["status"] != "complete":
                self.send_json({"error": "Evidence package not ready"}, 404)
                return
            bundle_path = Path(job["summary"]["outputs"].get("evidence_bundle", ""))
            output_dir = Path(job["output_dir"])
            if not bundle_path.is_file() or not is_within(bundle_path, output_dir) or bundle_path.suffix.lower() != ".zip":
                self.send_json({"error": "Verified evidence package is unavailable"}, 404)
                return
            self.serve_file(bundle_path)
            return
        if parsed.path == "/api/result-video":
            job_id = parse_qs(parsed.query).get("id", [""])[0]
            job = self.state.get_job(job_id)
            if not job or job["status"] != "complete":
                self.send_json({"error": "Result not ready"}, 404)
                return
            annotated_video = Path(job["summary"]["outputs"]["annotated_video"])
            output_dir = Path(job["output_dir"])
            if not is_within(annotated_video, output_dir):
                self.send_json({"error": "Invalid result path"}, 400)
                return
            self.serve_file(annotated_video)
            return
        if parsed.path == "/api/run-report":
            job_id = parse_qs(parsed.query).get("id", [""])[0]
            job = self.state.get_job(job_id)
            if not job or job["status"] != "complete":
                self.send_json({"error": "Run report not ready"}, 404)
                return
            report_path = Path(job["summary"]["outputs"]["run_report"])
            output_dir = Path(job["output_dir"])
            if not is_within(report_path, output_dir):
                self.send_json({"error": "Invalid report path"}, 400)
                return
            self.serve_file(report_path)
            return
        self.send_error(HTTPStatus.NOT_FOUND)

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path.startswith("/api/study/"):
            self.handle_study_post(parsed)
            return
        if parsed.path == "/api/camera-profile":
            self.handle_camera_profile()
            return
        if parsed.path == "/api/example-setup":
            self.handle_example_setup()
            return
        if parsed.path == "/api/suggest-lane":
            self.handle_suggest_lane()
            return
        if parsed.path == "/api/suggest-forward-lane":
            self.handle_suggest_forward_lane()
            return
        if parsed.path == "/api/analyze-intersection":
            self.handle_analyze_intersection()
            return

        if parsed.path == "/api/validate-road-geometry":
            self.handle_validate_road_geometry()
            return
        if parsed.path == "/api/detect-label-frame":
            self.handle_detect_label_frame()
            return
        if parsed.path == "/api/annotation-draft":
            self.handle_annotation_draft()
            return
        if parsed.path == "/api/annotation-quality":
            try:
                length = int(self.headers.get("Content-Length", "0"))
                payload = json.loads(self.rfile.read(length).decode("utf-8"))
                labels = payload.get("labels", [])
                if not isinstance(labels, list):
                    raise ValueError("Labels must be a list")
                issues = validate_annotation_quality(
                    labels, float(payload.get("duration_seconds", 0)), payload.get("lanes")
                )
                self.send_json({
                    "issues": issues,
                    "error_count": sum(issue["severity"] == "error" for issue in issues),
                    "warning_count": sum(issue["severity"] == "warning" for issue in issues),
                    "passed": not any(issue["severity"] == "error" for issue in issues),
                })
            except (ValueError, TypeError, json.JSONDecodeError) as exc:
                self.send_json({"error": str(exc)}, 400)
            return
        if parsed.path == "/api/evaluate":
            try:
                payload = json.loads(self.rfile.read(int(self.headers.get("Content-Length", "0"))))
                job = self.state.get_job(payload["job_id"])
                if not job or job["status"] != "complete":
                    raise ValueError("Complete an analysis before evaluating.")
                truth = payload["labels"]
                if not isinstance(truth, list) or not truth:
                    raise ValueError("Add confirmed vehicle labels first.")
                if payload.get("video_path") != str(Path(job["video_path"]).relative_to(PROJECT_ROOT).as_posix()):
                    raise ValueError("Labels must refer to the same video as the selected run.")
                result = evaluate_events(read_jsonl(Path(job["summary"]["outputs"]["events"])), truth)
                for kind in sorted({x["candidate_event"] for x in truth} | set(job["summary"]["candidate_events_by_type"])):
                    result.setdefault("by_type", {})[kind] = evaluate_events(
                        [x for x in read_jsonl(Path(job["summary"]["outputs"]["events"])) if x["candidate_event"] == kind],
                        [x for x in truth if x["candidate_event"] == kind])
                result["note"] = "Metrics cover supplied labels only. Label every event before reporting recall."
                destination = Path(job["output_dir"]) / ("evaluation-" + uuid.uuid4().hex[:8] + ".json")
                destination.write_text(json.dumps({"metrics": result, "ground_truth": truth}, indent=2), encoding="utf-8")
                self.send_json(result)
            except (ValueError, KeyError, TypeError) as exc:
                self.send_json({"error": str(exc)}, 400)
            return
        if parsed.path == "/api/upload":
            self.handle_upload(parsed)
            return
        if parsed.path == "/api/run":
            self.handle_run()
            return
        if parsed.path == '/api/review-collisions':
            try:
                length=int(self.headers.get('Content-Length','0'))
                if length<=0 or length>16384:raise ValueError('Invalid collision-review request size.')
                payload=json.loads(self.rfile.read(length).decode('utf-8'))
                path=(PROJECT_ROOT/str(payload['video_path'])).resolve()
                if not is_within(path,DATA_DIR) or not path.is_file():
                    raise ValueError('Import the source video into the project data directory first.')
                job_id=self.state.new_collision_job(path,float(payload.get('start_seconds',0)),float(payload['end_seconds']),str(payload.get('device','cpu')))
                self.send_json({'job_id':job_id})
            except (ValueError,TypeError,KeyError,json.JSONDecodeError) as error:
                self.send_json({'error':str(error)},400)
            return
        if parsed.path == "/api/register-finalized-ground-truth":
            self.handle_register_finalized_ground_truth()
            return
        if parsed.path == "/api/save-ground-truth":
            self.handle_save_ground_truth()
            return
        if parsed.path == "/api/open-folder":
            job_id = parse_qs(parsed.query).get("id", [""])[0]
            job = self.state.get_job(job_id)
            if not job:
                self.send_json({"error": "Job not found"}, 404)
                return
            output_dir = Path(job["output_dir"])
            if output_dir.exists():
                os.startfile(output_dir)
            self.send_json({"ok": True})
            return
        self.send_error(HTTPStatus.NOT_FOUND)

    def handle_camera_profile(self) -> None:
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if length <= 0 or length > 2 * 1024 * 1024:
                raise ValueError("Camera profile payload is empty or too large")
            payload = json.loads(self.rfile.read(length).decode("utf-8"))
            profile = validate_camera_profile(payload)
            CAMERA_PROFILE_DIR.mkdir(parents=True, exist_ok=True)
            destination = CAMERA_PROFILE_DIR / f"{profile['profile_id']}.json"
            if destination.is_file():
                try:
                    previous = json.loads(destination.read_text(encoding="utf-8"))
                    profile["created_at"] = previous.get("created_at", profile["created_at"])
                except (OSError, json.JSONDecodeError):
                    pass
            atomic_write_json(destination, profile)
            self.send_json({
                "profile": profile,
                "path": destination.relative_to(PROJECT_ROOT).as_posix(),
            })
        except (ValueError, TypeError, OSError, json.JSONDecodeError) as exc:
            self.send_json({"error": str(exc)}, 400)

    def handle_example_setup(self) -> None:
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if length <= 0 or length > 2 * 1024 * 1024:
                raise ValueError("Example setup payload is empty or too large")
            payload = json.loads(self.rfile.read(length).decode("utf-8"))
            example_key_raw = str(payload.get("example_key", "")).strip()
            if not example_key_raw:
                raise ValueError("Select an example before saving its setup")
            example_key = safe_name(example_key_raw).lower()
            profile = validate_camera_profile(payload.get("profile"))
            profile["example_key"] = example_key
            EXAMPLE_SETUP_DIR.mkdir(parents=True, exist_ok=True)
            destination = EXAMPLE_SETUP_DIR / f"{example_key}.json"
            if destination.is_file():
                try:
                    previous = json.loads(destination.read_text(encoding="utf-8"))
                    profile["created_at"] = previous.get("created_at", profile["created_at"])
                except (OSError, json.JSONDecodeError):
                    pass
            atomic_write_json(destination, profile)
            self.send_json({
                "profile": profile,
                "path": destination.relative_to(PROJECT_ROOT).as_posix(),
            })
        except (ValueError, TypeError, OSError, json.JSONDecodeError) as exc:
            self.send_json({"error": str(exc)}, 400)

    def handle_detect_label_frame(self) -> None:
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if length <= 0 or length > 6 * 1024 * 1024:
                raise ValueError("Captured-frame payload is empty or too large")
            payload = json.loads(self.rfile.read(length).decode("utf-8"))
            proposals = detect_vehicle_proposals(
                str(payload["image_data_url"]),
                str(payload.get("model_name", "yolov8n")),
                str(payload.get("device", "auto")),
                float(payload.get("confidence", 0.15)),
            )
            self.send_json({
                "proposals": proposals,
                "proposal_count": len(proposals),
                "warning": "Detector proposals are selection aids only. Reviewers must add missed vehicles manually.",
            })
        except (KeyError, ValueError, TypeError, RuntimeError, OSError, json.JSONDecodeError, binascii.Error) as exc:
            self.send_json({"error": str(exc)}, 400)

    def handle_study_get(self, parsed) -> None:
        try:
            query = parse_qs(parsed.query)
            if parsed.path == "/api/study/diagnosis":
                from .diagnostics import diagnose_error
                track = query.get("track", [None])[0]
                self.send_json(diagnose_error(query.get("run", [""])[0], query.get("error", [""])[0],
                                             int(track) if track is not None else None))
                return
            if parsed.path == "/api/study/cases":
                registry = study.read_registry()
                cases = [{key: value for key, value in case.items() if key != "configuration"}
                         | {"issues": study.case_issues(case)} for case in registry["cases"]]
                runs = []
                for path in sorted((PROJECT_ROOT / "runs").glob("study-*/report.json"), reverse=True)[:30]:
                    try:
                        report = json.loads(path.read_text(encoding="utf-8"))
                        runs.append({"run_name": path.parent.name, "split": report["split"],
                                     "model": report["model"], "case_count": len(report["cases"])})
                    except (OSError, ValueError, KeyError):
                        continue
                self.send_json({"revision": registry["revision"], "cases": cases, "runs": runs})
                return
            if parsed.path not in {"/api/study/report", "/api/study/asset", "/api/study/source"}:
                raise ValueError("Unknown study resource.")
            directory = study.report_directory(query.get("run", [""])[0])
            if not (directory / "report.json").is_file():
                raise ValueError("The study has no completed report.")
            if parsed.path == "/api/study/report":
                self.send_json(json.loads((directory / "report.json").read_text(encoding="utf-8")))
            elif parsed.path == "/api/study/asset":
                name = query.get("name", [""])[0]
                if name in {"report.json", "cases.csv", "errors.csv"}:
                    path = directory / name
                elif re.fullmatch(r"error-\d{5,}(?:(?:-crop)?\.jpg|-clip\.webm)", name):
                    path = directory / "evidence" / name
                else:
                    raise ValueError("Unknown study asset.")
                self.serve_file(path)
            else:
                inputs = json.loads((directory / "study_inputs.json").read_text(encoding="utf-8"))
                case = next((c for c in inputs["cases"] if c["id"] == query.get("case", [""])[0]), None)
                if case is None:
                    raise ValueError("Study case not found.")
                path = study.data_path(case["input_video"])
                if study.file_hash(path) != case["source_sha256"]:
                    raise ValueError("The original recording changed after this study.")
                self.serve_file(path)
        except (OSError, ValueError, TypeError, KeyError) as error:
            self.send_json({"error": str(error)}, 400)

    def handle_study_post(self, parsed) -> None:
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if not 0 < length <= 2 * 1024 * 1024:
                raise ValueError("Invalid study request size.")
            payload = json.loads(self.rfile.read(length).decode("utf-8"))
            if not isinstance(payload, dict):
                raise ValueError("A study request must be a JSON object.")
            if parsed.path == "/api/study/remove":
                study.remove_case(str(payload["case_id"]), int(payload["revision"]))
                self.send_json({"removed": True})
            elif parsed.path == "/api/study/run":
                job_id = self.state.new_study_job(str(payload["split"]), str(payload["model"]),
                                                  str(payload["tracker"]), str(payload.get("device", "auto")))
                self.send_json({"job_id": job_id})
            elif parsed.path == "/api/study/diagnosis-replay":
                job_id = self.state.new_diagnosis_job(str(payload["run"]), str(payload["error"]),
                                                       str(payload.get("device", "auto")))
                self.send_json({"job_id": job_id})
            elif parsed.path == "/api/study/cases":
                video = study.data_path(str(payload["video_path"]))
                _digest, identity = _annotation_identity(video)
                draft = json.loads((ANNOTATION_DRAFT_DIR / f"{identity}.draft.json").read_text(encoding="utf-8"))
                if not draft.get("review_complete") or not draft.get("final_ground_truth"):
                    raise ValueError("Finalize this video's original-footage review in Ground truth first.")
                truth = study.data_path(str(draft["final_ground_truth"]))
                if not is_within(truth, ANNOTATION_DIR):
                    raise ValueError("Final ground truth must be inside the annotation directory.")
                profile = validate_camera_profile(payload["profile"])
                first, inference = profile["lanes"][0], profile["inference"]
                config = create_config(
                    profile["camera_id"], first["polygon"], inference["confidence"],
                    signal_rectangle=first.get("signal_rectangle") or None,
                    direction=first.get("direction") or None, stop_line=first.get("stop_line") or None,
                    restricted_polygon=first.get("restricted_polygon") or None,
                    enabled_rules=first["enabled_rules"], image_size=inference["image_size"],
                    restricted_initial_presence=bool(first.get("restricted_initial_presence", False)),
                    lanes=profile["lanes"], model_name=inference["model_name"], tracker_name=inference["tracker_name"],
                )
                case = study.register_case(video, truth, config, name=payload["name"],
                                           source_group=payload["source_group_id"], camera_id=payload["camera_id"],
                                           split=payload["split"], expected_revision=int(payload["revision"]))
                self.send_json({"case_id": case["id"]})
            else:
                raise ValueError("Unknown study action.")
        except (OSError, ValueError, TypeError, KeyError) as error:
            self.send_json({"error": str(error)}, 400)

    def handle_register_finalized_ground_truth(self) -> None:
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if length <= 0 or length > 1024 * 1024:
                raise ValueError("Campaign registration payload is empty or too large")
            payload = json.loads(self.rfile.read(length).decode("utf-8"))
            relative = str(payload.get("video_path", ""))
            video_path = (PROJECT_ROOT / relative).resolve()
            if not relative or not is_within(video_path, DATA_DIR) or not video_path.is_file():
                raise ValueError("The campaign video must be inside the project data directory")
            job = self.state.get_job(str(payload.get("job_id", "")))
            if not job or Path(job.get("video_path", "")).resolve() != video_path:
                raise ValueError("Complete an analysis of this same video before campaign registration.")
            _source_hash, identity = _annotation_identity(video_path)
            draft_path = ANNOTATION_DRAFT_DIR / f"{identity}.draft.json"
            if not draft_path.is_file():
                raise ValueError("No finalized review was found for this video.")
            draft = json.loads(draft_path.read_text(encoding="utf-8"))
            if not draft.get("review_complete") or not draft.get("final_ground_truth"):
                raise ValueError("Finish and finalize the independent ground-truth review first.")
            ground_truth = (PROJECT_ROOT / str(draft["final_ground_truth"])).resolve()
            campaign_path = register_campaign_case(str(payload.get("camera_id", "")), job, ground_truth)
            self.send_json({"campaign_registration": campaign_path})
        except (KeyError, ValueError, TypeError, OSError, json.JSONDecodeError) as exc:
            self.send_json({"error": str(exc)}, 400)

    def handle_annotation_draft(self) -> None:
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if length <= 0 or length > 20 * 1024 * 1024:
                raise ValueError("Annotation draft is empty or too large")
            payload = json.loads(self.rfile.read(length).decode("utf-8"))
            relative = str(payload.get("video_path", ""))
            video_path = (PROJECT_ROOT / relative).resolve()
            if not relative or not is_within(video_path, DATA_DIR) or not video_path.is_file():
                raise ValueError("The draft video must be inside the project data directory")
            labels = payload.get("labels", [])
            if not isinstance(labels, list):
                raise ValueError("Draft labels must be a list")
            for label in labels:
                if not isinstance(label, dict) or label.get("status") not in {"confirmed", "uncertain"}:
                    raise ValueError("Every draft label must have a supported review status")
            duration = float(payload.get("duration_seconds", 0))
            intervals, coverage = _review_coverage(payload.get("reviewed_intervals", []), duration)
            source_hash, identity = _annotation_identity(video_path)
            draft = {
                "schema_version": 2,
                "video_name": str(payload.get("video_name", video_path.name)),
                "video_path": video_path.relative_to(PROJECT_ROOT).as_posix(),
                "source_sha256": source_hash,
                "reviewer": str(payload.get("reviewer", "")).strip()[:80],
                "mode": str(payload.get("mode", "ground_truth")),
                "labels": labels,
                "reviewed_intervals": intervals,
                "review_coverage_percent": round(coverage * 100, 3),
                "duration_seconds": round(duration, 3),
                "current_time_seconds": round(float(payload.get("current_time_seconds", 0)), 3),
                "updated_at": datetime.now().isoformat(timespec="seconds"),
                "review_complete": False,
            }
            ANNOTATION_DRAFT_DIR.mkdir(parents=True, exist_ok=True)
            destination = ANNOTATION_DRAFT_DIR / f"{identity}.draft.json"
            with ANNOTATION_WRITE_LOCK:
                previous = json.loads(destination.read_text(encoding='utf-8')) if destination.is_file() else {}
                if (previous.get('review_complete') and draft['review_coverage_percent'] >= 98
                        and _draft_review_content(previous) == _draft_review_content(draft)):
                    for key in ('review_complete', 'final_ground_truth', 'finalized_at'):
                        if key in previous:
                            draft[key] = previous[key]
                atomic_write_json(destination, draft)
            self.send_json({
                "path": destination.relative_to(PROJECT_ROOT).as_posix(),
                "review_coverage_percent": draft["review_coverage_percent"],
                "review_complete": draft["review_complete"],
            })
        except (ValueError, TypeError, OSError, json.JSONDecodeError) as exc:
            self.send_json({"error": str(exc)}, 400)

    def handle_suggest_lane(self) -> None:
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if length <= 0 or length > 6 * 1024 * 1024:
                raise ValueError("Setup frame payload is empty or too large")
            payload = json.loads(self.rfile.read(length).decode("utf-8"))
            encoded = str(payload["image_data_url"])
            if not encoded.startswith("data:image/") or "," not in encoded:
                raise ValueError("A setup-frame image is required")
            raw = base64.b64decode(encoded.split(",", 1)[1], validate=True)
            frame = cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), cv2.IMREAD_COLOR)
            semantic = segment_road(frame) if payload.get("semantic_road") else None
            try:
                result = suggest_lane_corridor(frame, payload["direction"], semantic.mask if semantic else None)
            except ValueError as exc:
                if semantic:
                    self.send_json({
                        "error": str(exc),
                        "road_understanding": semantic.metadata(),
                        "road_overlay_data_url": road_overlay_data_url(semantic.mask),
                    }, 422)
                    return
                raise
            if semantic:
                result["road_understanding"] = semantic.metadata()
                result["road_overlay_data_url"] = road_overlay_data_url(semantic.mask)
            self.send_json(result)
        except (KeyError, ValueError, TypeError, RuntimeError, OSError, json.JSONDecodeError, binascii.Error) as exc:
            self.send_json({"error": str(exc)}, 400)

    def handle_suggest_forward_lane(self) -> None:
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if length <= 0 or length > 6 * 1024 * 1024:
                raise ValueError("Setup frame payload is empty or too large")
            payload = json.loads(self.rfile.read(length).decode("utf-8"))
            encoded = str(payload["image_data_url"])
            if not encoded.startswith("data:image/") or "," not in encoded:
                raise ValueError("A setup-frame image is required")
            raw = base64.b64decode(encoded.split(",", 1)[1], validate=True)
            frame = cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), cv2.IMREAD_COLOR)
            self.send_json(suggest_forward_lane(frame, payload["direction"]))
        except (KeyError, ValueError, TypeError, RuntimeError, OSError, json.JSONDecodeError, binascii.Error) as exc:
            self.send_json({"error": str(exc)}, 422)

    def handle_analyze_intersection(self) -> None:
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if length <= 0 or length > 128 * 1024:
                raise ValueError("Intersection-analysis request is empty or too large")
            payload = json.loads(self.rfile.read(length).decode("utf-8"))
            video_path = (PROJECT_ROOT / str(payload["video_path"])).resolve()
            if not is_within(video_path, DATA_DIR) or not video_path.is_file():
                raise ValueError("Video must be inside the project data directory")
            model_name = str(payload.get("model_name", "yolov8n")).lower().strip()
            model_path = MODEL_PATHS.get(model_name)
            if model_path is None or not model_path.is_file():
                raise ValueError("A supported local YOLO model is required")
            image_size = int(payload.get("image_size", 960))
            if image_size not in {640, 960, 1280}:
                raise ValueError("Image size must be 640, 960 or 1280")
            max_seconds = max(5.0, min(60.0, float(payload.get("max_seconds", 20.0))))
            result = analyze_intersection_video(
                video_path,
                str(model_path.resolve()),
                max(0.05, min(0.50, float(payload.get("confidence", 0.15)))),
                image_size,
                max_seconds,
                reference_seconds=float(payload.get("reference_seconds", 0.0)),
                learned_markings=bool(payload.get("learned_markings", True)),
            )
            self.send_json(result)
        except (KeyError, ValueError, TypeError, RuntimeError, OSError, json.JSONDecodeError) as exc:
            self.send_json({"error": str(exc)}, 422)

    def handle_validate_road_geometry(self) -> None:
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if length <= 0 or length > 6 * 1024 * 1024:
                raise ValueError("Setup frame payload is empty or too large")
            payload = json.loads(self.rfile.read(length).decode("utf-8"))
            encoded = str(payload["image_data_url"])
            if not encoded.startswith("data:image/") or "," not in encoded:
                raise ValueError("A setup-frame image is required")
            raw = base64.b64decode(encoded.split(",", 1)[1], validate=True)
            frame = cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), cv2.IMREAD_COLOR)
            if frame is None:
                raise ValueError("The setup frame could not be decoded")
            lanes = payload.get("lanes")
            if not isinstance(lanes, list) or not lanes:
                raise ValueError("At least one configured lane is required")
            result = segment_road(frame)
            checks = []
            for index, lane in enumerate(lanes):
                if not isinstance(lane, dict):
                    raise ValueError("Every lane must be an object")
                enabled = lane.get("enabled_rules") or {}
                lane_name = str(lane.get("name") or lane.get("id") or f"Lane {index + 1}")
                if any(bool(enabled.get(name)) for name in ("red_light", "wrong_way")):
                    check = assess_polygon_road_overlap(result.mask, lane.get("polygon") or [])
                    checks.append({"lane_id": lane.get("id"), "lane_name": lane_name, "geometry": "monitored_lane", **check})
            self.send_json({
                "road_understanding": result.metadata(),
                "road_overlay_data_url": road_overlay_data_url(result.mask),
                "geometry_checks": checks,
            })
        except (KeyError, ValueError, TypeError, RuntimeError, OSError, json.JSONDecodeError, binascii.Error) as exc:
            self.send_json({"error": str(exc)}, 400)

    def handle_upload(self, parsed) -> None:
        length = int(self.headers.get("Content-Length", "0"))
        if length <= 0:
            self.send_json({"error": "Empty upload"}, 400)
            return
        raw_name = parse_qs(parsed.query).get("filename", ["video.mp4"])[0]
        suffix = Path(raw_name).suffix.lower()
        if suffix not in {".mp4", ".avi", ".mov", ".mkv", ".webm"}:
            suffix = ".mp4"
        UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
        destination = UPLOAD_DIR / f"{safe_name(Path(raw_name).stem)}-{uuid.uuid4().hex[:12]}{suffix}"
        remaining = length
        with destination.open("wb") as stream:
            while remaining:
                chunk = self.rfile.read(min(1024 * 1024, remaining))
                if not chunk:
                    break
                stream.write(chunk)
                remaining -= len(chunk)
        if remaining:
            destination.unlink(missing_ok=True)
            self.send_json({"error": "Incomplete upload"}, 400)
            return
        relative = destination.resolve().relative_to(PROJECT_ROOT.resolve()).as_posix()
        self.send_json({"video_path": relative})

    def handle_run(self) -> None:
        try:
            length = int(self.headers.get("Content-Length", "0"))
            payload = json.loads(self.rfile.read(length).decode("utf-8"))
            video_path = (PROJECT_ROOT / payload["video_path"]).resolve()
            if not is_within(video_path, DATA_DIR) or not video_path.is_file():
                raise ValueError("Video must be inside the project data directory")
            backend_name = str(payload.get("backend", "yolo"))
            if backend_name != "yolo":
                raise ValueError("Unsupported tracking backend")
            job_id = self.state.new_job(
                video_path,
                payload["polygon"],
                payload["signal_rectangle"],
                payload["direction"],
                payload["stop_line"],
                payload.get("restricted_polygon", []),
                payload.get("enabled_rules", {"red_light": True}),
                payload.get("camera_id", video_path.stem),
                float(payload.get("confidence", 0.25)),
                int(payload.get("image_size", 960)),
                backend_name,
                bool(payload.get("restricted_initial_presence", False)),
                payload.get("lane_assist"),
                str(payload.get("tracker_name", "bytetrack")),
                payload.get("lanes"),
                str(payload.get("model_name", "yolov8n")),
            )
            self.send_json({"job_id": job_id})
        except (KeyError, ValueError, TypeError, json.JSONDecodeError) as exc:
            self.send_json({"error": str(exc)}, 400)

    def handle_save_ground_truth(self) -> None:
        try:
            length = int(self.headers.get("Content-Length", "0"))
            payload = json.loads(self.rfile.read(length).decode("utf-8"))
            raw_annotations = payload["annotations"]
            review_complete = bool(payload.get("review_complete", False))
            reviewer = str(payload.get("reviewer", "")).strip()[:80]
            if not isinstance(raw_annotations, list):
                raise ValueError("Annotations must be a list")
            if not raw_annotations and not review_complete:
                raise ValueError("At least one confirmed annotation is required unless the video is fully reviewed as negative")
            if review_complete and not reviewer:
                raise ValueError("A reviewer name or ID is required for completed ground truth")
            relative_video = str(payload.get("video_path", ""))
            video_path = (PROJECT_ROOT / relative_video).resolve()
            if not relative_video or not is_within(video_path, DATA_DIR) or not video_path.is_file():
                raise ValueError("Completed ground truth must reference a video inside the project data directory")
            duration = float(payload.get("duration_seconds", 0))
            reviewed_intervals, review_coverage = _review_coverage(
                payload.get("reviewed_intervals", []), duration
            )
            if review_complete and review_coverage < MINIMUM_REVIEW_COVERAGE:
                raise ValueError(
                    f"Measured full-video review coverage is {review_coverage * 100:.1f}%; "
                    f"at least {MINIMUM_REVIEW_COVERAGE * 100:.0f}% is required"
                )
            source_hash, annotation_identity = _annotation_identity(video_path)
            quality_issues = validate_annotation_quality(
                raw_annotations, duration, payload.get("lanes")
            )
            blocking_quality = [issue["message"] for issue in quality_issues if issue["severity"] == "error"]
            if blocking_quality:
                raise ValueError("Annotation quality check failed: " + " ".join(blocking_quality))
            annotations = []
            allowed_events = {
                "candidate_red_signal_violation",
                "candidate_wrong_way_driving",
                "candidate_restricted_zone_entry",
            }
            for item in raw_annotations:
                timestamp_seconds = float(item["timestamp_seconds"])
                if not math.isfinite(timestamp_seconds) or timestamp_seconds < 0:
                    raise ValueError("Annotation timestamps must be finite and non-negative")
                bbox = [float(value) for value in item["vehicle_bbox_normalized"]]
                point = [float(value) for value in item["vehicle_point_normalized"]]
                if len(bbox) != 4 or len(point) != 2 or any(not 0.0 <= value <= 1.0 for value in bbox + point):
                    raise ValueError("Vehicle coordinates must be normalized from 0 to 1")
                if bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
                    raise ValueError("Vehicle bounding boxes must have a positive area")
                candidate_event = str(item.get("candidate_event", ""))
                if candidate_event not in allowed_events:
                    raise ValueError("Unsupported violation type")
                lane_id = safe_name(str(item.get("lane_id", "lane-1"))).lower()
                lane_name = str(item.get("lane_name", lane_id)).strip()[:60] or lane_id
                annotations.append(
                    {
                        "timestamp_seconds": round(timestamp_seconds, 3),
                        "lane_id": lane_id,
                        "lane_name": lane_name,
                        "candidate_event": candidate_event,
                        "vehicle_bbox_normalized": [round(value, 6) for value in bbox],
                        "vehicle_point_normalized": [round(value, 6) for value in point],
                        "vehicle_class": str(item.get("vehicle_class", "vehicle"))[:30],
                        "selection_method": str(item.get("selection_method", "manual_box"))[:60],
                        "description": str(item.get("description", "Confirmed traffic violation")),
                        "annotation_status": "confirmed",
                    }
                )
            annotations.sort(key=lambda item: item["timestamp_seconds"])
            stem = safe_name(Path(str(payload.get("video_name", video_path.name))).stem)
            timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            ANNOTATION_DIR.mkdir(parents=True, exist_ok=True)
            destination = ANNOTATION_DIR / f"{stem}_ground_truth_{timestamp}_{uuid.uuid4().hex[:8]}.json"
            atomic_write_json(destination, annotations)
            review_path = destination.with_suffix(".review.json")
            atomic_write_json(review_path, {
                "video_name": str(payload.get("video_name", "video")),
                "video_path": video_path.relative_to(PROJECT_ROOT).as_posix(),
                "source_sha256": source_hash,
                "reviewer": reviewer,
                "review_complete": review_complete,
                "reviewed_intervals": reviewed_intervals,
                "review_coverage_percent": round(review_coverage * 100, 3),
                "duration_seconds": round(duration, 3),
                "confirmed_events": len(annotations),
                "negative_video": review_complete and not annotations,
                "review_note": str(payload.get("review_note", "")),
                "saved_at": datetime.now().isoformat(timespec="seconds"),
            })
            response = {
                "path": destination.resolve().relative_to(PROJECT_ROOT.resolve()).as_posix(),
                "review_metadata": review_path.resolve().relative_to(PROJECT_ROOT.resolve()).as_posix(),
                "review_coverage_percent": round(review_coverage * 100, 3),
                "source_sha256": source_hash,
                "quality_warnings": [issue for issue in quality_issues if issue["severity"] == "warning"],
            }
            if bool(payload.get("register_campaign", False)):
                job = self.state.get_job(str(payload.get("job_id", "")))
                if not job or job.get("status") != "complete":
                    response["campaign_registration"] = "Complete an analysis with the frozen camera geometry before campaign registration."
                else:
                    try:
                        response["campaign_registration"] = register_campaign_case(
                            str(payload.get("camera_id", "")), job, destination
                        )
                    except ValueError as exc:
                        response["campaign_registration"] = str(exc)
            draft_path = ANNOTATION_DRAFT_DIR / f"{annotation_identity}.draft.json"
            if review_complete:
                try:
                    with ANNOTATION_WRITE_LOCK:
                        draft = json.loads(draft_path.read_text(encoding="utf-8")) if draft_path.is_file() else {}
                        final_draft = dict(draft, schema_version=2,
                            video_name=str(payload.get('video_name', video_path.name)),
                            video_path=video_path.relative_to(PROJECT_ROOT).as_posix(),
                            source_sha256=source_hash, reviewer=reviewer,
                            duration_seconds=round(duration, 3), reviewed_intervals=reviewed_intervals,
                            review_coverage_percent=round(review_coverage*100, 3),
                            labels=[dict(item, status='confirmed') for item in raw_annotations],
                            review_complete=True,
                            final_ground_truth=destination.relative_to(PROJECT_ROOT).as_posix(),
                            finalized_at=datetime.now().isoformat(timespec='seconds'))
                        if _draft_review_content(draft) == _draft_review_content(final_draft):
                            final_draft['labels'] = draft['labels']
                        atomic_write_json(draft_path, final_draft)
                except (OSError, json.JSONDecodeError) as exc:
                    response['draft_warning'] = f'Ground truth saved, but draft synchronization failed: {exc}'
            self.send_json(response)
        except (KeyError, ValueError, TypeError, json.JSONDecodeError) as exc:
            self.send_json({"error": str(exc)}, 400)

    def serve_file(self, path: Path) -> None:
        if not path.is_file():
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        size = path.stat().st_size
        start, end = 0, size - 1
        range_header = self.headers.get("Range")
        status = 200
        if range_header:
            match = re.match(r"bytes=(\d*)-(\d*)", range_header)
            if match:
                if match.group(1):
                    start = int(match.group(1))
                if match.group(2):
                    end = min(int(match.group(2)), size - 1)
                status = 206
        length = max(0, end - start + 1)
        self.send_response(status)
        self.send_header("Content-Type", mimetypes.guess_type(path.name)[0] or "application/octet-stream")
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Length", str(length))
        if status == 206:
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        self.end_headers()
        with path.open("rb") as stream:
            stream.seek(start)
            remaining = length
            while remaining:
                chunk = stream.read(min(1024 * 1024, remaining))
                if not chunk:
                    break
                try:
                    self.wfile.write(chunk)
                except (BrokenPipeError, ConnectionAbortedError, ConnectionResetError):
                    # Browsers cancel range requests while seeking or closing a video.
                    break
                remaining -= len(chunk)


def make_handler(state: AppState):
    class BoundHandler(RequestHandler):
        pass

    BoundHandler.state = state
    return BoundHandler


def main() -> None:
    parser = argparse.ArgumentParser(description="Local browser UI for the traffic event prototype")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--no-browser", action="store_true")
    parser.add_argument("--smoke-test", action="store_true")
    args = parser.parse_args()
    if args.smoke_test:
        assert "roiCanvas" in HTML and "labelCanvas" in HTML and "Start analysis" in HTML
        assert 'id="laneMode"' in HTML and "Add as many vertices as needed" in HTML
        assert "Analyze road + signals" in HTML and "/api/analyze-intersection" in HTML
        assert "Try lane-line model" in HTML and "/api/suggest-forward-lane" in HTML
        assert 'id="zoneMode"' in HTML and "human_vertex_refinement" in HTML
        assert "sampleSelect" in HTML and "Real fixed-camera development sample" in HTML
        assert "caseGallery" in HTML and "LABELLED DEVELOPMENT REPLAY" in HTML and "3 rules" in HTML
        assert "Tracking continuity" in HTML and "conservative ID-switch stitches" in HTML
        assert "Two-carriageway wrong-way preset" not in HTML
        assert "/media/example/wrong-way-two-lane" not in HTML
        assert "/api/validate-road-geometry" in HTML
        assert "/media/example/red-light-multi" in HTML
        assert "/media/example/idiap-late-stop" not in HTML
        assert 'id="noStopRule"' not in HTML and 'value="candidate_failure_to_stop"' not in HTML
        assert "/media/example/idiap-pedestrian-area" in HTML
        assert "/media/example/zenodo-video4-stress" in HTML and "Labelled pickup matched at 11.100 s" in HTML
        assert "FETV:" not in HTML and "motorcycles" in HTML
        assert "Synthetic:" not in HTML and "/media/builtin/" not in HTML
        assert "Annotation project queue" not in HTML and "/api/annotation-project" not in HTML
        assert "Run quality checks" in HTML and "Source FPS" in HTML
        assert 'id="cameraProfileSelect"' in HTML and "/api/camera-profile" in HTML
        assert 'aria-label="Workspaces"' in HTML and "same unchanged physical camera" in HTML
        create_config(
            "smoke-camera",
            [[0.1, 0.1], [0.9, 0.1], [0.5, 0.9]],
            0.25,
            [[0.1, 0.1], [0.2, 0.2]],
            [[0.2, 0.8], [0.5, 0.3]],
            [[0.2, 0.5], [0.8, 0.5]],
        )
        print("Web UI smoke test OK")
        return

    state = AppState()
    server = ThreadingHTTPServer((args.host, args.port), make_handler(state))
    host, port = server.server_address
    url = f"http://{host}:{port}/"
    print(f"Capstone UI: {url}")
    if not args.no_browser:
        threading.Timer(0.5, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
