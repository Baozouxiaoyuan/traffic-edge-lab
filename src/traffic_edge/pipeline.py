from __future__ import annotations

import json
import hashlib
import time
import zipfile
from collections import defaultdict, deque
from contextlib import ExitStack
from math import hypot, sqrt
from pathlib import Path
from typing import Any, Callable

import cv2
import numpy as np

from .backends import SyntheticColourTracker, TrackingBackend, UltralyticsByteTrackBackend
from .geometry import denormalize_polygon, point_in_polygon
from .rules import DirectedStopLineRule, RegionEntryRule, WrongWayRule
from .scope import validate_config_scope
from .signal import RedSignalDetector, denormalize_rectangle


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        return json.load(stream)


def build_backend(name: str, config: dict[str, Any]) -> TrackingBackend:
    if name == "demo":
        return SyntheticColourTracker()
    if name == "yolo":
        yolo = config["yolo"]
        return UltralyticsByteTrackBackend(
            model_name=yolo["model"],
            confidence=float(yolo["confidence"]),
            iou=float(yolo["iou"]),
            vehicle_classes=[int(value) for value in yolo["vehicle_classes"]],
            image_size=int(yolo.get("image_size", 960)),
            tracker_name=str(yolo.get("tracker", "bytetrack")),
            device=str(yolo.get("device", "auto")),
        )
    raise ValueError(f"Unknown backend: {name}")


def display_event_type(event_type: str) -> str:
    names = {
        "candidate_red_signal_violation": "RED LIGHT",
        "candidate_wrong_way_driving": "WRONG WAY",
        "candidate_restricted_zone_entry": "RESTRICTED ZONE",
        "candidate_lane_region_violation": "LANE ENTRY",
    }
    return names.get(event_type, event_type.replace("candidate_", "").replace("_", " ").upper())


def overlay_style(width: int, height: int) -> dict[str, float | int]:
    """Return legible annotation sizes without overwhelming small CCTV frames."""
    shortest_side = max(1, min(width, height))
    scale = max(0.55, min(1.45, shortest_side / 480.0))
    return {
        "font": round(0.48 * scale, 3),
        "small_font": round(0.40 * scale, 3),
        "text_thickness": 1 if shortest_side < 540 else 2,
        "line_thickness": max(1, round(2 * scale)),
        "event_thickness": max(2, round(4 * scale)),
        "header_height": max(15, round(25 * scale)),
        "padding": max(3, round(5 * scale)),
    }


def _fit_overlay_text(text: str, max_width: int, font_scale: float, thickness: int) -> str:
    """Ellipsize an annotation so it remains inside the available frame width."""
    if max_width <= 0:
        return ""
    if cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)[0][0] <= max_width:
        return text
    suffix = "..."
    candidate = text
    while candidate and cv2.getTextSize(
        candidate + suffix, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness
    )[0][0] > max_width:
        candidate = candidate[:-1]
    return candidate.rstrip() + suffix if candidate else ""


def evidence_quality_score(trigger: Any, signal_confidence: float | None = None) -> float:
    """Return an explainable evidence-strength score, not a probability of guilt."""
    detector = max(0.0, min(1.0, float(trigger.track.confidence)))
    alignment = abs(max(-1.0, min(1.0, float(trigger.direction_alignment))))
    if trigger.rule_name == "candidate_wrong_way_driving":
        persistence = max(0.0, min(1.0, float(trigger.opposite_step_ratio or 0.0)))
        observations = min(1.0, float(trigger.observations_used) / 12.0)
        return round(0.30 * detector + 0.25 * alignment + 0.30 * persistence + 0.15 * observations, 4)
    if trigger.rule_name == "candidate_red_signal_violation":
        signal = max(0.0, min(1.0, float(signal_confidence or 0.0)))
        return round(0.35 * detector + 0.30 * alignment + 0.35 * signal, 4)
    return round(0.55 * detector + 0.45 * alignment, 4)


def _polygon_area(polygon: list[tuple[int, int]]) -> float:
    return abs(sum(
        first[0] * second[1] - second[0] * first[1]
        for first, second in zip(polygon, polygon[1:] + polygon[:1])
    )) / 2.0


def _polygon_vertex_centroid(polygon: list[tuple[int, int]]) -> tuple[float, float]:
    return (
        sum(point[0] for point in polygon) / len(polygon),
        sum(point[1] for point in polygon) / len(polygon),
    )


def _point_to_axis_distance(
    point: tuple[float, float],
    axis: tuple[tuple[float, float], tuple[float, float]],
) -> float | None:
    start, end = axis
    dx, dy = end[0] - start[0], end[1] - start[1]
    length = hypot(dx, dy)
    if length <= 1e-6:
        return None
    return abs(dx * (start[1] - point[1]) - (start[0] - point[0]) * dy) / length


def assign_lane_for_point(
    point: tuple[float, float],
    lanes: list[dict[str, Any]],
    ambiguity_margin_normalized: float = 0.04,
) -> tuple[dict[str, Any] | None, str]:
    """Assign one lane to a road-contact point, or abstain when ownership is unclear.

    The controlled approach and an optional restricted-zone polygon define each
    lane's ownership area.  If ownership areas overlap, the point is compared
    with each lane's travel-direction axis, falling back to polygon-centroid
    distance when no direction is configured.  Distances are normalized by the
    containing polygon's area so differently sized lane polygons remain
    comparable.  Near ties deliberately abstain instead of evaluating one
    observation against multiple rule engines.
    """
    candidates: list[dict[str, Any]] = []
    for lane in lanes:
        polygons = lane.get("ownership_polygons") or [lane["polygon"]]
        containing = [polygon for polygon in polygons if point_in_polygon(point, polygon)]
        if containing:
            candidates.append({"lane": lane, "containing": containing})
    if not candidates:
        return None, "outside_all_lanes"
    if len(candidates) == 1:
        return candidates[0]["lane"], "assigned_single_lane"

    scored: list[tuple[float, float, dict[str, Any], str]] = []
    for candidate in candidates:
        lane = candidate["lane"]
        containing = candidate["containing"]
        # Use the closest containing region when a lane owns both an approach
        # polygon and a distinct restricted zone.
        region_metrics = []
        for polygon in containing:
            scale = max(1.0, sqrt(_polygon_area(polygon)))
            centroid = _polygon_vertex_centroid(polygon)
            centroid_distance = hypot(point[0] - centroid[0], point[1] - centroid[1]) / scale
            region_metrics.append((centroid_distance, scale))
        centroid_distance, scale = min(region_metrics)
        axis_distance = None
        if lane.get("direction_axis") is not None:
            raw_axis_distance = _point_to_axis_distance(point, lane["direction_axis"])
            if raw_axis_distance is not None:
                axis_distance = raw_axis_distance / scale
        method = "axis" if axis_distance is not None else "centroid"
        primary = axis_distance if axis_distance is not None else centroid_distance
        # Centroid distance is a small deterministic tie-breaker when two lane
        # axes are nearly parallel or coincident.
        score = primary + 0.05 * centroid_distance
        scored.append((score, centroid_distance, lane, method))
    scored.sort(key=lambda item: (item[0], item[1], str(item[2].get("id", ""))))
    best, second = scored[0], scored[1]
    if second[0] - best[0] <= max(0.0, ambiguity_margin_normalized):
        return None, "ambiguous_lane_overlap"
    return best[2], f"assigned_overlap_by_{best[3]}"


def _nearest_lane_for_state_update(
    point: tuple[float, float],
    lanes: list[dict[str, Any]],
    ambiguity_margin_normalized: float = 0.04,
) -> dict[str, Any] | None:
    """Choose at most one nearby lane for outside-to-inside rule state.

    An outside observation may need to record ``outside`` before a later region
    entry.  This helper never grants lane ownership and its result cannot by
    itself authorize an event; it only routes state to one unambiguous engine.
    """
    scored: list[tuple[float, float, dict[str, Any]]] = []
    for lane in lanes:
        polygons = lane.get("ownership_polygons") or [lane["polygon"]]
        region_metrics = []
        for polygon in polygons:
            scale = max(1.0, sqrt(_polygon_area(polygon)))
            centroid = _polygon_vertex_centroid(polygon)
            centroid_distance = hypot(point[0] - centroid[0], point[1] - centroid[1]) / scale
            region_metrics.append((centroid_distance, scale))
        centroid_distance, scale = min(region_metrics)
        axis_distance = None
        if lane.get("direction_axis") is not None:
            raw_axis_distance = _point_to_axis_distance(point, lane["direction_axis"])
            if raw_axis_distance is not None:
                axis_distance = raw_axis_distance / scale
        primary = axis_distance if axis_distance is not None else centroid_distance
        scored.append((primary + 0.05 * centroid_distance, centroid_distance, lane))
    scored.sort(key=lambda item: (item[0], item[1], str(item[2].get("id", ""))))
    if not scored:
        return None
    if len(scored) > 1 and scored[1][0] - scored[0][0] <= max(0.0, ambiguity_margin_normalized):
        return None
    return scored[0][2]


def create_event_clips(
    annotated_video: Path,
    evidence_dir: Path,
    events: list[dict[str, Any]],
    pre_seconds: float = 2.0,
    post_seconds: float = 3.0,
) -> int:
    """Create short annotated WebM clips in one video pass for human review."""
    if not events:
        return 0
    capture = cv2.VideoCapture(str(annotated_video))
    if not capture.isOpened():
        return 0
    fps = capture.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    ranges = []
    for event in events:
        timestamp = float(event["timestamp_seconds"])
        clip_name = f"{event['event_id']}-clip.webm"
        event.pop('evidence_clip_pointer', None)
        event.pop('evidence_clip_window_seconds', None)
        ranges.append({
            "start": max(0, int((timestamp - pre_seconds) * fps)),
            "end": max(0, int((timestamp + post_seconds) * fps)),
            "path": evidence_dir / clip_name,
            "writer": None,
            "event": event,
            "first_frame": None,
            "last_frame": None,
        })
    frame_index = 0
    created = 0
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            for item in ranges:
                if item["start"] <= frame_index <= item["end"]:
                    if item["writer"] is None:
                        item["writer"] = cv2.VideoWriter(
                            str(item["path"]),
                            cv2.VideoWriter_fourcc(*"VP80"),
                            fps,
                            (width, height),
                        )
                        if not item["writer"].isOpened():
                            item["writer"].release()
                            item["writer"] = False
                        else:
                            created += 1
                    if item["writer"]:
                        item["writer"].write(frame)
                        if item['first_frame'] is None:
                            item['first_frame'] = frame_index
                        item['last_frame'] = frame_index
            frame_index += 1
    finally:
        capture.release()
        for item in ranges:
            if item["writer"]:
                item["writer"].release()
                if item['first_frame'] is not None:
                    item['event']['evidence_clip_pointer'] = f"evidence/{item['path'].name}"
                    # Record the actual decoded interval, with an exclusive end.
                    # Near EOF the requested post-event context may not exist.
                    item['event']['evidence_clip_window_seconds'] = [
                        round(item['first_frame']/fps, 3),
                        round((item['last_frame']+1)/fps, 3),
                    ]
    return created


def write_run_report(
    path: Path,
    summary: dict[str, Any],
    events: list[dict[str, Any]],
) -> None:
    """Write a concise English report suitable for supervisor review."""
    counts = summary.get("candidate_events_by_type", {})
    lanes = summary.get("candidate_events_by_lane", {})
    real_time = summary.get("real_time_factor")
    lines = [
        "# Traffic-video analysis run report",
        "",
        "> Research candidate events only. This report is not a legal infringement decision.",
        "",
        "## Executive outcome",
        "",
        f"- Processed **{summary['frames_processed']}** frames from `{Path(summary['input_video']).name}`.",
        f"- Generated **{summary['candidate_events']}** candidate events across **{summary['lane_count']}** independently configured lanes.",
        f"- Consolidated those events into **{summary.get('candidate_incidents', summary['candidate_events'])}** vehicle incidents; **{summary.get('multi_rule_incidents', 0)}** contain more than one rule type.",
        f"- Reliability policy: **{summary.get('fallback_observations_excluded', 0)}** provisional observations were excluded from non-motion rules; **{summary.get('fallback_wrong_way_observations_evaluated', 0)}** were eligible only for persistent wrong-way analysis; **{summary.get('lane_assignment_diagnostics', {}).get('ambiguous_lane_overlap', 0)}** overlapping-lane observations were rejected as ambiguous.",
        f"- Detector/tracker: **{summary['model']} + {summary['tracker']}**.",
        f"- End-to-end throughput: **{summary['processing_fps']} FPS** ({real_time:.2f}x source rate)." if real_time is not None else "- End-to-end throughput was unavailable.",
        f"- Metadata plus evidence reduction versus source video: **{summary['metadata_plus_evidence_reduction_percent_vs_video']}%**.",
        "",
        "## Candidate counts",
        "",
        "| Rule | Candidates |",
        "|---|---:|",
    ]
    if counts:
        lines.extend(f"| {display_event_type(rule)} | {count} |" for rule, count in counts.items())
    else:
        lines.append("| No candidate events | 0 |")
    lines.extend(["", "| Lane | Candidates |", "|---|---:|"])
    if lanes:
        lines.extend(f"| {lane} | {count} |" for lane, count in lanes.items())
    else:
        lines.append("| No lane-owned candidates | 0 |")
    lines.extend([
        "",
        "## Event review queue",
        "",
        "| Time (s) | Lane | Rule | Track | Evidence strength | Review flags |",
        "|---:|---|---|---:|---:|---|",
    ])
    if events:
        for event in events:
            flags = ", ".join(event.get("review_flags", [])) or "none"
            lines.append(
                f"| {event['timestamp_seconds']:.3f} | {event.get('lane_name', event.get('lane_id', 'unknown'))} | "
                f"{display_event_type(event['candidate_event'])} | {event['track_id']} | "
                f"{event.get('evidence_quality_score', 0):.2f} | {flags} |"
            )
    else:
        lines.append("| — | — | No candidate event | — | — | — |")
    lines.extend([
        "",
        "## Why this output is useful",
        "",
        "The system reuses one vehicle detection/tracking pass for multiple lane-specific traffic rules. Each candidate is tied to a vehicle, lane, time, frozen configuration and review evidence, allowing event-level accuracy analysis and selective edge-to-cloud transfer instead of uploading the complete source video.",
        "",
        "## Required interpretation",
        "",
        "- A candidate must be checked against the original video by a human reviewer.",
        "- Evidence strength is an explainability heuristic, not a calibrated probability.",
        "- A zero-candidate result does not prove that no violation occurred.",
        "- Precision, recall and F1 require complete independent ground truth from a frozen test split.",
        "- Pixel-space motion is not a legal speed measurement.",
        "",
        f"Configuration SHA-256: `{summary['config_sha256']}`",
    ])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_incidents(events: list[dict[str, Any]], merge_window_seconds: float = 2.0) -> list[dict[str, Any]]:
    """Consolidate near-simultaneous rule events for the same tracked vehicle and lane."""
    groups: list[list[dict[str, Any]]] = []
    for event in sorted(events, key=lambda item: float(item["timestamp_seconds"])):
        matching = next((
            group for group in reversed(groups)
            if group[-1].get("camera_id") == event.get("camera_id")
            and group[-1].get("lane_id") == event.get("lane_id")
            and group[-1].get("track_id") == event.get("track_id")
            and float(event["timestamp_seconds"]) - float(group[-1]["timestamp_seconds"]) <= merge_window_seconds
        ), None)
        if matching is None:
            groups.append([event])
        else:
            matching.append(event)
    incidents = []
    for index, group in enumerate(groups, start=1):
        first = group[0]
        rule_types = sorted({str(event["candidate_event"]) for event in group})
        flags = sorted({flag for event in group for flag in event.get("review_flags", [])})
        incidents.append({
            "incident_id": f"{first['camera_id']}-{first.get('lane_id', 'lane')}-track{first['track_id']}-incident{index:04d}",
            "camera_id": first["camera_id"],
            "lane_id": first.get("lane_id"),
            "lane_name": first.get("lane_name"),
            "track_id": first["track_id"],
            "start_timestamp_seconds": min(float(event["timestamp_seconds"]) for event in group),
            "end_timestamp_seconds": max(float(event["timestamp_seconds"]) for event in group),
            "candidate_event_types": rule_types,
            "event_ids": [event["event_id"] for event in group],
            "event_count": len(group),
            "multi_rule": len(rule_types) > 1,
            "maximum_evidence_quality_score": max(float(event.get("evidence_quality_score", 0.0)) for event in group),
            "review_flags": flags,
            "evidence_pointers": sorted({
                pointer for event in group
                for pointer in (event.get("evidence_pointer"), event.get("vehicle_evidence_pointer"), event.get("evidence_clip_pointer"))
                if pointer
            }),
            "research_only": True,
        })
    return incidents


def create_evidence_package(output_dir: Path, included_paths: list[Path]) -> tuple[Path, Path]:
    """Create a checksummed, portable evidence archive without the full annotated video."""
    records = []
    files = []
    for path in included_paths:
        if path.is_dir():
            files.extend(sorted(candidate for candidate in path.rglob("*") if candidate.is_file()))
        elif path.is_file():
            files.append(path)
    for path in files:
        records.append({
            "path": path.resolve().relative_to(output_dir.resolve()).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        })
    manifest_path = output_dir / "EVIDENCE_MANIFEST.json"
    manifest_path.write_text(json.dumps({
        "algorithm": "SHA-256",
        "purpose": "Integrity manifest for research evidence; not a legal chain-of-custody certification.",
        "files": records,
    }, indent=2), encoding="utf-8")
    bundle_path = output_dir / "evidence_bundle.zip"
    with zipfile.ZipFile(bundle_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in files + [manifest_path]:
            archive.write(path, path.resolve().relative_to(output_dir.resolve()).as_posix())
    return manifest_path, bundle_path


def run_pipeline(
    input_video: Path,
    output_dir: Path,
    config_path: Path,
    backend_name: str,
    progress_callback: Callable[[int, int], None] | None = None,
    device: str | None = None,
) -> dict[str, Any]:
    # Initialization may fail before entering the frame loop.
    with ExitStack() as resources:
        return _run_pipeline(input_video, output_dir, config_path, backend_name, progress_callback, resources, device)


def _run_pipeline(
    input_video: Path, output_dir: Path, config_path: Path, backend_name: str,
    progress_callback: Callable[[int, int], None] | None, resources: ExitStack,
    device: str | None = None,
) -> dict[str, Any]:
    config = load_config(config_path)
    if device is not None:
        from .compute import normalize_device
        config.setdefault("yolo", {})["device"] = normalize_device(device)
    validate_config_scope(config)
    output_dir.mkdir(parents=True, exist_ok=True)
    if (output_dir / "summary.json").exists() or (output_dir / "events.jsonl").exists():
        raise ValueError("Output directory already contains a run. Choose a new directory to preserve evidence.")
    config_bytes = json.dumps(config, sort_keys=True, indent=2).encode("utf-8")
    (output_dir / "config_used.json").write_bytes(config_bytes)
    config_digest = hashlib.sha256(config_bytes).hexdigest()
    evidence_dir = output_dir / "evidence"
    evidence_dir.mkdir(exist_ok=True)

    capture = cv2.VideoCapture(str(input_video))
    resources.callback(capture.release)
    if not capture.isOpened():
        raise FileNotFoundError(f"Could not open input video: {input_video}")
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = capture.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    raw_lanes = config.get("lanes") or [config]
    lane_contexts: list[dict[str, Any]] = []
    for lane_index, lane_config in enumerate(raw_lanes):
        polygon = denormalize_polygon(lane_config["forbidden_region"]["normalized_polygon"], width, height)
        restricted_config = lane_config.get("restricted_zone")
        restricted_polygon = None
        if restricted_config and restricted_config.get("normalized_polygon"):
            restricted_polygon = denormalize_polygon(restricted_config["normalized_polygon"], width, height)
        direction_points = lane_config.get("controlled_movement", {}).get("normalized_direction")
        controlled_direction = None
        direction_axis = None
        if direction_points and len(direction_points) == 2:
            direction_axis = tuple(
                (float(point[0]) * width, float(point[1]) * height)
                for point in direction_points
            )
            controlled_direction = (
                direction_axis[1][0] - direction_axis[0][0],
                direction_axis[1][1] - direction_axis[0][1],
            )
        signal_config = lane_config.get("traffic_signal")
        signal_detector = None
        if (backend_name != 'demo' and lane_config.get('enabled_rules', {}).get('red_light', True)
                and lane_config.get('rule', {}).get('event_type') == 'candidate_red_signal_violation'
                and not signal_config):
            raise ValueError('Red-light detection requires a controlling-signal region; missing signal evidence cannot be assumed red')
        if signal_config:
            signal_rectangle = denormalize_rectangle(signal_config["normalized_rectangle"], width, height)
            signal_detector = RedSignalDetector(
                signal_rectangle,
                minimum_red_ratio=float(signal_config.get("minimum_red_ratio", 0.015)),
                history_size=int(signal_config.get("history_frames", 5)),
                minimum_colour_ratio=float(signal_config.get("minimum_colour_ratio", 0.003)),
                red_dominance_ratio=float(signal_config.get("red_dominance_ratio", 1.35)),
                minimum_blob_ratio=float(signal_config.get("minimum_blob_ratio", 0.001)),
            )
        lane_contexts.append({
            "id": str(lane_config.get("id", f"lane-{lane_index + 1}")),
            "name": str(lane_config.get("name", f"Lane {lane_index + 1}")),
            "config": lane_config,
            "polygon": polygon,
            "restricted_config": restricted_config,
            "restricted_polygon": restricted_polygon,
            "ownership_polygons": [polygon] + ([restricted_polygon] if restricted_polygon is not None else []),
            "direction_points": direction_points,
            "direction_axis": direction_axis,
            "controlled_direction": controlled_direction,
            "signal_detector": signal_detector,
            "signal_is_red": True,
            "red_signal_frames": 0,
            "unknown_signal_frames": 0,
        })

    annotated_path = output_dir / "annotated.webm"
    annotated_scale = min(1.0, 640.0 / width)
    annotated_width = max(2, int(width * annotated_scale) // 2 * 2)
    annotated_height = max(2, int(height * annotated_scale) // 2 * 2)
    annotated_stride = 2 if fps >= 20 else 1
    annotated_fps = fps / annotated_stride
    writer = cv2.VideoWriter(
        str(annotated_path),
        cv2.VideoWriter_fourcc(*"VP80"),
        annotated_fps,
        (annotated_width, annotated_height),
    )
    resources.callback(writer.release)
    if not writer.isOpened():
        raise RuntimeError(f"Could not create annotated video: {annotated_path}")

    backend = build_backend(backend_name, config)
    tracker_name = (
        str(config.get("yolo", {}).get("tracker", "bytetrack"))
        if backend_name == "yolo"
        else "not_applicable"
    )
    rule_engines: list[tuple[dict[str, Any], str, Any]] = []
    for lane in lane_contexts:
        lane_config = lane["config"]
        enabled_rules = lane_config.get("enabled_rules", {})
        red_light_enabled = bool(enabled_rules.get("red_light", True))
        wrong_way_enabled = bool(enabled_rules.get("wrong_way", False))
        restricted_zone_enabled = bool(enabled_rules.get("restricted_zone", False))
        direction_points = lane["direction_points"]
        controlled_direction = lane["controlled_direction"]
        stop_line_points = lane_config.get("stop_line", {}).get("normalized_points")
        stop_line = None
        if stop_line_points and len(stop_line_points) == 2:
            stop_line = tuple((float(point[0]) * width, float(point[1]) * height) for point in stop_line_points)
        lane["stop_line"] = stop_line
        if red_light_enabled and stop_line is not None and direction_points and controlled_direction:
            direction_start = (float(direction_points[0][0]) * width, float(direction_points[0][1]) * height)
            rule_engines.append((lane, "red_light", DirectedStopLineRule(
                approach_polygon=lane["polygon"], stop_line=stop_line, direction_start=direction_start,
                controlled_direction=controlled_direction, region_name=lane["name"],
                event_type=lane_config["rule"]["event_type"],
                min_track_age_frames=int(lane_config["rule"]["min_track_age_frames"]),
                min_direction_alignment=float(lane_config.get("controlled_movement", {}).get("minimum_alignment", 0.75)),
                min_motion_pixels=float(lane_config.get("controlled_movement", {}).get("minimum_motion_pixels", 12.0)),
            )))
        elif red_light_enabled:
            rule_engines.append((lane, "red_light", RegionEntryRule(
                polygon=lane["polygon"], region_name=lane["name"],
                event_type=lane_config["rule"]["event_type"],
                min_track_age_frames=int(lane_config["rule"]["min_track_age_frames"]),
                controlled_direction=controlled_direction,
                min_direction_alignment=float(lane_config.get("controlled_movement", {}).get("minimum_alignment", 0.55)),
                min_motion_pixels=float(lane_config.get("controlled_movement", {}).get("minimum_motion_pixels", 12.0)),
            )))
        if wrong_way_enabled:
            if controlled_direction is None:
                raise ValueError(f"{lane['name']}: wrong-way detection requires a legal-direction arrow")
            wrong_way_config = lane_config.get("wrong_way", {})
            reference_height = float(wrong_way_config.get("reference_frame_height", 720.0))
            scaled_motion_threshold = max(3.0, float(wrong_way_config.get("minimum_motion_pixels", 12.0)) * height / reference_height)
            minimum_inside_frames = max(3, round(fps * float(wrong_way_config.get("minimum_inside_seconds", 0.2))))
            max_occlusion_gap_frames = max(1, round(fps * float(wrong_way_config.get("maximum_occlusion_seconds", 0.5))))
            max_tracklet_stitch_gap_frames = max(1, round(fps * float(wrong_way_config.get("maximum_tracklet_stitch_seconds", 0.8))))
            rule_engines.append((lane, "wrong_way", WrongWayRule(
                polygon=lane["polygon"], legal_direction=controlled_direction, region_name=lane["name"],
                event_type="candidate_wrong_way_driving", min_track_age_frames=minimum_inside_frames,
                maximum_alignment=float(wrong_way_config.get("maximum_alignment", -0.7)),
                min_motion_pixels=scaled_motion_threshold,
                consecutive_frames=int(wrong_way_config.get("consecutive_frames", 2)),
                minimum_opposite_step_ratio=float(wrong_way_config.get("minimum_opposite_step_ratio", 0.6)),
                max_occlusion_gap_frames=max_occlusion_gap_frames,
                enable_tracklet_stitching=bool(wrong_way_config.get("enable_tracklet_stitching", False)),
                max_tracklet_stitch_gap_frames=max_tracklet_stitch_gap_frames,
                max_stitch_distance_bbox_diagonals=float(wrong_way_config.get("maximum_stitch_distance_bbox_diagonals", 1.5)),
                min_motion_bbox_height_ratio=float(wrong_way_config.get("minimum_motion_bbox_height_ratio", 0.15)),
            )))
        if restricted_zone_enabled:
            restricted_config = lane["restricted_config"]
            if lane["restricted_polygon"] is None:
                raise ValueError(f"{lane['name']}: restricted-zone detection requires a polygon")
            rule_engines.append((lane, "restricted_zone", RegionEntryRule(
                polygon=lane["restricted_polygon"], region_name=f"{lane['name']} restricted zone",
                event_type="candidate_restricted_zone_entry",
                min_track_age_frames=int(restricted_config.get("min_track_age_frames", 5)),
                trigger_on_initial_presence=bool(restricted_config.get("trigger_on_initial_presence", False)),
            )))
    if not rule_engines:
        raise ValueError("At least one violation rule must be enabled")
    event_path = output_dir / "events.jsonl"
    event_stream = event_path.open("w", encoding="utf-8")
    resources.callback(event_stream.close)
    trajectories: dict[int, deque[tuple[int, int]]] = defaultdict(lambda: deque(maxlen=40))
    violation_highlights: dict[int, dict[str, dict[str, Any]]] = {}
    event_counts_by_type: dict[str, int] = defaultdict(int)
    event_counts_by_lane: dict[str, int] = defaultdict(int)
    event_count = 0
    recorded_events: list[dict[str, Any]] = []
    frame_index = 0
    fallback_observations_excluded = 0
    fallback_track_ids_excluded: set[int] = set()
    fallback_wrong_way_observations_evaluated = 0
    fallback_wrong_way_track_ids: set[int] = set()
    lane_assignment_counts: dict[str, int] = defaultdict(int)
    ambiguous_lane_track_ids: set[int] = set()
    outside_lane_track_ids: set[int] = set()
    last_owned_lane_by_track: dict[int, dict[str, Any]] = {}
    last_owned_frame_by_track: dict[int, int] = {}
    started = time.perf_counter()

    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            frame_index += 1
            if progress_callback and (frame_index == 1 or frame_index % 5 == 0):
                progress_callback(frame_index, total_frames)
            for lane in lane_contexts:
                detector = lane["signal_detector"]
                lane["signal_is_red"] = detector.update(frame) if detector else True
                if detector and lane["signal_is_red"]:
                    lane["red_signal_frames"] += 1
                if detector and detector.last_state == "unknown":
                    lane["unknown_signal_frames"] += 1
            observations = backend.track(frame)
            assigned_lanes_by_track: dict[int, dict[str, Any]] = {}
            state_update_lanes_by_track: dict[int, dict[str, Any]] = {}
            recent_lane_exit_tracks: set[int] = set()
            assignment_status_by_track: dict[int, str] = {}
            for observation in observations:
                assigned_lane, assignment_status = assign_lane_for_point(
                    observation.ground_point,
                    lane_contexts,
                )
                is_fallback = observation.track_id >= 1_000_000
                fallback_wrong_way_allowed = bool(
                    is_fallback
                    and assigned_lane is not None
                    and assigned_lane["config"].get("enabled_rules", {}).get("wrong_way", False)
                )
                if is_fallback and not fallback_wrong_way_allowed:
                    fallback_observations_excluded += 1
                    fallback_track_ids_excluded.add(observation.track_id)
                    assignment_status_by_track[observation.track_id] = "excluded_fallback_tracking_id"
                    continue
                if fallback_wrong_way_allowed:
                    fallback_wrong_way_observations_evaluated += 1
                    fallback_wrong_way_track_ids.add(observation.track_id)
                assignment_status_by_track[observation.track_id] = assignment_status
                lane_assignment_counts[assignment_status] += 1
                if assigned_lane is not None:
                    assigned_lanes_by_track[observation.track_id] = assigned_lane
                    state_update_lanes_by_track[observation.track_id] = assigned_lane
                    last_owned_lane_by_track[observation.track_id] = assigned_lane
                    last_owned_frame_by_track[observation.track_id] = frame_index
                elif assignment_status == "ambiguous_lane_overlap":
                    ambiguous_lane_track_ids.add(observation.track_id)
                else:
                    outside_lane_track_ids.add(observation.track_id)
                    previous_lane = last_owned_lane_by_track.get(observation.track_id)
                    previous_frame = last_owned_frame_by_track.get(observation.track_id, -1)
                    exit_grace_frames = max(2, round(fps * 0.30))
                    if previous_lane is not None and frame_index - previous_frame <= exit_grace_frames:
                        state_update_lanes_by_track[observation.track_id] = previous_lane
                        recent_lane_exit_tracks.add(observation.track_id)
                        lane_assignment_counts["outside_recent_lane_state_update"] += 1
                    else:
                        nearest_lane = _nearest_lane_for_state_update(observation.ground_point, lane_contexts)
                        if nearest_lane is not None:
                            state_update_lanes_by_track[observation.track_id] = nearest_lane
                            lane_assignment_counts["outside_nearest_lane_state_update"] += 1
            for rule_lane, rule_kind, rule_engine in rule_engines:
                # Crossing/entry state must break when detection or unique lane
                # ownership is lost. Wrong-way keeps its own bounded gap policy.
                frame_lanes = assigned_lanes_by_track if rule_kind == 'wrong_way' else state_update_lanes_by_track
                rule_engine.begin_frame(frame_index, {
                    track_id for track_id, assigned_lane in frame_lanes.items()
                    if assigned_lane['id'] == rule_lane['id'] and (rule_kind == 'wrong_way' or track_id < 1_000_000)
                })
            source_frame = frame.copy()
            style = overlay_style(width, height)
            font_scale = float(style["font"])
            small_font_scale = float(style["small_font"])
            text_thickness = int(style["text_thickness"])
            line_thickness = int(style["line_thickness"])
            event_thickness = int(style["event_thickness"])
            header_height = int(style["header_height"])
            text_padding = int(style["padding"])
            compact_overlay = min(width, height) < 480
            lane_colours = [(0, 0, 255), (255, 120, 0), (0, 170, 255), (180, 0, 180), (0, 180, 90), (200, 80, 80)]
            for lane_index, lane in enumerate(lane_contexts):
                lane_colour = lane_colours[lane_index % len(lane_colours)]
                polygon = lane["polygon"]
                overlay = frame.copy()
                cv2.fillPoly(overlay, [np.array(polygon, dtype=np.int32)], lane_colour)
                frame = cv2.addWeighted(overlay, 0.10, frame, 0.90, 0)
                cv2.polylines(frame, [np.array(polygon, dtype=np.int32)], True, lane_colour, line_thickness)
                label_point = tuple(int(value) for value in polygon[0])
                if not compact_overlay:
                    lane_label = _fit_overlay_text(
                        str(lane["name"]).upper(), width - label_point[0] - text_padding, small_font_scale, text_thickness
                    )
                    cv2.putText(
                        frame, lane_label, (label_point[0], max(12, label_point[1] - text_padding)),
                        cv2.FONT_HERSHEY_SIMPLEX, small_font_scale, lane_colour, text_thickness,
                    )
                if lane["restricted_polygon"] is not None:
                    restricted_overlay = frame.copy()
                    cv2.fillPoly(restricted_overlay, [np.array(lane["restricted_polygon"], dtype=np.int32)], (180, 0, 180))
                    frame = cv2.addWeighted(restricted_overlay, 0.14, frame, 0.86, 0)
                    cv2.polylines(frame, [np.array(lane["restricted_polygon"], dtype=np.int32)], True, (255, 0, 255), line_thickness)
                if lane.get("stop_line"):
                    cv2.line(frame, tuple(round(value) for value in lane["stop_line"][0]), tuple(round(value) for value in lane["stop_line"][1]), lane_colour, event_thickness)
                detector = lane["signal_detector"]
                if detector:
                    sx1, sy1, sx2, sy2 = detector.rectangle
                    signal_colour = ((0, 0, 255) if lane["signal_is_red"] else
                                     (0, 190, 255) if detector.last_state == "unknown" else (0, 210, 0))
                    signal_label = "RED" if lane["signal_is_red"] else "UNKNOWN" if detector.last_state == "unknown" else "NOT RED"
                    cv2.rectangle(frame, (sx1, sy1), (sx2, sy2), signal_colour, 2)
                    signal_text = _fit_overlay_text(
                        f"SIGNAL {signal_label} {detector.last_confidence:.2f}", width - sx1 - text_padding,
                        small_font_scale, text_thickness,
                    )
                    cv2.putText(frame, signal_text, (sx1, max(12, sy1 - text_padding)), cv2.FONT_HERSHEY_SIMPLEX, small_font_scale, signal_colour, text_thickness)
                direction_points = lane["direction_points"]
                if direction_points:
                    arrow_start = (round(float(direction_points[0][0]) * width), round(float(direction_points[0][1]) * height))
                    arrow_end = (round(float(direction_points[1][0]) * width), round(float(direction_points[1][1]) * height))
                    cv2.arrowedLine(frame, arrow_start, arrow_end, lane_colour, line_thickness, tipLength=0.15)

            triggers = []
            for observation in observations:
                x1, y1, x2, y2 = observation.bbox
                ground = tuple(round(value) for value in observation.ground_point)
                trajectories[observation.track_id].append(ground)
                assigned_lane = assigned_lanes_by_track.get(observation.track_id)
                assignment_status = assignment_status_by_track.get(observation.track_id, "outside_all_lanes")
                inside_controlled_approach = assigned_lane is not None
                if len(trajectories[observation.track_id]) > 1:
                    trajectory_colour = (0, 255, 255) if inside_controlled_approach else (130, 130, 130)
                    cv2.polylines(frame, [np.array(trajectories[observation.track_id])], False, trajectory_colour, line_thickness)
                box_colour = (0, 230, 0) if inside_controlled_approach else (130, 130, 130)
                cv2.rectangle(frame, (x1, y1), (x2, y2), box_colour, line_thickness)
                cv2.circle(frame, ground, max(2, line_thickness + 1), (0, 255, 255), -1)
                # Label in-scope vehicles; use grey boxes for excluded traffic
                # to keep evidence frames readable.
                if inside_controlled_approach and not compact_overlay:
                    label = _fit_overlay_text(
                        f"{observation.class_name} {observation.confidence:.2f}",
                        width - x1 - text_padding, small_font_scale, text_thickness,
                    )
                    cv2.putText(
                        frame, label, (x1, max(12, y1 - text_padding)), cv2.FONT_HERSHEY_SIMPLEX,
                        small_font_scale, box_colour, text_thickness,
                    )
                rule_lane = state_update_lanes_by_track.get(observation.track_id)
                if rule_lane is None:
                    continue
                for lane, rule_kind, rule_engine in rule_engines:
                    if lane["id"] != rule_lane["id"]:
                        continue
                    # Provisional fallback IDs can support wrong-way candidates
                    # only: persistent opposite motion provides a multi-frame
                    # identity check. Instantaneous/entry rules require a
                    # tracker-confirmed ID to avoid edge-entry false positives.
                    if observation.track_id >= 1_000_000 and rule_kind != "wrong_way":
                        continue
                    if rule_kind == "red_light":
                        trigger = rule_engine.evaluate(observation, signal_is_red=lane["signal_is_red"])
                    else:
                        trigger = rule_engine.evaluate(observation)
                    trigger_is_lane_owned = assigned_lane is not None
                    trigger_is_boundary_crossing = (
                        observation.track_id in recent_lane_exit_tracks
                        and rule_kind == "red_light"
                    )
                    if trigger is not None and (trigger_is_lane_owned or trigger_is_boundary_crossing):
                        triggers.append((trigger, lane))

            for trigger, lane in triggers:
                highlight_key = f"{lane['id']}:{trigger.rule_name}"
                violation_highlights.setdefault(trigger.track.track_id, {})[highlight_key] = {
                    "remaining": max(1, round(2.0 * fps)), "event_type": trigger.rule_name, "lane_name": lane["name"]
                }

            observations_by_id = {observation.track_id: observation for observation in observations}
            for track_id in list(violation_highlights):
                observation = observations_by_id.get(track_id)
                event_timers = violation_highlights[track_id]
                active_events = [value for value in event_timers.values() if value["remaining"] > 0]
                if observation is not None:
                    x1, y1, x2, y2 = observation.bbox
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), event_thickness)
                    if y1 >= header_height + text_padding:
                        header_top = y1 - header_height
                        header_bottom = y1
                    else:
                        header_top = y2
                        header_bottom = min(height - 1, y2 + header_height)
                    type_text = "/".join(display_event_type(event["event_type"]) for event in active_events)
                    lane_text = "/".join(dict.fromkeys(str(event["lane_name"]).upper() for event in active_events))
                    raw_event_text = (
                        f"{lane_text} | {type_text}"
                        if compact_overlay
                        else f"{lane_text} | {type_text} | VEHICLE {track_id}"
                    )
                    event_text = _fit_overlay_text(
                        raw_event_text, width - x1 - 2 * text_padding,
                        font_scale, text_thickness,
                    )
                    text_width = cv2.getTextSize(
                        event_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_thickness
                    )[0][0]
                    header_right = min(width - 1, max(x2, x1 + text_width + 2 * text_padding))
                    cv2.rectangle(frame, (x1, header_top), (header_right, header_bottom), (0, 0, 210), -1)
                    cv2.putText(
                        frame,
                        event_text,
                        (x1 + text_padding, min(height - text_padding, header_bottom - text_padding)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        font_scale,
                        (255, 255, 255),
                        text_thickness,
                    )
                for event_key in list(event_timers):
                    event_timers[event_key]["remaining"] -= 1
                    if event_timers[event_key]["remaining"] <= 0:
                        del event_timers[event_key]
                if not event_timers:
                    del violation_highlights[track_id]
            timestamp_seconds = (frame_index - 1) / fps
            for trigger, lane in triggers:
                event_count += 1
                event_counts_by_type[trigger.rule_name] += 1
                event_counts_by_lane[lane["id"]] += 1
                event_id = f"{config['camera_id']}-{lane['id']}-track{trigger.track.track_id}-{event_count:04d}"
                evidence_name = f"{event_id}.jpg"
                vehicle_evidence_name = f"{event_id}-vehicle.jpg"
                cv2.imwrite(str(evidence_dir / evidence_name), frame)
                x1, y1, x2, y2 = trigger.track.bbox
                padding_x = round((x2 - x1) * 0.2)
                padding_y = round((y2 - y1) * 0.2)
                crop_x1, crop_y1 = max(0, x1 - padding_x), max(0, y1 - padding_y)
                crop_x2, crop_y2 = min(width, x2 + padding_x), min(height, y2 + padding_y)
                vehicle_crop = source_frame[crop_y1:crop_y2, crop_x1:crop_x2]
                if vehicle_crop.size:
                    cv2.imwrite(str(evidence_dir / vehicle_evidence_name), vehicle_crop)
                signal_detector = lane["signal_detector"]
                signal_confidence = (
                    signal_detector.last_confidence
                    if trigger.rule_name == "candidate_red_signal_violation" and signal_detector
                    else None
                )
                quality_score = evidence_quality_score(trigger, signal_confidence)
                event = {
                    "event_id": event_id,
                    "timestamp_seconds": round(timestamp_seconds, 3),
                    "frame_index": frame_index,
                    "camera_id": config["camera_id"],
                    "lane_id": lane["id"],
                    "lane_name": lane["name"],
                    "track_id": trigger.track.track_id,
                    "candidate_event": trigger.rule_name,
                    "region": trigger.region_name,
                    "object_class": trigger.track.class_name,
                    "detector_confidence": round(trigger.track.confidence, 4),
                    "signal_state": "red" if trigger.rule_name == "candidate_red_signal_violation" else "not_applicable",
                    "signal_confidence": round(signal_confidence, 4) if signal_confidence is not None else None,
                    "evidence_quality_score": quality_score,
                    "evidence_quality_note": "Heuristic evidence strength for review; not a calibrated probability or legal conclusion.",
                    "direction_alignment": round(trigger.direction_alignment, 4),
                    "rule_explanation": {
                        "observations_used": trigger.observations_used,
                        "opposite_step_ratio": (
                            round(trigger.opposite_step_ratio, 4)
                            if trigger.opposite_step_ratio is not None else None
                        ),
                        "tracklet_stitched": trigger.tracklet_stitched,
                        "previous_track_id": trigger.previous_track_id,
                    },
                    "review_flags": [
                        flag for flag, applies in (
                            ("low_detector_confidence", trigger.track.confidence < 0.25),
                            ("fallback_tracking_id", trigger.track.track_id >= 1_000_000),
                            ("stitched_tracklet", trigger.tracklet_stitched),
                            ("low_signal_confidence", signal_confidence is not None and signal_confidence < 0.6),
                            ("weak_combined_evidence", quality_score < 0.5),
                        ) if applies
                    ],
                    "rule_result": True,
                    "ground_point": [round(value, 2) for value in trigger.track.ground_point],
                    "ground_point_normalized": [
                        round(trigger.track.ground_point[0] / width, 6),
                        round(trigger.track.ground_point[1] / height, 6),
                    ],
                    "bbox_normalized": [
                        round(trigger.track.bbox[0] / width, 6),
                        round(trigger.track.bbox[1] / height, 6),
                        round(trigger.track.bbox[2] / width, 6),
                        round(trigger.track.bbox[3] / height, 6),
                    ],
                    "evidence_pointer": f"evidence/{evidence_name}",
                    "vehicle_evidence_pointer": f"evidence/{vehicle_evidence_name}",
                    "research_only": True,
                    "tracking_source": "fallback" if trigger.track.track_id >= 1_000_000 else tracker_name,
                    "config_sha256": config_digest,
                }
                event_stream.write(json.dumps(event, ensure_ascii=False) + "\n")
                event_stream.flush()
                recorded_events.append(event)
            if (frame_index - 1) % annotated_stride == 0:
                annotated_frame = frame
                if (annotated_width, annotated_height) != (width, height):
                    annotated_frame = cv2.resize(
                        frame,
                        (annotated_width, annotated_height),
                        interpolation=cv2.INTER_AREA,
                    )
                writer.write(annotated_frame)
    finally:
        elapsed = time.perf_counter() - started
        resources.close()

    if frame_index == 0:
        raise ValueError('The video contains no readable frames; no completed analysis was produced')

    clip_count = 0
    evidence_config = config.get("evidence", {})
    if evidence_config.get("event_clips", False):
        clip_count = create_event_clips(
            annotated_path,
            evidence_dir,
            recorded_events,
            pre_seconds=float(evidence_config.get("clip_pre_seconds", 2.0)),
            post_seconds=float(evidence_config.get("clip_post_seconds", 3.0)),
        )
        with event_path.open("w", encoding="utf-8") as stream:
            for event in recorded_events:
                stream.write(json.dumps(event, ensure_ascii=False) + "\n")

    incidents = build_incidents(
        recorded_events,
        merge_window_seconds=float(evidence_config.get("incident_merge_window_seconds", 2.0)),
    )
    incidents_path = output_dir / "incidents.json"
    incidents_path.write_text(json.dumps(incidents, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    input_bytes = input_video.stat().st_size
    if progress_callback:
        progress_callback(frame_index, total_frames)
    metadata_bytes = event_path.stat().st_size + incidents_path.stat().st_size
    evidence_bytes = sum(path.stat().st_size for path in evidence_dir.iterdir() if path.is_file())
    processing_fps = frame_index / elapsed if elapsed else None
    real_time_factor = processing_fps / fps if processing_fps and fps else None
    average_processing_ms = elapsed / frame_index * 1000.0 if frame_index else None
    recommended_headroom_factor = 1.25
    summary = {
        "backend": backend_name,
        "compute_device": getattr(backend, "device", "cpu"),
        "model": Path(config.get("yolo", {}).get("model", "not_applicable")).name if backend_name == "yolo" else "not_applicable",
        "tracker": tracker_name,
        "config_sha256": config_digest,
        "input_video": str(input_video.resolve()),
        "frames_processed": frame_index,
        "source_fps": round(fps, 3),
        "processing_seconds": round(elapsed, 3),
        "processing_fps": round(processing_fps, 3) if processing_fps else None,
        "average_processing_ms_per_frame": round(average_processing_ms, 3) if average_processing_ms else None,
        "real_time_factor": round(real_time_factor, 4) if real_time_factor is not None else None,
        "meets_source_frame_rate": bool(real_time_factor is not None and real_time_factor >= 1.0),
        "recommended_headroom_factor": recommended_headroom_factor,
        "meets_recommended_real_time_headroom": bool(
            real_time_factor is not None and real_time_factor >= recommended_headroom_factor
        ),
        "required_fps_for_recommended_headroom": round(fps * recommended_headroom_factor, 3),
        "candidate_events": event_count,
        "candidate_incidents": len(incidents),
        "multi_rule_incidents": sum(1 for incident in incidents if incident["multi_rule"]),
        "fallback_observations_excluded": fallback_observations_excluded,
        "fallback_track_ids_excluded": len(fallback_track_ids_excluded),
        "fallback_wrong_way_observations_evaluated": fallback_wrong_way_observations_evaluated,
        "fallback_wrong_way_track_ids": len(fallback_wrong_way_track_ids),
        "lane_assignment_diagnostics": {
            **dict(sorted(lane_assignment_counts.items())),
            "ambiguous_track_ids": len(ambiguous_lane_track_ids),
            "outside_track_ids": len(outside_lane_track_ids),
        },
        "event_evidence_clips": clip_count,
        "candidate_events_by_type": dict(sorted(event_counts_by_type.items())),
        "candidate_events_by_lane": dict(sorted(event_counts_by_lane.items())),
        "tracking_diagnostics_by_lane": {
            lane["id"]: {
                "lane_name": lane["name"],
                "short_occlusion_recoveries": rule.short_occlusion_recoveries,
                "long_gap_resets": rule.long_gap_resets,
                "occlusion_tolerance_frames": rule.max_occlusion_gap_frames,
                "tracklet_stitches": rule.tracklet_stitches,
                "tracklet_stitch_tolerance_frames": rule.max_tracklet_stitch_gap_frames,
            }
            for lane, rule_kind, rule in rule_engines if rule_kind == "wrong_way"
        },
        "lane_count": len(lane_contexts),
        "enabled_rules": sorted({rule_kind for _lane, rule_kind, _rule in rule_engines}),
        "configuration_used": {
            "lanes": [
                {
                    "lane_id": lane["id"],
                    "lane_name": lane["name"],
                    "controlled_approach_normalized": lane["config"]["forbidden_region"]["normalized_polygon"],
                    "legal_direction_normalized": lane["direction_points"],
                    "enabled_rules": lane["config"].get("enabled_rules", {}),
                }
                for lane in lane_contexts
            ],
            # Backward-compatible first-lane fields for existing result viewers.
            "controlled_approach_normalized": lane_contexts[0]["config"]["forbidden_region"]["normalized_polygon"],
            "legal_direction_normalized": lane_contexts[0]["direction_points"],
        },
        "annotated_video_resolution": [annotated_width, annotated_height],
        "annotated_video_fps": round(annotated_fps, 3),
        "red_signal_by_lane": {
            lane["id"]: {
                "lane_name": lane["name"],
                "red_frames": lane["red_signal_frames"],
                "unknown_frames": lane["unknown_signal_frames"],
                "red_frame_percent": round(lane["red_signal_frames"] / frame_index * 100, 3) if frame_index else 0.0,
                "unknown_frame_percent": round(lane["unknown_signal_frames"] / frame_index * 100, 3) if frame_index else 0.0,
            }
            for lane in lane_contexts if lane["signal_detector"]
        },
        "red_signal_frames": lane_contexts[0]["red_signal_frames"] if lane_contexts[0]["signal_detector"] else None,
        "red_signal_frame_percent": round(lane_contexts[0]["red_signal_frames"] / frame_index * 100, 3) if lane_contexts[0]["signal_detector"] and frame_index else None,
        "input_video_bytes": input_bytes,
        "metadata_bytes": metadata_bytes,
        "evidence_bytes": evidence_bytes,
        "metadata_reduction_percent_vs_video": round((1 - metadata_bytes / input_bytes) * 100, 3) if input_bytes else None,
        "metadata_plus_evidence_bytes": metadata_bytes + evidence_bytes,
        "metadata_plus_evidence_reduction_percent_vs_video": round(
            (1 - (metadata_bytes + evidence_bytes) / input_bytes) * 100, 3
        ) if input_bytes else None,
        "outputs": {
            "configuration": str((output_dir / "config_used.json").resolve()),
            "annotated_video": str(annotated_path.resolve()),
            "events": str(event_path.resolve()),
            "incidents": str(incidents_path.resolve()),
            "evidence_directory": str(evidence_dir.resolve()),
            "run_report": str((output_dir / "RUN_REPORT.md").resolve()),
            "integrity_manifest": str((output_dir / "EVIDENCE_MANIFEST.json").resolve()),
            "evidence_bundle": str((output_dir / "evidence_bundle.zip").resolve()),
        },
    }
    write_run_report(output_dir / "RUN_REPORT.md", summary, recorded_events)
    with (output_dir / "summary.json").open("w", encoding="utf-8") as stream:
        json.dump(summary, stream, indent=2, ensure_ascii=False)
    if evidence_config.get("export_bundle", False):
        create_evidence_package(output_dir, [
            output_dir / "config_used.json",
            event_path,
            incidents_path,
            output_dir / "RUN_REPORT.md",
            output_dir / "summary.json",
            evidence_dir,
        ])
    return summary
