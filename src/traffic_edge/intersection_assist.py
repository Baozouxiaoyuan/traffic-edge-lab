"""Video-based intersection calibration proposals for fixed CCTV.

Calibration proposals use three sources of evidence:

* vehicle trajectories propose movement corridors and travel directions;
* object detection proposes traffic-light rectangles;
* painted-line geometry proposes stop-line candidates perpendicular to a flow.

Every output is a proposal for human confirmation.  A single road mask is not
treated as a lane or as proof of traffic-control ownership.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from math import hypot
from pathlib import Path
from statistics import median
import time
import uuid

import cv2
import numpy as np

from .backends import UltralyticsByteTrackBackend
from .scene_calibration import (
    scene_evidence, marking_boundaries, recognize_signals,
    recognize_arrows, scene_overlay, signal_reference_evidence,
)
from .marking_curves import trace_markings, lane_regions


@dataclass
class Trajectory:
    track_id: int
    points: list[tuple[float, float]]
    widths: list[float]

    @property
    def start(self) -> np.ndarray:
        return np.median(np.asarray(self.points[: min(4, len(self.points))]), axis=0)

    @property
    def end(self) -> np.ndarray:
        return np.median(np.asarray(self.points[-min(4, len(self.points)) :]), axis=0)

    @property
    def vector(self) -> np.ndarray:
        return self.end - self.start

    @property
    def midpoint(self) -> np.ndarray:
        return (self.start + self.end) / 2.0


def _unit(vector: np.ndarray) -> np.ndarray:
    length = float(np.linalg.norm(vector))
    return vector / max(length, 1e-9)


def _trajectory_compatible(left: Trajectory, right: Trajectory) -> bool:
    direction_similarity = float(np.dot(_unit(left.vector), _unit(right.vector)))
    midpoint_distance = float(np.linalg.norm(left.midpoint - right.midpoint))
    # Direction is more important than exact lateral position. Nearby parallel
    # lanes are separated later using a tighter centreline split.
    return direction_similarity >= 0.82 and midpoint_distance <= 0.28


def _split_parallel_lanes(group: list[Trajectory]) -> list[list[Trajectory]]:
    if len(group) <= 1:
        return [group]
    direction = _unit(np.mean([item.vector for item in group], axis=0))
    perpendicular = np.asarray([-direction[1], direction[0]])
    ordered = sorted(group, key=lambda item: float(np.dot(item.midpoint, perpendicular)))
    widths = [median(item.widths) for item in ordered if item.widths]
    separation = max(0.035, (median(widths) if widths else 0.04) * 1.6)
    lanes: list[list[Trajectory]] = [[ordered[0]]]
    previous_offset = float(np.dot(ordered[0].midpoint, perpendicular))
    for item in ordered[1:]:
        offset = float(np.dot(item.midpoint, perpendicular))
        if offset - previous_offset > separation:
            lanes.append([])
        lanes[-1].append(item)
        previous_offset = offset
    return lanes


def cluster_trajectories(trajectories: list[Trajectory]) -> list[list[Trajectory]]:
    """Greedily cluster tracks into directional movement lanes.

    The implementation has no sklearn dependency and is intentionally
    conservative: a weak or short trajectory is filtered before this stage.
    """

    directional_groups: list[list[Trajectory]] = []
    for trajectory in sorted(trajectories, key=lambda item: -float(np.linalg.norm(item.vector))):
        best_index = None
        best_score = float("inf")
        for index, group in enumerate(directional_groups):
            representative = group[0]
            if not _trajectory_compatible(trajectory, representative):
                continue
            score = float(np.linalg.norm(trajectory.midpoint - representative.midpoint))
            if score < best_score:
                best_score, best_index = score, index
        if best_index is None:
            directional_groups.append([trajectory])
        else:
            directional_groups[best_index].append(trajectory)
    lanes: list[list[Trajectory]] = []
    for group in directional_groups:
        lanes.extend(_split_parallel_lanes(group))
    return lanes


def _movement_from_group(group: list[Trajectory], index: int) -> dict:
    direction = _unit(np.mean([item.vector for item in group], axis=0))
    starts = np.asarray([item.start for item in group])
    ends = np.asarray([item.end for item in group])
    start = np.median(starts, axis=0)
    end = np.median(ends, axis=0)
    perpendicular = np.asarray([-direction[1], direction[0]])
    typical_width = median([value for item in group for value in item.widths] or [0.04])
    padding = max(0.018, min(0.08, typical_width * 0.75))
    cloud: list[list[float]] = []
    for trajectory in group:
        for point in trajectory.points:
            base = np.asarray(point)
            cloud.append((base + perpendicular * padding).tolist())
            cloud.append((base - perpendicular * padding).tolist())
    hull = cv2.convexHull(np.asarray(cloud, dtype=np.float32)).reshape(-1, 2)
    hull_pixels = np.round(hull * 1000).astype(np.int32)
    simplified = cv2.approxPolyDP(hull_pixels, epsilon=12, closed=True).reshape(-1, 2) / 1000.0
    polygon = [[round(float(np.clip(x, 0, 1)), 6), round(float(np.clip(y, 0, 1)), 6)] for x, y in simplified]
    displacement = float(np.linalg.norm(end - start))
    confidence = min(0.95, 0.30 + 0.12 * len(group) + 0.55 * min(1.0, displacement / 0.45))
    horizontal = "right" if direction[0] > 0.20 else "left" if direction[0] < -0.20 else ""
    vertical = "down" if direction[1] > 0.20 else "up" if direction[1] < -0.20 else ""
    label = "-".join(part for part in (vertical, horizontal) if part) or "stationary"
    return {
        "proposal_id": index + 1,
        "name": f"Observed flow {index + 1} ({label})",
        "polygon": polygon,
        "direction": [
            [round(float(start[0]), 6), round(float(start[1]), 6)],
            [round(float(end[0]), 6), round(float(end[1]), 6)],
        ],
        "track_count": len(group),
        "confidence": round(confidence, 4),
        "review_required": True,
    }


def movement_proposals(raw_tracks: dict[int, list[tuple[float, float, float]]]) -> list[dict]:
    trajectories: list[Trajectory] = []
    for track_id, observations in raw_tracks.items():
        if len(observations) < 6:
            continue
        points = [(float(x), float(y)) for x, y, _width in observations]
        widths = [float(width) for _x, _y, width in observations]
        trajectory = Trajectory(track_id, points, widths)
        if float(np.linalg.norm(trajectory.vector)) < 0.075:
            continue
        trajectories.append(trajectory)
    groups = cluster_trajectories(trajectories)
    proposals = [_movement_from_group(group, index) for index, group in enumerate(groups)]
    return sorted(proposals, key=lambda item: (-item["track_count"], -item["confidence"]))


def detect_stop_line_candidates(
    frame: np.ndarray,
    direction: list[list[float]],
    polygon: list[list[float]] | None = None,
    limit: int = 5,
    marking_mask: np.ndarray | None = None,
    polygon_margin_pixels: float = 0,
    line_segments: list[list[float]] | None = None,
) -> list[dict]:
    """Find bright line segments perpendicular to a proposed vehicle flow."""

    height, width = frame.shape[:2]
    start = np.asarray(direction[0], dtype=float)
    end = np.asarray(direction[1], dtype=float)
    flow_vector=np.asarray([(end[0]-start[0])*width,(end[1]-start[1])*height])
    if not np.all(np.isfinite(flow_vector)) or np.linalg.norm(flow_vector)<1e-6:
        return []
    flow = _unit(flow_vector)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    white = cv2.inRange(hsv, np.asarray([0, 0, 145]), np.asarray([179, 95, 255]))
    edges = cv2.Canny(gray, 60, 170)
    white_edges = cv2.bitwise_and(edges, cv2.dilate(white, np.ones((3, 3), np.uint8)))
    # Generic edges alone include vehicle roofs, kerbs and shadows. Require
    # painted evidence; learned crosswalk labels are excluded by the caller.
    evidence = white_edges
    if marking_mask is not None:
        evidence = cv2.Canny(marking_mask, 40, 100)
    lines = np.asarray(line_segments,dtype=np.int32).reshape(-1,1,4) if line_segments is not None else cv2.HoughLinesP(
        evidence,
        1,
        np.pi / 180,
        threshold=max(14, int(min(width, height) * 0.02)),
        minLineLength=max(30, int(width * 0.035)),
        maxLineGap=max(16, int(width * 0.05)),
    )
    polygon_pixels = None
    if polygon and len(polygon) >= 3:
        polygon_pixels = np.asarray([[x * width, y * height] for x, y in polygon], dtype=np.float32)
    candidates: list[dict] = []
    for raw in [] if lines is None else lines[:, 0, :]:
        x1, y1, x2, y2 = (int(value) for value in raw)
        visible,first,last=cv2.clipLine((0,0,width,height),(x1,y1),(x2,y2))
        if not visible:
            continue
        x1,y1=first
        x2,y2=last
        if polygon_pixels is not None:
            sample_count=max(2,min(4096,round(np.hypot(x2-x1,y2-y1)/2)+1))
            samples = np.linspace([x1,y1],[x2,y2],sample_count)
            inside=np.flatnonzero([cv2.pointPolygonTest(polygon_pixels,tuple(map(float,point)),True)>=-polygon_margin_pixels for point in samples])
            if len(inside)<2:
                continue
            runs=np.split(inside,np.flatnonzero(np.diff(inside)>1)+1)
            owned=max(runs,key=len)
            if len(owned)<2:
                continue
            x1,y1=np.round(samples[owned[0]]).astype(int)
            x2,y2=np.round(samples[owned[-1]]).astype(int)
        segment = np.asarray([x2 - x1, y2 - y1], dtype=float)
        length = float(np.linalg.norm(segment))
        if length < 15:
            continue
        # Perspective projection does not preserve right angles.  A physical
        # stop line can look only mildly transverse to an oblique image-space
        # trajectory, so this is a ranking signal rather than a 90-degree gate.
        transverse_evidence = 1.0 - abs(float(np.dot(_unit(segment), flow)))
        midpoint = np.asarray([(x1 + x2) / 2, (y1 + y2) / 2], dtype=np.float32)
        if transverse_evidence < 0.03:
            continue
        if polygon_pixels is not None and cv2.pointPolygonTest(polygon_pixels, tuple(midpoint), True) < -polygon_margin_pixels:
            continue
        if marking_mask is not None:
            samples = np.linspace([x1,y1], [x2,y2], 40).astype(int)
            radius=max(2,min(12,round(polygon_margin_pixels/2))) if line_segments is not None else 2
            neighbourhood = cv2.dilate(marking_mask,np.ones((2*radius+1,2*radius+1),np.uint8))
            if np.mean(neighbourhood[np.clip(samples[:,1],0,height-1),np.clip(samples[:,0],0,width-1)] > 0) < .5:
                continue
        brightness = float(np.mean(gray[max(0, min(y1, y2) - 2) : min(height, max(y1, y2) + 3), max(0, min(x1, x2) - 2) : min(width, max(x1, x2) + 3)])) / 255.0
        score = 0.30 * transverse_evidence + 0.50 * min(1.0, length / (width * 0.30)) + 0.20 * brightness
        candidates.append({
            "points": [[round(x1 / width, 6), round(y1 / height, 6)], [round(x2 / width, 6), round(y2 / height, 6)]],
            "confidence": round(float(score), 4),
            "review_required": True,
            "classification": "transverse_marking_requires_stop_line_confirmation",
        })
    candidates.sort(key=lambda item: -item["confidence"])
    accepted: list[dict] = []
    for candidate in candidates:
        midpoint = np.mean(np.asarray(candidate["points"]), axis=0)
        if any(float(np.linalg.norm(midpoint - np.mean(np.asarray(other["points"]), axis=0))) < 0.035 for other in accepted):
            continue
        accepted.append(candidate)
        if len(accepted) >= limit:
            break
    return accepted


def _merge_signal_boxes(detections: list[dict]) -> list[dict]:
    from .scene_calibration import merge_signal_detections
    return merge_signal_detections(detections)


def analyze_intersection_video(
    video_path: Path,
    model_name: str,
    confidence: float = 0.15,
    image_size: int = 960,
    max_seconds: float = 20.0,
    reference_seconds: float = 0.0,
    learned_markings: bool = True,
) -> dict:
    """Analyze a short calibration window and return auditable proposals."""

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise ValueError(f"Could not open video: {video_path}")
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 25.0)
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps
    if not np.isfinite(reference_seconds) or reference_seconds < 0 or reference_seconds > duration:
        capture.release()
        raise ValueError("Setup-frame time must be inside the video")
    capture.set(cv2.CAP_PROP_POS_MSEC, min(reference_seconds, max(0,duration-1/fps))*1000)
    ok, reference = capture.read()
    if not ok:
        capture.release()
        raise ValueError("Could not read the selected setup frame")
    window_start = max(0, reference_seconds-max_seconds/2)
    capture.set(cv2.CAP_PROP_POS_MSEC, window_start*1000)
    frame_limit = max(1, int(max_seconds * fps))
    try:
        tracker = UltralyticsByteTrackBackend(model_name, confidence, 0.5, [2, 5, 7], image_size, "bytetrack")
        raw_tracks: dict[int, list[tuple[float, float, float]]] = defaultdict(list)
        sampled_frames: list[np.ndarray] = []
        sampled_boxes: list[list] = []
        frame_index = 0
        started = time.perf_counter()
        while frame_index < frame_limit:
            ok, frame = capture.read()
            if not ok:
                break
            height, width = frame.shape[:2]
            observations=tracker.track(frame)
            for observation in observations:
                x, y = observation.ground_point
                raw_tracks[observation.track_id].append((x / width, y / height, (observation.bbox[2] - observation.bbox[0]) / width))
            if frame_index % max(1, int(fps * 2.0)) == 0:
                sampled_frames.append(frame.copy())
                sampled_boxes.append([list(observation.bbox) for observation in observations])
            frame_index += 1
    finally:
        capture.release()
    movements = movement_proposals(raw_tracks)
    # Sampling the whole window supports persistence without doing heavy
    # tiled specialist inference on every frame.
    signal_frames = [reference]
    if sampled_frames:
        signal_frames += [sampled_frames[0], sampled_frames[-1]]
    signals, signal_metadata = recognize_signals(tracker.model, signal_frames, image_size)
    for signal in signals:
        signal['reference_evidence']=signal_reference_evidence(reference,signal['rectangle'])
    labels, semantic_metadata, boundaries, lanes, curves = None, None, [], [], []
    warnings = []
    if learned_markings:
        try:
            detections=tracker.model.predict(reference,conf=.10,classes=[0,1,2,3,5,7],imgsz=image_size,verbose=False)[0]
            reference_boxes=detections.boxes.xyxy.cpu().tolist() if detections.boxes is not None else []
            selected_frames=[sampled_frames[0],sampled_frames[-1]] if sampled_frames else []
            selected_boxes=[sampled_boxes[0],sampled_boxes[-1]] if sampled_boxes else []
            labels, semantic_metadata = scene_evidence(reference,selected_frames,reference_boxes,selected_boxes)
            boundaries = marking_boundaries(labels)
            curves = trace_markings(labels,boundaries)
            lanes = lane_regions(labels,curves,raw_tracks)
        except (OSError, RuntimeError, ImportError, ValueError) as exc:
            warnings.append(f"Learned marking analysis unavailable: {exc}")
    arrows, arrow_metadata = recognize_arrows(reference)
    if not arrow_metadata.get("available"):
        warnings.append("Turn-arrow recognition is unavailable; verify permitted turns manually.")
    mask = (labels == 24).astype(np.uint8)*255 if labels is not None else None
    transverse_mask=np.zeros(reference.shape[:2],np.uint8)
    transverse_segments=[]
    for boundary in boundaries:
        if boundary.get("role") != "transverse_marking":
            continue
        endpoints=np.round(np.asarray(boundary["points"])*[reference.shape[1],reference.shape[0]]).astype(int)
        transverse_segments.append(endpoints.flatten().tolist())
        cv2.line(transverse_mask,tuple(endpoints[0]),tuple(endpoints[1]),255,max(5,round(boundary["stroke_width"]*3)))
    if mask is not None:
        transverse_mask=cv2.bitwise_and(transverse_mask,mask)
    for movement in movements:
        movement["stop_line_candidates"] = (
            detect_stop_line_candidates(reference, movement["direction"], movement["polygon"], marking_mask=mask)
        )
        movement["geometry_role"] = "observed_trajectory_extent_not_physical_lane"
    for lane in lanes:
        # A transverse line can be ranked even when traffic direction is unknown.
        axis = lane["axis"]
        margin=lane.get("boundary_inset_pixels",0)+max(12,min(reference.shape[:2])*.015)
        lane["stop_line_candidates"] = detect_stop_line_candidates(reference,axis,lane["polygon"],limit=1,marking_mask=transverse_mask,polygon_margin_pixels=margin,line_segments=transverse_segments)
    return {
        "proposal_set_id":uuid.uuid4().hex,
        "method": "learned_scene_and_tiled_signals_v2",
        "frames_analyzed": frame_index,
        "seconds_analyzed": round(frame_index / max(fps, 1e-9), 3),
        "processing_seconds": round(time.perf_counter() - started, 3),
        "movement_proposals": movements,
        "lane_proposals": lanes,
        "marking_boundaries": boundaries,
        "marking_curves": curves,
        "road_marking_proposals": arrows,
        "traffic_light_proposals": signals,
        "scene_model": semantic_metadata,
        "signal_detection": signal_metadata,
        "arrow_detection": arrow_metadata,
        "warnings": warnings,
        "reference_seconds": round(reference_seconds,3),
        "window_start_seconds": round(window_start,3),
        "scene_preview": scene_overlay(reference, labels, boundaries, signals, lanes),
        "limitations": [
            "Marked lanes and observed vehicle paths are separate proposals; missing paint remains unknown.",
            "Traffic-light boxes are object proposals, not proof that a signal controls a movement.",
            "Stop-line candidates are ranked painted segments and require human confirmation.",
        ],
        "review_required": True,
    }
