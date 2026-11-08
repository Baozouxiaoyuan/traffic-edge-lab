"""Record decision inputs and inspect them without treating association as ground truth."""
from __future__ import annotations

import base64
import copy
import hashlib
import json
import math
from pathlib import Path
import re

import cv2

from .storage import atomic_write_json

RULES = {"candidate_red_signal_violation": "red_light", "candidate_wrong_way_driving": "wrong_way",
         "candidate_restricted_zone_entry": "restricted_zone"}


class DiagnosticRecorder:
    def __init__(self, stream, source: Path, config: dict, config_hash: str, width: int, height: int, fps: float):
        self.stream, self.width, self.height, self.fps = stream, width, height, fps
        with source.open("rb") as video:
            source_hash = hashlib.file_digest(video, "sha256").hexdigest()
        self._write({"kind": "header", "schema_version": 1, "source_sha256": source_hash,
                     "config_sha256": config_hash, "width": width, "height": height, "fps": fps,
                     "configuration": config,
                     "detector_note": "Captured boxes have already passed configured class, confidence and NMS filters."})

    def _write(self, value: dict) -> None:
        self.stream.write(json.dumps(value, allow_nan=False, separators=(",", ":")) + "\n")

    def box(self, box) -> list[float]:
        return [round(float(v) / (self.width if i % 2 == 0 else self.height), 6) for i, v in enumerate(box)]

    def record(self, frame: int, observations: list, lanes: list, *, assigned: dict, state_lanes: dict,
               statuses: dict, recent_exit: set, last_owned_frame: dict, decisions: dict,
               raw_detections: list | None, raw_status: str) -> None:
        tracks = []
        for observation in observations:
            track_id = observation.track_id
            owner, state_lane = assigned.get(track_id), state_lanes.get(track_id)
            tracks.append({"track_id": track_id, "class_name": observation.class_name,
                           "confidence": float(observation.confidence), "bbox": self.box(observation.bbox),
                           "ground_point": [round(observation.ground_point[0] / self.width, 6),
                                            round(observation.ground_point[1] / self.height, 6)],
                           "provisional": track_id >= 1_000_000, "assignment_status": statuses.get(track_id),
                           "lane_id": owner["id"] if owner else None,
                           "state_lane_id": state_lane["id"] if state_lane else None,
                           "recent_lane_exit": track_id in recent_exit,
                           "frames_since_lane_owned": frame-last_owned_frame[track_id] if track_id in last_owned_frame else None,
                           "rules": decisions.get(track_id, [])})
        signals = {lane["id"]: {"state": lane["signal_detector"].last_state if lane["signal_detector"] else "not_configured",
                                "is_red": bool(lane["signal_is_red"]),
                                "confidence": float(lane["signal_detector"].last_confidence) if lane["signal_detector"] else None}
                   for lane in lanes}
        raw = None if raw_detections is None else [
            {**item, "bbox": self.box(item["bbox"])} for item in raw_detections
        ]
        self._write({"kind": "frame", "frame_index": frame, "time": round((frame-1)/self.fps, 6),
                     "detections": raw, "raw_detection_status": raw_status, "tracks": tracks, "signals": signals})


def _iou(first: list, second: list) -> float:
    left, top = max(first[0], second[0]), max(first[1], second[1])
    right, bottom = min(first[2], second[2]), min(first[3], second[3])
    overlap = max(0, right-left) * max(0, bottom-top)
    union = max(0, first[2]-first[0])*max(0, first[3]-first[1]) + max(0, second[2]-second[0])*max(0, second[3]-second[1]) - overlap
    return overlap / union if union else 0.0


def _compatibility(observation: dict, error: dict) -> tuple[float, float]:
    box = error.get("bbox_normalized")
    point = error.get("ground_point_normalized") or ([.5*(box[0]+box[2]), box[3]] if box else None)
    candidate = observation.get("ground_point") or [.5*(observation["bbox"][0]+observation["bbox"][2]), observation["bbox"][3]]
    distance = math.dist(point, candidate) if point else math.inf
    return _iou(observation["bbox"], box) if box else 0.0, distance


def inspect_trace(path: Path, error: dict, selected_track: int | None = None) -> dict:
    target = float(error["timestamp_seconds"])
    frames = []
    with path.open(encoding="utf-8") as stream:
        header = json.loads(next(stream))
        if header.get("kind") != "header" or header.get("schema_version") != 1:
            raise ValueError("Unsupported diagnostic trace.")
        for line in stream:
            row = json.loads(line)
            if row["time"] > target + 2:
                break
            if row["time"] >= max(0, target-2):
                frames.append(row)
    if not frames:
        raise ValueError("The trace contains no frames around this event.")
    anchor = min(frames, key=lambda row: abs(row["time"]-target))
    candidates = {}
    for row in frames:
        if abs(row["time"]-target) > .5:
            continue
        for track in row["tracks"]:
            overlap, distance = _compatibility(track, error)
            if overlap < .05 and distance > .12:
                continue
            rank = (abs(row["time"]-target), -overlap, distance)
            previous = candidates.get(track["track_id"])
            if previous is None or rank < previous["rank"]:
                candidates[track["track_id"]] = {"track_id": track["track_id"], "class_name": track["class_name"],
                    "time": row["time"], "iou": round(overlap, 4), "distance": round(distance, 4), "rank": rank}
    ordered = sorted(candidates.values(), key=lambda item: item["rank"])[:12]
    if selected_track is not None and selected_track not in {t["track_id"] for row in frames for t in row["tracks"]}:
        raise ValueError("The selected track does not appear in this diagnostic window.")
    if selected_track is None:
        strong = [t for t in anchor["tracks"] if _compatibility(t, error)[0] >= .35]
        competing = [t for t in anchor["tracks"] if _compatibility(t, error)[0] >= .15]
        if len(strong) == 1 and len(competing) == 1:
            selected_track = strong[0]["track_id"]
    kind = RULES.get(error["candidate_event"])
    timeline = []
    for row in frames:
        track = next((t for t in row["tracks"] if t["track_id"] == selected_track), None)
        decision = next((d for d in (track or {}).get("rules", [])
                         if d["rule"] == kind and d["lane_id"] == error.get("lane_id")), None)
        timeline.append({"frame_index": row["frame_index"], "time": row["time"],
                         "detection_count": len(row["detections"]) if row["detections"] is not None else None,
                         "track": track, "signal": row["signals"].get(error.get("lane_id")), "decision": decision})
    result = {"available": True, "selected_track": selected_track,
              "candidates": [{k: v for k, v in item.items() if k != "rank"} for item in ordered],
              "anchor": {"frame_index": anchor["frame_index"], "time": anchor["time"], "tracks": anchor["tracks"],
                         "detections": anchor["detections"], "raw_detection_status": anchor["raw_detection_status"]},
              "timeline": timeline, "detector_filters": header["configuration"].get("yolo", {}),
              "association_note": "Candidates are matched to one labelled box near the event time. A reviewer must verify vehicle identity, especially across gaps or changed IDs."}
    if selected_track is None:
        compatible_detections = [d for d in anchor.get("detections") or [] if _compatibility(d, error)[0] >= .15]
        if ordered:
            result.update(stage="association", finding="Vehicle association needs confirmation. Select a track after checking the labelled vehicle and the frame overlay.")
        elif anchor["raw_detection_status"] == "before_association":
            result.update(stage="tracking" if compatible_detections else "detection",
                          finding="A filtered detector box overlaps the label, but no compatible track was returned." if compatible_detections else
                                  "No detector box surviving the configured filters overlaps the labelled vehicle at the event frame.")
        else:
            result.update(stage="insufficient_recording", finding="No compatible tracked vehicle is recorded. This trace cannot separate detection filtering from tracking association.")
        return result
    emitted = [row for row in timeline if row["decision"] and row["decision"].get("emitted")]
    if emitted:
        result.update(stage="event_matching", finding=f"The selected ID emitted a candidate at {emitted[0]['time']:.3f} s. Compare its time, lane and position with the label and matching tolerances.")
        return result
    crossings = [row for row in timeline if row["decision"] and
                 (row["decision"]["checks"].get("crossed_finite_stop_line") or row["decision"]["checks"].get("entry_observed"))]
    focus = min(crossings or timeline, key=lambda row: abs(row["time"]-target))
    track, decision = focus["track"], focus["decision"]
    result["focus_time"] = focus["time"]
    if track is None:
        result.update(stage="tracking_gap", finding="This selected ID has no observation at the event frame. The trace does not establish whether a nearby different ID is the same vehicle.")
    elif track["provisional"] and kind != "wrong_way":
        result.update(stage="tracking", finding="The vehicle has a provisional fallback ID. Crossing and zone-entry rules require a confirmed tracker ID.")
    elif decision is None or decision["checks"].get("lane_output_eligible") is False:
        result.update(stage="lane_assignment", finding="The selected observation was not eligible to emit this rule for the labelled lane. Check ownership, overlap ambiguity and the lane-exit grace period.")
    else:
        blockers = [name for name, passed in decision["checks"].items() if passed is False]
        result["blocking_checks"] = blockers
        if "not_already_fired" in blockers:
            result.update(stage="event_history", finding="This rule had already fired for the selected ID. Check the earlier event and vehicle identity before treating this label as a separate missed event.")
        elif any(name in blockers for name in ("signal_red", "previous_signal_red")) and crossings:
            result.update(stage="signal", finding="The recorded crossing failed the current or previous red-signal gate. Inspect the controlling signal region and its recorded state.")
        elif any(name in blockers for name in ("previous_observation", "track_mature", "approach_observed")):
            result.update(stage="track_history", finding="The rule lacks sufficient continuous approach history or track age at this frame.")
        elif "crossed_finite_stop_line" in blockers:
            result.update(stage="crossing_geometry", finding="No qualifying crossing of the configured finite stop line was recorded for this ID near the label time. Check the line, ground point and vehicle association.")
        else:
            result.update(stage="rule_conditions", finding="The rule did not emit because recorded conditions were unmet. The timeline lists the actual gates and threshold values.")
    return result


def _context(run_name: str, error_id: str) -> tuple[Path, dict, dict, dict]:
    from . import study
    directory = study.report_directory(run_name)
    report = json.loads((directory / "report.json").read_text(encoding="utf-8"))
    error = next((e for e in report["errors"] if e["event_id"] == error_id), None)
    if error is None:
        raise ValueError("Evaluation error not found.")
    inputs = json.loads((directory / "study_inputs.json").read_text(encoding="utf-8"))
    case = next((c for c in inputs["cases"] if c["id"] == error["case_id"]), None)
    if case is None or not re.fullmatch(r"[0-9a-f]{16}", case["id"]):
        raise ValueError("Frozen study case not found.")
    source = study.data_path(case["input_video"])
    if study.file_hash(source) != case["source_sha256"]:
        raise ValueError("The recording changed after this batch.")
    return directory, inputs, case, error


def _trace_location(directory: Path, case: dict) -> tuple[Path, str] | None:
    from .study import file_hash
    originals = [(directory / case["id"], "original")]
    replays = [(p.parent, "replay") for p in sorted(
        (directory / "diagnostic-replays").glob(case["id"] + "-*/summary.json"), reverse=True)]
    for folder, mode in originals + replays:
        path = folder / "decision_trace.jsonl"
        if not path.is_file() or not (folder / "summary.json").is_file():
            continue
        if mode == "replay":
            provenance = json.loads((folder / "diagnosis_provenance.json").read_text(encoding="utf-8"))
            if not provenance.get("complete"):
                continue
        summary = json.loads((folder / "summary.json").read_text(encoding="utf-8"))
        if summary.get("decision_trace_sha256") != file_hash(path):
            raise ValueError("The diagnostic trace no longer matches its recorded checksum.")
        with path.open(encoding="utf-8") as stream:
            header = json.loads(next(stream))
        if header.get("source_sha256") != case["source_sha256"] or header.get("config_sha256") != summary["config_sha256"]:
            raise ValueError("The diagnostic trace belongs to different inputs.")
        return path, mode
    return None


def diagnose_error(run_name: str, error_id: str, selected_track: int | None = None) -> dict:
    from . import study
    directory, inputs, case, error = _context(run_name, error_id)
    trace = _trace_location(directory, case)
    if trace is None:
        return {"available": False, "finding": "This batch predates frame traces. Replay this case with its saved configuration and model to inspect a new diagnostic run."}
    result = inspect_trace(trace[0], error, selected_track)
    result.update(mode=trace[1], case_name=case["name"], error_id=error_id,
                  trace_note="Recorded during the evaluated batch." if trace[1] == "original" else
                             "Diagnostic replay using the current implementation and saved model/settings. It may differ from the original run.")
    capture = cv2.VideoCapture(str(study.data_path(case["input_video"])))
    try:
        capture.set(cv2.CAP_PROP_POS_FRAMES, result["anchor"]["frame_index"]-1)
        ok, frame = capture.read()
        if ok:
            height, width = frame.shape[:2]
            boxes = [(d["bbox"], "D", (180, 180, 180)) for d in result["anchor"]["detections"] or []]
            boxes += [(t["bbox"], str(t["track_id"]), (210, 140, 0)) for t in result["anchor"]["tracks"]]
            if error.get("bbox_normalized"):
                boxes.append((error["bbox_normalized"], "LABEL", (0, 230, 255)))
            for box, text, color in boxes:
                x1, y1, x2, y2 = [round(v*(width if i % 2 == 0 else height)) for i, v in enumerate(box)]
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, text, (max(0, x1), max(16, y1-5)), cv2.FONT_HERSHEY_SIMPLEX, .55, color, 2)
            encoded, data = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            if encoded:
                result["anchor_image"] = "data:image/jpeg;base64," + base64.b64encode(data).decode("ascii")
    finally:
        capture.release()
    return result


def replay_error(run_name: str, error_id: str, output: Path, device: str, progress=None) -> None:
    from . import study
    from .pipeline import run_pipeline
    from .compute import resolve_device
    directory, inputs, case, error = _context(run_name, error_id)
    if _trace_location(directory, case) is not None:
        return
    model = (study.ROOT / inputs["model"]).resolve()
    if not model.is_relative_to(study.ROOT.resolve()) or not model.is_file() or study.file_hash(model) != inputs["model_sha256"]:
        raise ValueError("The original detector weights are missing or changed. Restore the exact saved model before replaying.")
    config_path = directory / case["id"] / "config_used.json"
    original_summary = json.loads((directory / case["id"] / "summary.json").read_text(encoding="utf-8"))
    if study.file_hash(config_path) != original_summary["config_sha256"]:
        raise ValueError("The original camera configuration changed.")
    config = copy.deepcopy(json.loads(config_path.read_text(encoding="utf-8")))
    config["diagnostics"] = {"record_frames": True}
    resolved = resolve_device(device)
    config.setdefault("yolo", {}).update(model=str(model), device="cuda" if resolved.startswith("cuda") else "cpu")
    output.mkdir(parents=True, exist_ok=False)
    replay_config = output / "replay_config.json"
    atomic_write_json(replay_config, config)
    provenance = {"complete": False, "original_run": run_name, "case_id": case["id"],
                  "original_config_sha256": original_summary["config_sha256"],
                  "model_sha256": inputs["model_sha256"], "implementation_hashes": study.implementation_hashes()}
    atomic_write_json(output / "diagnosis_provenance.json", provenance)
    run_pipeline(study.data_path(case["input_video"]), output, replay_config, "yolo", progress_callback=progress)
    if (study.file_hash(model) != inputs["model_sha256"] or study.file_hash(study.data_path(case["input_video"])) != case["source_sha256"]
            or study.file_hash(config_path) != provenance["original_config_sha256"]
            or study.implementation_hashes() != provenance["implementation_hashes"]):
        raise ValueError("Replay inputs changed during processing. Diagnostic results were withheld.")
    provenance["complete"] = True
    atomic_write_json(output / "diagnosis_provenance.json", provenance)
