"""Persist reviewed cases and evaluate one frozen model on one declared split."""
from __future__ import annotations

import copy
import csv
import hashlib
import json
import math
from pathlib import Path
import re
import threading
from typing import Callable

import cv2

from .evaluation import evaluate_study
from .pipeline import create_event_clips, run_pipeline
from .scope import validate_config_scope
from .storage import atomic_write_json

ROOT = Path(__file__).resolve().parents[2]
REGISTRY = ROOT / "data" / "study_registry.json"
WRITE_LOCK = threading.RLock()
SPLITS = {"development", "validation", "test"}


def file_hash(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def implementation_hashes() -> dict[str, str]:
    return {path.name: file_hash(path) for path in sorted(Path(__file__).parent.glob("*.py"))}


def data_path(value: str) -> Path:
    path = (ROOT / value).resolve()
    if not path.is_relative_to((ROOT / "data").resolve()) or not path.is_file():
        raise ValueError("Study inputs must be existing files inside the project data directory.")
    return path


def read_registry() -> dict:
    if not REGISTRY.is_file():
        return {"schema_version": 1, "revision": 0, "cases": []}
    return json.loads(REGISTRY.read_text(encoding="utf-8"))


def _reviewed_inputs(video: Path, truth: Path, config: dict) -> tuple[dict, list]:
    review = json.loads(truth.with_suffix(".review.json").read_text(encoding="utf-8"))
    labels = json.loads(truth.read_text(encoding="utf-8"))
    if not isinstance(labels, list):
        raise ValueError("Ground truth must contain a list of vehicle events.")
    duration = float(review.get("duration_seconds", 0))
    if not math.isfinite(duration) or duration <= 0:
        raise ValueError("The finalized review has no valid duration.")
    coverage = float(review.get("review_coverage_percent", 0))
    if not review.get("review_complete") or not math.isfinite(coverage) or coverage < 98:
        raise ValueError("Finalize the original-video review with at least 98% playback coverage first.")
    if not str(review.get("reviewer", "")).strip():
        raise ValueError("The finalized review requires a reviewer identity.")
    if review.get("source_sha256") != file_hash(video):
        raise ValueError("The annotation belongs to different video content.")
    if int(review.get("confirmed_events", -1)) != len(labels):
        raise ValueError("The review record and label count disagree.")
    if bool(review.get("negative_video")) != (not labels):
        raise ValueError("The review record and negative-video status disagree.")
    validate_config_scope(config)
    event_rules = {"candidate_red_signal_violation": "red_light",
                   "candidate_wrong_way_driving": "wrong_way",
                   "candidate_restricted_zone_entry": "restricted_zone"}
    lanes = config.get("lanes") or [{"id": "lane-1", **config}]
    by_id = {str(lane.get("id", "lane-1")): lane for lane in lanes}
    for label in labels:
        timestamp = float(label["timestamp_seconds"])
        point = label.get("vehicle_point_normalized", [])
        lane = by_id.get(str(label.get("lane_id", "")))
        rule = event_rules.get(label.get("candidate_event"))
        if not math.isfinite(timestamp) or not 0 <= timestamp < duration:
            raise ValueError("A label lies outside the reviewed video.")
        if len(point) != 2 or any(not math.isfinite(float(x)) or not 0 <= float(x) <= 1 for x in point):
            raise ValueError("Every study label needs a normalized vehicle ground point.")
        if lane is None or rule is None or not lane.get("enabled_rules", config.get("enabled_rules", {})).get(rule):
            raise ValueError("A labelled lane or rule is absent from the frozen camera setup.")
    return review, labels


def case_issues(case: dict) -> list[str]:
    try:
        video = data_path(case["input_video"])
        truth = data_path(case["ground_truth"])
        for key, path in (("source_sha256", video), ("truth_sha256", truth),
                          ("review_sha256", truth.with_suffix(".review.json"))):
            if file_hash(path) != case[key]:
                raise ValueError("Saved input content changed. Register the reviewed case again before running.")
        _reviewed_inputs(video, truth, case["configuration"])
        return []
    except (OSError, ValueError, KeyError, TypeError) as error:
        return [str(error)]


def _check_groups(cases: list[dict]) -> None:
    for key in ("source_group_id", "camera_id"):
        test = {c[key].strip().casefold() for c in cases if c["split"] == "test"}
        development = {c[key].strip().casefold() for c in cases if c["split"] != "test"}
        if test & development:
            raise ValueError(f"Test sources overlap development or validation by {key}: {', '.join(sorted(test & development))}.")
    hashes = [c["source_sha256"] for c in cases]
    if len(hashes) != len(set(hashes)):
        raise ValueError("Identical video content is already registered, even if its filename or group changed.")


def register_case(video: Path, truth: Path, config: dict, *, name: str, source_group: str,
                  camera_id: str, split: str, expected_revision: int) -> dict:
    name, source_group, camera_id = (str(value).strip() for value in (name, source_group, camera_id))
    if not all((name, source_group, camera_id)) or max(map(len, (name, source_group, camera_id))) > 120:
        raise ValueError("Enter a case name, source group and physical camera ID of at most 120 characters each.")
    if split not in SPLITS:
        raise ValueError("Choose development, validation or test.")
    video, truth = data_path(str(video)), data_path(str(truth))
    review, labels = _reviewed_inputs(video, truth, config)
    digest = file_hash(video)
    case = {
        "id": digest[:16], "name": name, "source_group_id": source_group,
        "camera_id": camera_id, "split": split,
        "input_video": video.relative_to(ROOT).as_posix(),
        "ground_truth": truth.relative_to(ROOT).as_posix(),
        "source_sha256": digest, "truth_sha256": file_hash(truth),
        "review_sha256": file_hash(truth.with_suffix(".review.json")),
        "configuration": copy.deepcopy(config), "duration_seconds": review["duration_seconds"],
        "reviewer": review["reviewer"], "label_count": len(labels), "negative_video": not labels,
    }
    with WRITE_LOCK:
        registry = read_registry()
        if registry["revision"] != expected_revision:
            raise ValueError("The study register changed in another request. Refresh it and try again.")
        previous = next((c for c in registry["cases"] if c["id"] == case["id"]), None)
        if previous and previous["split"] != "test" and split == "test":
            raise ValueError("A development or validation video cannot be promoted into the test split. Use new test footage.")
        cases = [c for c in registry["cases"] if c["id"] != case["id"]] + [case]
        _check_groups(cases)
        registry.update(cases=cases, revision=registry["revision"] + 1)
        atomic_write_json(REGISTRY, registry)
    return case


def remove_case(case_id: str, expected_revision: int) -> None:
    with WRITE_LOCK:
        registry = read_registry()
        if registry["revision"] != expected_revision:
            raise ValueError("The study register changed. Refresh it before removing a case.")
        remaining = [c for c in registry["cases"] if c["id"] != case_id]
        if len(remaining) == len(registry["cases"]):
            raise ValueError("Study case not found.")
        registry.update(cases=remaining, revision=registry["revision"] + 1)
        atomic_write_json(REGISTRY, registry)


def snapshot_cases(split: str) -> list[dict]:
    if split not in SPLITS:
        raise ValueError("Choose a study split.")
    with WRITE_LOCK:
        registry = read_registry()
        _check_groups(registry["cases"])
        cases = copy.deepcopy([c for c in registry["cases"] if c["split"] == split])
    if not cases:
        raise ValueError("Register at least one reviewed case in this split first.")
    for case in cases:
        issues = case_issues(case)
        if issues:
            raise ValueError(f"{case['name']}: {' '.join(issues)}")
    return cases


def report_directory(name: str) -> Path:
    if not re.fullmatch(r"study-\d{8}-\d{6}-[0-9a-f]{12}", name):
        raise ValueError("Invalid study run identifier.")
    path = (ROOT / "runs" / name).resolve()
    if not path.is_relative_to((ROOT / "runs").resolve()) or not path.is_dir():
        raise ValueError("Study run not found.")
    return path


def _write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            # A user-entered source/name must remain text in spreadsheet applications.
            writer.writerow({k: "'" + v if isinstance(v, str) and v.startswith(("=", "+", "-", "@")) else v
                             for k, v in row.items()})


def _error_evidence(video: Path, output: Path, errors: list[dict]) -> None:
    if not errors:
        return
    evidence = output / "evidence"
    evidence.mkdir(exist_ok=True)
    capture = cv2.VideoCapture(str(video))
    try:
        for error in errors:
            capture.set(cv2.CAP_PROP_POS_MSEC, float(error["timestamp_seconds"]) * 1000)
            ok, frame = capture.read()
            if ok:
                filename = error["event_id"] + ".jpg"
                if cv2.imwrite(str(evidence / filename), frame):
                    error["frame"] = "evidence/" + filename
                box = error.get("bbox_normalized")
                if box and len(box) == 4:
                    height, width = frame.shape[:2]
                    x1, y1, x2, y2 = [round(max(0, min(1, float(v))) * (width if i % 2 == 0 else height))
                                       for i, v in enumerate(box)]
                    crop = frame[y1:y2, x1:x2]
                    crop_name = error["event_id"] + "-crop.jpg"
                    if crop.size and cv2.imwrite(str(evidence / crop_name), crop):
                        error["crop"] = "evidence/" + crop_name
    finally:
        capture.release()
    create_event_clips(video, evidence, errors)
    for error in errors:
        if error.get("evidence_clip_pointer"):
            error["clip"] = error["evidence_clip_pointer"]


def run_study(cases: list[dict], output: Path, *, model: Path, tracker: str, device: str,
              progress: Callable[[int, int, str], None] | None = None, backend: str = "yolo") -> dict:
    if not cases or len({c["split"] for c in cases}) != 1:
        raise ValueError("A batch must contain reviewed cases from exactly one split.")
    if tracker not in {"bytetrack", "botsort"}:
        raise ValueError("Choose ByteTrack or BoT-SORT.")
    from .compute import resolve_device
    resolved_device = resolve_device(device)
    runtime_device = "cuda" if resolved_device.startswith("cuda") else "cpu"
    model_hash = file_hash(model)
    _check_groups(cases)
    for case in cases:
        if case_issues(case):
            raise ValueError(f"{case['name']}: frozen inputs changed before processing.")
    output.mkdir(parents=True, exist_ok=False)
    inputs = {"cases": cases, "model": model.name, "model_sha256": model_hash,
              "tracker": tracker, "device": resolved_device, "implementation_hashes": implementation_hashes()}
    atomic_write_json(output / "study_inputs.json", inputs)
    split = cases[0]["split"]
    videos, rows, errors = [], [], []
    for index, case in enumerate(cases):
        if progress:
            progress(index, len(cases), case["name"])
        config = copy.deepcopy(case["configuration"])
        config.setdefault("yolo", {}).update(model=str(model.resolve()), tracker=tracker, device=runtime_device)
        config_path = output / f"{case['id']}-config.json"
        atomic_write_json(config_path, config)
        def frame_progress(current: int, total: int) -> None:
            if progress:
                progress(index, len(cases), f"{case['name']} · frame {current}/{total}")
        summary = run_pipeline(data_path(case["input_video"]), output / case["id"], config_path, backend,
                               progress_callback=frame_progress)
        videos.append({"video_id": case["id"], "source_group_id": case["source_group_id"],
                       "split": split, "status": "labelled", "duration_seconds": case["duration_seconds"],
                       "events_path": summary["outputs"]["events"],
                       "ground_truth_path": str(data_path(case["ground_truth"]))})
    manifest = {"study_name": output.name, "tolerance_seconds": .5, "spatial_tolerance_normalized": .12,
                "bootstrap_iterations": 2000, "bootstrap_seed": 20260923, "videos": videos}
    atomic_write_json(output / "study_manifest.json", manifest)
    metrics = evaluate_study(output / "study_manifest.json", evaluation_split=split)
    for case, result in zip(cases, metrics["per_video"]):
        score = result["overall"]
        rows.append({"case_id": case["id"], "name": case["name"], "split": split,
                     "source_group": case["source_group_id"], "negative_video": case["negative_video"],
                     "ground_truth_events": result["ground_truth_events"], "predicted_events": result["predicted_events"],
                     **{key: score[key] for key in ("true_positives", "false_positives", "false_negatives", "precision", "recall", "f1")}})
        case_errors = []
        for kind, events in (("FP", score["false_positive_events"]), ("FN", score["unmatched_ground_truth_events"])):
            for event in events:
                error = {"event_id": f"error-{len(errors) + len(case_errors):05d}", "kind": kind,
                         "case_id": case["id"], "name": case["name"],
                         "timestamp_seconds": event["timestamp_seconds"], "candidate_event": event["candidate_event"],
                         "lane_id": event.get("lane_id"),
                         "bbox_normalized": event.get("bbox_normalized") or event.get("vehicle_bbox_normalized")}
                case_errors.append(error)
        _error_evidence(data_path(case["input_video"]), output, case_errors)
        errors.extend(case_errors)
    if (model_hash != file_hash(model) or any(case_issues(c) for c in cases)
            or implementation_hashes() != inputs["implementation_hashes"]):
        raise ValueError("Study inputs changed during processing. Results were withheld. Start a fresh batch.")
    _write_csv(output / "cases.csv", rows, list(rows[0]))
    _write_csv(output / "errors.csv", errors,
               ["event_id", "kind", "case_id", "name", "timestamp_seconds", "candidate_event", "lane_id", "frame", "crop", "clip"])
    report = {"run_name": output.name, "split": split, "model": model.name, "tracker": tracker,
              "compute_device": resolved_device, "metrics": metrics, "cases": rows, "errors": errors, "inputs": inputs,
              "normal_video_count": sum(c["negative_video"] for c in cases),
              "metric_notes": "The evaluator stores zero-denominator metrics as 0. The interface displays them as undefined."}
    atomic_write_json(output / "report.json", report)
    if progress:
        progress(len(cases), len(cases), "Complete")
    return report
