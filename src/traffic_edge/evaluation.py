from __future__ import annotations

import json
from math import hypot
from pathlib import Path
import random
from statistics import mean
from typing import Any


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def evaluate_events(
    predicted: list[dict[str, Any]],
    ground_truth: list[dict[str, Any]],
    tolerance_seconds: float = 0.5,
    spatial_tolerance_normalized: float = 0.12,
    _include_diagnostics: bool = True,
) -> dict[str, Any]:
    """Match event type, time and, when available, normalized vehicle location."""
    unmatched_truth = set(range(len(ground_truth)))
    matches: list[dict[str, Any]] = []
    false_positives: list[str] = []

    for event in sorted(predicted, key=lambda item: float(item["timestamp_seconds"])):
        candidates = []
        for index in unmatched_truth:
            truth = ground_truth[index]
            if truth["candidate_event"] != event["candidate_event"]:
                continue
            if truth.get("lane_id") and event.get("lane_id") and truth["lane_id"] != event["lane_id"]:
                continue
            difference = abs(float(truth["timestamp_seconds"]) - float(event["timestamp_seconds"]))
            if difference > tolerance_seconds:
                continue
            event_point = event.get("ground_point_normalized")
            truth_point = truth.get("vehicle_point_normalized")
            spatial_error = None
            if event_point is not None and truth_point is not None:
                spatial_error = hypot(
                    float(event_point[0]) - float(truth_point[0]),
                    float(event_point[1]) - float(truth_point[1]),
                )
                if spatial_error > spatial_tolerance_normalized:
                    continue
            candidates.append((difference, spatial_error if spatial_error is not None else 0.0, index))
        if not candidates:
            false_positives.append(str(event.get("event_id", "unknown")))
            continue
        difference, spatial_error, truth_index = min(candidates)
        unmatched_truth.remove(truth_index)
        matches.append(
            {
                "event_id": event.get("event_id"),
                "ground_truth_index": truth_index,
                "timestamp_error_seconds": round(difference, 3),
                "spatial_error_normalized": round(spatial_error, 4),
            }
        )

    true_positives = len(matches)
    fp_count = len(false_positives)
    false_negatives = len(unmatched_truth)
    precision = true_positives / (true_positives + fp_count) if true_positives + fp_count else 0.0
    recall = true_positives / (true_positives + false_negatives) if true_positives + false_negatives else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    timestamp_errors = [float(item["timestamp_error_seconds"]) for item in matches]
    spatial_errors = [float(item["spatial_error_normalized"]) for item in matches]
    result = {
        "tolerance_seconds": tolerance_seconds,
        "spatial_tolerance_normalized": spatial_tolerance_normalized,
        "true_positives": true_positives,
        "false_positives": fp_count,
        "false_negatives": false_negatives,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "matches": matches,
        "false_positive_event_ids": false_positives,
        "false_positive_events": [
            event for event in predicted if str(event.get("event_id", "unknown")) in false_positives
        ],
        "unmatched_ground_truth_indices": sorted(unmatched_truth),
        "unmatched_ground_truth_events": [ground_truth[index] for index in sorted(unmatched_truth)],
        "localization_errors": {
            "mean_timestamp_error_seconds": round(mean(timestamp_errors), 4) if timestamp_errors else None,
            "p95_timestamp_error_seconds": round(_percentile(timestamp_errors, 0.95), 4) if timestamp_errors else None,
            "mean_spatial_error_normalized": round(mean(spatial_errors), 4) if spatial_errors else None,
            "p95_spatial_error_normalized": round(_percentile(spatial_errors, 0.95), 4) if spatial_errors else None,
        },
    }
    if _include_diagnostics and predicted and any("evidence_quality_score" in event for event in predicted):
        result["evidence_threshold_diagnostics"] = [
            {
                "minimum_evidence_strength": threshold,
                **{
                    key: value for key, value in evaluate_events(
                        [event for event in predicted if float(event.get("evidence_quality_score", 0.0)) >= threshold],
                        ground_truth,
                        tolerance_seconds,
                        spatial_tolerance_normalized,
                        False,
                    ).items()
                    if key in {"true_positives", "false_positives", "false_negatives", "precision", "recall", "f1"}
                },
            }
            for threshold in (0.0, 0.25, 0.5, 0.75)
        ]
        result["evidence_threshold_warning"] = (
            "Post-hoc diagnostic only. Select a threshold on validation data and evaluate it once on a frozen test set."
        )
    return result


def evaluate_files(
    events_path: Path,
    ground_truth_path: Path,
    tolerance_seconds: float,
    spatial_tolerance_normalized: float = 0.12,
) -> dict[str, Any]:
    predicted = read_jsonl(events_path)
    with ground_truth_path.open("r", encoding="utf-8") as stream:
        ground_truth = json.load(stream)
    return evaluate_events(predicted, ground_truth, tolerance_seconds, spatial_tolerance_normalized)


def _score_counts(true_positives: int, false_positives: int, false_negatives: int) -> dict[str, Any]:
    precision = true_positives / (true_positives + false_positives) if true_positives + false_positives else 0.0
    recall = true_positives / (true_positives + false_negatives) if true_positives + false_negatives else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "true_positives": true_positives,
        "false_positives": false_positives,
        "false_negatives": false_negatives,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
    }


def _percentile(values: list[float], proportion: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return 0.0
    position = (len(ordered) - 1) * proportion
    lower = int(position)
    upper = min(len(ordered) - 1, lower + 1)
    fraction = position - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def _bootstrap_intervals(
    counts_by_sampling_unit: list[tuple[int, int, int]],
    iterations: int,
    seed: int,
) -> dict[str, list[float]] | None:
    if len(counts_by_sampling_unit) < 2 or iterations < 1:
        return None
    generator = random.Random(seed)
    samples = {"precision": [], "recall": [], "f1": []}
    for _ in range(iterations):
        selected = [generator.choice(counts_by_sampling_unit) for _index in counts_by_sampling_unit]
        score = _score_counts(
            sum(item[0] for item in selected),
            sum(item[1] for item in selected),
            sum(item[2] for item in selected),
        )
        for metric in samples:
            samples[metric].append(float(score[metric]))
    return {
        metric: [round(_percentile(values, 0.025), 4), round(_percentile(values, 0.975), 4)]
        for metric, values in samples.items()
    }


def evaluate_study(manifest_path: Path, evaluation_split: str = "test") -> dict[str, Any]:
    """Aggregate one declared split without silently mixing model selection and final test data."""
    manifest_path = manifest_path.resolve()
    with manifest_path.open("r", encoding="utf-8") as stream:
        manifest = json.load(stream)
    tolerance = float(manifest.get("tolerance_seconds", 0.5))
    spatial_tolerance = float(manifest.get("spatial_tolerance_normalized", 0.12))
    iterations = int(manifest.get("bootstrap_iterations", 2000))
    seed = int(manifest.get("bootstrap_seed", 20260923))
    records: list[dict[str, Any]] = []
    skipped: list[dict[str, str]] = []
    source_groups_by_split: dict[str, set[str]] = {}

    def resolve(value: str) -> Path:
        path = Path(value)
        return path if path.is_absolute() else (manifest_path.parent / path).resolve()

    for item in manifest.get("videos", []):
        split = str(item.get("split", "test"))
        source_group_id = str(
            item.get("source_group_id") or item.get("camera_id") or item.get("video_id", "unknown")
        )
        source_groups_by_split.setdefault(split, set()).add(source_group_id)
        if split != evaluation_split or item.get("status") != "labelled":
            skipped.append({"video_id": str(item.get("video_id", "unknown")), "reason": f"{split}/{item.get('status', 'unknown')}"})
            continue
        events_path = resolve(str(item["events_path"]))
        truth_path = resolve(str(item["ground_truth_path"]))
        predicted = read_jsonl(events_path)
        with truth_path.open("r", encoding="utf-8") as stream:
            truth = json.load(stream)
        overall = evaluate_events(predicted, truth, tolerance, spatial_tolerance)
        event_types = sorted({entry["candidate_event"] for entry in predicted + truth})
        by_type = {
            event_type: evaluate_events(
                [entry for entry in predicted if entry["candidate_event"] == event_type],
                [entry for entry in truth if entry["candidate_event"] == event_type],
                tolerance,
                spatial_tolerance,
            )
            for event_type in event_types
        }
        records.append({
            "video_id": str(item["video_id"]),
            "source_group_id": source_group_id,
            "duration_seconds": float(item["duration_seconds"]),
            "ground_truth_events": len(truth),
            "predicted_events": len(predicted),
            "overall": overall,
            "by_type": by_type,
        })
    if not records:
        raise ValueError(f"The study contains no labelled {evaluation_split} videos")

    overall_counts = [
        (record["overall"]["true_positives"], record["overall"]["false_positives"], record["overall"]["false_negatives"])
        for record in records
    ]
    source_group_counts: dict[str, list[int]] = {}
    for record, counts in zip(records, overall_counts):
        source_key = record["source_group_id"]
        aggregate = source_group_counts.setdefault(source_key, [0, 0, 0])
        for index, value in enumerate(counts):
            aggregate[index] += int(value)
    micro = _score_counts(
        sum(item[0] for item in overall_counts),
        sum(item[1] for item in overall_counts),
        sum(item[2] for item in overall_counts),
    )
    all_types = sorted({event_type for record in records for event_type in record["by_type"]})
    by_type: dict[str, Any] = {}
    for offset, event_type in enumerate(all_types):
        counts = []
        counts_by_source_group: dict[str, list[int]] = {}
        for record in records:
            metrics = record["by_type"].get(event_type)
            video_counts = (
                int(metrics["true_positives"]) if metrics else 0,
                int(metrics["false_positives"]) if metrics else 0,
                int(metrics["false_negatives"]) if metrics else 0,
            )
            counts.append(video_counts)
            source_key = record["source_group_id"]
            aggregate = counts_by_source_group.setdefault(source_key, [0, 0, 0])
            for index, value in enumerate(video_counts):
                aggregate[index] += int(value)
        score = _score_counts(sum(x[0] for x in counts), sum(x[1] for x in counts), sum(x[2] for x in counts))
        score["confidence_intervals_95"] = _bootstrap_intervals(
            [tuple(values) for values in counts_by_source_group.values()], iterations, seed + offset + 1
        )
        by_type[event_type] = score

    duration_hours = sum(record["duration_seconds"] for record in records) / 3600.0
    development_sources = source_groups_by_split.get("development", set()) | source_groups_by_split.get("validation", set())
    overlap = sorted(source_groups_by_split.get("test", set()) & development_sources)
    return {
        "study_name": manifest.get("study_name", manifest_path.stem),
        "protocol": {
            "evaluation_split": evaluation_split,
            "event_matching_unit": "video",
            "bootstrap_sampling_unit": "source_group",
            "tolerance_seconds": tolerance,
            "spatial_tolerance_normalized": spatial_tolerance,
            "bootstrap_iterations": iterations,
            "bootstrap_seed": seed,
        },
        "dataset": {
            "evaluation_split": evaluation_split,
            "labelled_videos": len(records),
            "labelled_test_videos": len(records) if evaluation_split == "test" else 0,
            "unique_source_groups": len({record["source_group_id"] for record in records}),
            "unique_test_source_groups": len({record["source_group_id"] for record in records}) if evaluation_split == "test" else 0,
            "duration_hours": round(duration_hours, 4),
            "ground_truth_events": sum(record["ground_truth_events"] for record in records),
            "predicted_events": sum(record["predicted_events"] for record in records),
        },
        "micro_average": micro,
        "macro_f1_across_rules": round(mean(item["f1"] for item in by_type.values()), 4) if by_type else 0.0,
        "false_positives_per_hour": round(micro["false_positives"] / duration_hours, 4) if duration_hours else None,
        "confidence_intervals_95": _bootstrap_intervals(
            [tuple(values) for values in source_group_counts.values()], iterations, seed
        ),
        "by_type": by_type,
        "per_video": records,
        "skipped_manifest_entries": skipped,
        "source_overlap_warnings": ([f"Source groups appear in both development/validation and test: {', '.join(overlap)}"] if overlap else []),
    }


def audit_study_manifest(manifest_path: Path) -> dict[str, Any]:
    """Check whether a study manifest is ready for a defensible held-out evaluation."""
    manifest_path = manifest_path.resolve()
    with manifest_path.open("r", encoding="utf-8") as stream:
        manifest = json.load(stream)
    videos = list(manifest.get("videos", []))
    video_ids = [str(item.get("video_id", "")) for item in videos]
    duplicate_ids = sorted({video_id for video_id in video_ids if video_id and video_ids.count(video_id) > 1})
    source_groups_by_split: dict[str, set[str]] = {}
    missing_files: list[dict[str, str]] = []
    incomplete: list[str] = []
    status_counts: dict[str, int] = {}
    split_counts: dict[str, int] = {}

    def resolve(value: str) -> Path:
        path = Path(value)
        return path if path.is_absolute() else (manifest_path.parent / path).resolve()

    for item in videos:
        video_id = str(item.get("video_id", "unknown"))
        split = str(item.get("split", "unspecified"))
        status = str(item.get("status", "unspecified"))
        source_group_id = str(
            item.get("source_group_id") or item.get("camera_id") or item.get("video_id", "unknown")
        )
        source_groups_by_split.setdefault(split, set()).add(source_group_id)
        split_counts[split] = split_counts.get(split, 0) + 1
        status_counts[status] = status_counts.get(status, 0) + 1
        if split == "test" and status != "labelled":
            incomplete.append(video_id)
        if split == "test" and status == "labelled":
            for field in ("events_path", "ground_truth_path"):
                value = item.get(field)
                if not value or not resolve(str(value)).is_file():
                    missing_files.append({"video_id": video_id, "field": field})

    development = source_groups_by_split.get("development", set()) | source_groups_by_split.get("validation", set())
    test_sources = source_groups_by_split.get("test", set())
    overlap = sorted(test_sources & development)
    blockers = []
    if duplicate_ids:
        blockers.append("Duplicate video IDs must be removed.")
    if overlap:
        blockers.append("Source-group overlap exists between development/validation and test splits.")
    if incomplete:
        blockers.append("Some test videos are not fully labelled.")
    if missing_files:
        blockers.append("Some labelled test files are missing.")
    return {
        "study_name": manifest.get("study_name", manifest_path.stem),
        "video_count": len(videos),
        "split_counts": dict(sorted(split_counts.items())),
        "status_counts": dict(sorted(status_counts.items())),
        "unique_source_groups_by_split": {key: len(value - {"unknown"}) for key, value in sorted(source_groups_by_split.items())},
        "duplicate_video_ids": duplicate_ids,
        "source_group_overlap": overlap,
        "incomplete_test_videos": incomplete,
        "missing_files": missing_files,
        "ready_for_final_accuracy_report": not blockers,
        "blockers": blockers,
    }


def compare_studies(ablation_manifest_path: Path) -> dict[str, Any]:
    """Evaluate equivalent study variants and report deltas from the declared baseline."""
    ablation_manifest_path = ablation_manifest_path.resolve()
    with ablation_manifest_path.open("r", encoding="utf-8") as stream:
        manifest = json.load(stream)
    variants = manifest.get("variants", [])
    if len(variants) < 2:
        raise ValueError("An ablation requires at least two variants")
    results = []
    for item in variants:
        study_path = Path(str(item["study_manifest"]))
        if not study_path.is_absolute():
            study_path = (ablation_manifest_path.parent / study_path).resolve()
        metrics = evaluate_study(study_path)
        results.append({
            "name": str(item["name"]),
            "detector": item.get("detector"),
            "tracker": item.get("tracker"),
            "micro_average": metrics["micro_average"],
            "macro_f1_across_rules": metrics["macro_f1_across_rules"],
            "false_positives_per_hour": metrics["false_positives_per_hour"],
            "confidence_intervals_95": metrics["confidence_intervals_95"],
            "by_type": metrics["by_type"],
            "dataset": metrics["dataset"],
        })
    baseline = results[0]
    for result in results:
        result["delta_vs_baseline"] = {
            metric: round(float(result["micro_average"][metric]) - float(baseline["micro_average"][metric]), 4)
            for metric in ("precision", "recall", "f1")
        }
    ranking = sorted(results, key=lambda item: (-float(item["micro_average"]["f1"]), -float(item["micro_average"]["recall"])))
    return {
        "experiment_name": manifest.get("experiment_name", ablation_manifest_path.stem),
        "baseline": baseline["name"],
        "selection_warning": "Choose on validation data only. Do not select the winner after inspecting the frozen test set.",
        "variants": results,
        "ranking_by_micro_f1_then_recall": [item["name"] for item in ranking],
    }
