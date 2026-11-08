from __future__ import annotations

import copy
import csv
import hashlib
import json
from pathlib import Path
from typing import Any

from .evaluation import evaluate_study
from .pipeline import run_pipeline
from .scope import validate_config_scope
from .storage import atomic_write_json


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SELECTION_RULE = (
    "Compare all declared variants on validation data only. Select the highest event-level micro F1; "
    "break ties by recall, then end-to-end real-time factor, then manifest order. Freeze that choice "
    "before running exactly one selected variant on the held-out test split."
)


def _source_group(case: dict[str, Any]) -> str:
    """Return the independent source grouping used to prevent split overlap."""
    return str(
        case.get("source_group_id")
        or case.get("camera_id")  # Backward compatibility with older manifests.
        or case.get("video_id", "unknown")
    )


def _resolve(base: Path, value: str) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (base / path).resolve()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _model_path(base: Path, value: str) -> Path:
    path = _resolve(base, value)
    if path.is_file():
        return path
    return (PROJECT_ROOT / value).resolve()


def validate_campaign(manifest_path: Path) -> dict[str, Any]:
    manifest_path = manifest_path.resolve()
    with manifest_path.open("r", encoding="utf-8") as stream:
        manifest = json.load(stream)
    cases = list(manifest.get("cases", []))
    variants = list(manifest.get("variants", []))
    blockers: list[str] = []
    warnings: list[str] = []
    base = manifest_path.parent
    if not cases:
        blockers.append("No video cases are registered.")
    if len(variants) < 2:
        blockers.append("At least two model/tracker variants are required for validation-stage selection.")
    case_ids = [str(case.get("video_id", "")) for case in cases]
    if not all(case_ids) or len(case_ids) != len(set(case_ids)):
        blockers.append("Video IDs must be present and unique.")
    variant_names = [str(variant.get("name", "")) for variant in variants]
    if not all(variant_names) or len(variant_names) != len(set(variant_names)):
        blockers.append("Variant names must be present and unique.")
    validation_cases = [case for case in cases if case.get("split") == "validation"]
    test_cases = [case for case in cases if case.get("split") == "test"]
    unsupported_splits = sorted(
        {str(case.get("split", "unspecified")) for case in cases} - {"validation", "test"}
    )
    if unsupported_splits:
        blockers.append(
            f"Campaign cases must use validation or test splits, not: {', '.join(unsupported_splits)}."
        )
    if not validation_cases:
        blockers.append("At least one labelled validation case is required for model selection.")
    if not test_cases:
        blockers.append("At least one labelled test case is required for final evaluation.")
    for case in cases:
        video_id = str(case.get("video_id", "unknown"))
        if case.get("status") != "labelled":
            blockers.append(f"{video_id}: case is not registered as labelled with frozen geometry (status: {case.get('status', 'missing')}).")
        for field in ("input_video", "config", "ground_truth"):
            value = case.get(field)
            if not value or not _resolve(base, str(value)).is_file():
                blockers.append(f"{video_id}: missing {field} file.")
        config_path = _resolve(base, str(case.get('config', '')))
        if config_path.is_file():
            try:
                validate_config_scope(json.loads(config_path.read_text(encoding='utf-8')))
            except (ValueError, TypeError, AttributeError, OSError) as exc:
                blockers.append(f'{video_id}: invalid rule scope: {exc}')
    for variant in variants:
        name = str(variant.get("name", "unnamed"))
        model = variant.get("model")
        tracker = str(variant.get("tracker", ""))
        if not model:
            blockers.append(f"{name}: model is missing.")
        elif not _model_path(base, str(model)).is_file():
            blockers.append(f"{name}: model file does not exist.")
        if tracker not in {"bytetrack", "botsort"}:
            blockers.append(f"{name}: tracker must be bytetrack or botsort.")
    validation_sources = {_source_group(case) for case in validation_cases}
    test_sources = {_source_group(case) for case in test_cases}
    overlap = sorted((validation_sources & test_sources) - {"unknown"})
    if overlap:
        blockers.append(f"Source-group overlap between validation and test: {', '.join(overlap)}.")
    hashes_by_split: dict[str, set[str]] = {'validation': set(), 'test': set()}
    for case in cases:
        path = _resolve(base, str(case.get('input_video', '')))
        split = str(case.get('split', ''))
        if split in hashes_by_split and path.is_file():
            hashes_by_split[split].add(_sha256_file(path))
    if hashes_by_split['validation'] & hashes_by_split['test']:
        blockers.append('Identical video content occurs in validation and test, despite declared source-group names.')
    return {
        "experiment_name": manifest.get("experiment_name", manifest_path.stem),
        "case_count": len(cases),
        "validation_case_count": len(validation_cases),
        "test_case_count": len(test_cases),
        "variant_count": len(variants),
        "planned_runs": len(validation_cases) * len(variants) + len(test_cases),
        "unique_validation_source_groups": len(validation_sources - {"unknown"}),
        "unique_test_source_groups": len(test_sources - {"unknown"}),
        "ready": not blockers,
        "blockers": blockers,
        "warnings": warnings,
        "manifest": manifest,
    }


def _case_hashes(base: Path, cases: list[dict[str, Any]]) -> dict[str, dict[str, str]]:
    records: dict[str, dict[str, str]] = {}
    for case in cases:
        video_id = str(case["video_id"])
        records[video_id] = {
            f"{field}_sha256": _sha256_file(_resolve(base, str(case[field])))
            for field in ("input_video", "config", "ground_truth")
        }
        review_path = _resolve(base, str(case['ground_truth'])).with_suffix('.review.json')
        if review_path.is_file():
            records[video_id]['review_sha256'] = _sha256_file(review_path)
    return records


def _campaign_inputs(manifest_path: Path, manifest: dict[str, Any]) -> dict[str, Any]:
    """Bind cached inference to the exact inputs, weights and implementation."""
    base = manifest_path.parent
    sources = sorted((PROJECT_ROOT/'src'/'traffic_edge').glob('*.py'))
    sources.append(PROJECT_ROOT/'requirements.txt')
    return {
        'schema_version': 1,
        'manifest_sha256': _sha256_file(manifest_path),
        'case_file_hashes': _case_hashes(base, manifest['cases']),
        'model_hashes': {str(v['name']): _sha256_file(_model_path(base, str(v['model'])))
                         for v in manifest['variants']},
        'implementation_hashes': {p.relative_to(PROJECT_ROOT).as_posix(): _sha256_file(p)
                                  for p in sources},
    }


def _freeze_campaign_inputs(output_dir: Path, inputs: dict[str, Any], resume: bool) -> None:
    ledger = output_dir/'campaign_inputs.json'
    if ledger.is_file():
        if not resume:
            raise ValueError('This campaign directory is already frozen; use --resume or a new directory.')
        if json.loads(ledger.read_text(encoding='utf-8')) != inputs:
            raise ValueError('Campaign inputs, model weights or implementation changed; refusing to reuse cached runs. Use a new campaign directory.')
    else:
        if any(output_dir.iterdir()):
            raise ValueError('Existing campaign files have no input ledger; refusing to reuse unverified runs. Use a new campaign directory.')
        atomic_write_json(ledger, inputs)


def _run_variant_cases(
    manifest: dict[str, Any],
    base: Path,
    stage_dir: Path,
    variant: dict[str, Any],
    cases: list[dict[str, Any]],
    evaluation_split: str,
    seed_offset: int,
    resume: bool,
) -> dict[str, Any]:
    variant_name = str(variant["name"])
    variant_dir = stage_dir / variant_name
    variant_dir.mkdir(parents=True, exist_ok=True)
    model_path = _model_path(base, str(variant["model"]))
    study_videos = []
    run_summaries = []
    run_directories = []
    for case in cases:
        video_id = str(case["video_id"])
        case_dir = variant_dir / video_id
        summary_path = case_dir / "summary.json"
        if summary_path.exists() and resume:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
        else:
            if summary_path.exists():
                raise ValueError(
                    f"Existing run found at {case_dir}; use --resume or choose a new output directory"
                )
            config_path = _resolve(base, str(case["config"]))
            config = json.loads(config_path.read_text(encoding="utf-8"))
            frozen = copy.deepcopy(config)
            frozen.setdefault("yolo", {})["model"] = str(model_path)
            frozen["yolo"]["tracker"] = str(variant["tracker"])
            variant_config = variant_dir / "configs" / f"{video_id}.json"
            variant_config.parent.mkdir(exist_ok=True)
            variant_config.write_text(json.dumps(frozen, indent=2), encoding="utf-8")
            summary = run_pipeline(
                _resolve(base, str(case["input_video"])),
                case_dir,
                variant_config,
                "yolo",
            )
        run_summaries.append(summary)
        run_directories.append(str(case_dir.resolve()))
        duration = float(
            case.get("duration_seconds")
            or (summary["frames_processed"] / summary["source_fps"])
        )
        study_videos.append({
            "video_id": video_id,
            "source_group_id": _source_group(case),
            "split": evaluation_split,
            "status": "labelled",
            "duration_seconds": duration,
            "events_path": summary["outputs"]["events"],
            "ground_truth_path": str(_resolve(base, str(case["ground_truth"]))),
        })
    study_manifest = {
        "study_name": f"{manifest.get('experiment_name', 'campaign')}-{evaluation_split}-{variant_name}",
        "tolerance_seconds": float(manifest.get("tolerance_seconds", 0.5)),
        "spatial_tolerance_normalized": float(manifest.get("spatial_tolerance_normalized", 0.12)),
        "bootstrap_iterations": int(manifest.get("bootstrap_iterations", 2000)),
        "bootstrap_seed": int(manifest.get("bootstrap_seed", 20260923)) + seed_offset,
        "videos": study_videos,
    }
    study_path = variant_dir / "study_manifest.json"
    study_path.write_text(json.dumps(study_manifest, indent=2), encoding="utf-8")
    metrics = evaluate_study(study_path, evaluation_split=evaluation_split)
    (variant_dir / "evaluation.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )
    frames = sum(int(summary["frames_processed"]) for summary in run_summaries)
    seconds = sum(float(summary["processing_seconds"]) for summary in run_summaries)
    source_seconds = sum(
        int(summary["frames_processed"]) / float(summary["source_fps"])
        for summary in run_summaries
    )
    return {
        "name": variant_name,
        "model": model_path.name,
        "model_sha256": _sha256_file(model_path),
        "tracker": str(variant["tracker"]),
        "processing_fps": round(frames / seconds, 3) if seconds else 0.0,
        "real_time_factor": round(source_seconds / seconds, 4) if seconds else 0.0,
        "metrics": metrics,
        "run_directories": run_directories,
    }


def _summary_row(result: dict[str, Any]) -> dict[str, Any]:
    score = result["metrics"]["micro_average"]
    return {
        "variant": result["name"],
        "model": result["model"],
        "tracker": result["tracker"],
        "precision": score["precision"],
        "recall": score["recall"],
        "f1": score["f1"],
        "false_positives_per_hour": result["metrics"]["false_positives_per_hour"],
        "processing_fps": result["processing_fps"],
        "real_time_factor": result["real_time_factor"],
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _write_markdown(report: dict[str, Any], path: Path) -> None:
    lines = [
        f"# {report['experiment_name']} - Accuracy Analysis",
        "",
        "Candidate events are research outputs, not legal decisions.",
        "",
        "## Validation-only model selection",
        "",
        "| Variant | Detector | Tracker | Precision | Recall | F1 | FP/hour | Processing FPS | Real-time factor |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for variant in report["validation"]["variants"]:
        score = variant["metrics"]["micro_average"]
        lines.append(
            f"| {variant['name']} | {variant['model']} | {variant['tracker']} | "
            f"{score['precision']:.4f} | {score['recall']:.4f} | {score['f1']:.4f} | "
            f"{variant['metrics']['false_positives_per_hour']} | {variant['processing_fps']:.3f} | "
            f"{variant['real_time_factor']:.3f} |"
        )
    final = report["test"]
    final_score = final["metrics"]["micro_average"]
    lines.extend([
        "",
        "## Frozen selection",
        "",
        f"- Selected variant: **{report['selected_variant']['name']}**",
        f"- Selection file: `selected_variant.json` (SHA-256 `{report['selection_file_sha256']}`)",
        f"- Rule: {report['protocol']['selection_rule']}",
        "",
        "## Final held-out test - selected variant only",
        "",
        "No multi-variant test leaderboard is produced.",
        "",
        "| Variant | Precision | Recall | F1 | FP/hour | Processing FPS | Real-time factor |",
        "|---|---:|---:|---:|---:|---:|---:|",
        f"| {final['name']} | {final_score['precision']:.4f} | {final_score['recall']:.4f} | "
        f"{final_score['f1']:.4f} | {final['metrics']['false_positives_per_hour']} | "
        f"{final['processing_fps']:.3f} | {final['real_time_factor']:.3f} |",
        "",
        "## Dataset",
        "",
        f"- Labelled validation cases: {report['dataset']['validation_cases']}",
        f"- Labelled test cases: {report['dataset']['test_cases']}",
        f"- Independent test source groups: {report['dataset']['test_source_groups']}",
        f"- Completed inference runs: {report['dataset']['completed_runs']}",
        "",
        "Per-rule metrics, confidence intervals, per-video errors and exact run paths are retained in `accuracy_analysis.json`.",
    ])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_campaign(manifest_path: Path, output_dir: Path, resume: bool = False) -> dict[str, Any]:
    readiness = validate_campaign(manifest_path)
    if not readiness["ready"]:
        raise ValueError("Campaign is not ready:\n- " + "\n- ".join(readiness["blockers"]))
    manifest = readiness["manifest"]
    manifest_path = manifest_path.resolve()
    base = manifest_path.parent
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    frozen_inputs = _campaign_inputs(manifest_path, manifest)
    _freeze_campaign_inputs(output_dir, frozen_inputs, resume)
    validation_cases = [case for case in manifest["cases"] if case.get("split") == "validation"]
    test_cases = [case for case in manifest["cases"] if case.get("split") == "test"]
    manifest_hash = _sha256_file(manifest_path)
    frozen_case_hashes = _case_hashes(base, list(manifest["cases"]))

    validation_results = []
    validation_root = output_dir / "validation"
    for variant_index, variant in enumerate(manifest["variants"]):
        validation_results.append(_run_variant_cases(
            manifest,
            base,
            validation_root,
            variant,
            validation_cases,
            "validation",
            variant_index,
            resume,
        ))
    baseline_score = validation_results[0]["metrics"]["micro_average"]
    for result in validation_results:
        result["delta_vs_baseline"] = {
            metric: round(
                float(result["metrics"]["micro_average"][metric])
                - float(baseline_score[metric]),
                4,
            )
            for metric in ("precision", "recall", "f1")
        }
    ranked = sorted(
        enumerate(validation_results),
        key=lambda indexed: (
            -float(indexed[1]["metrics"]["micro_average"]["f1"]),
            -float(indexed[1]["metrics"]["micro_average"]["recall"]),
            -float(indexed[1]["real_time_factor"]),
            indexed[0],
        ),
    )
    selected_index, selected_validation = ranked[0]
    selected_manifest_variant = manifest["variants"][selected_index]

    if (
        _campaign_inputs(manifest_path, manifest) != frozen_inputs
    ):
        raise ValueError(
            "Campaign inputs changed during validation; selection was not frozen and test was not run"
        )
    selection = {
        "schema_version": 1,
        "experiment_name": manifest.get("experiment_name", manifest_path.stem),
        "campaign_manifest_sha256": manifest_hash,
        "selection_rule": SELECTION_RULE,
        "selected_variant": {
            "name": selected_validation["name"],
            "model": str(selected_manifest_variant["model"]),
            "model_sha256": selected_validation["model_sha256"],
            "tracker": selected_validation["tracker"],
        },
        "case_file_hashes": frozen_case_hashes,
        "validation_result": {
            "micro_average": selected_validation["metrics"]["micro_average"],
            "processing_fps": selected_validation["processing_fps"],
            "real_time_factor": selected_validation["real_time_factor"],
        },
    }
    selection_path = output_dir / "selected_variant.json"
    if selection_path.exists():
        if not resume:
            raise ValueError(
                "A frozen selection already exists; use --resume or choose a new output directory"
            )
        existing_selection = json.loads(selection_path.read_text(encoding="utf-8"))
        if existing_selection != selection:
            raise ValueError(
                "Frozen selection or campaign input hashes changed; refusing to run the test split"
            )
    else:
        selection_path.write_text(json.dumps(selection, indent=2) + "\n", encoding="utf-8")
    selection_file_hash = _sha256_file(selection_path)

    final_test = _run_variant_cases(
        manifest,
        base,
        output_dir / "test",
        selected_manifest_variant,
        test_cases,
        "test",
        selected_index,
        resume,
    )
    if _campaign_inputs(manifest_path, manifest) != frozen_inputs:
        raise ValueError('Campaign inputs changed during held-out inference; refusing to publish accuracy results.')
    report = {
        "experiment_name": manifest.get("experiment_name", manifest_path.stem),
        "protocol": {
            "model_selection_split": "validation",
            "final_evaluation_split": "test",
            "selection_rule": SELECTION_RULE,
            "test_policy": "Run and report the frozen selected variant only; no multi-variant test ranking.",
        },
        "campaign_manifest_sha256": manifest_hash,
        "selection_file_sha256": selection_file_hash,
        "dataset": {
            "validation_cases": len(validation_cases),
            "test_cases": len(test_cases),
            "validation_source_groups": len({_source_group(case) for case in validation_cases}),
            "test_source_groups": len({_source_group(case) for case in test_cases}),
            "completed_runs": len(validation_cases) * len(manifest["variants"]) + len(test_cases),
        },
        "validation": {
            "baseline": str(manifest["variants"][0]["name"]),
            "ranking_by_f1_recall_realtime": [item[1]["name"] for item in ranked],
            "variants": validation_results,
        },
        "selected_variant": selection["selected_variant"],
        "test": final_test,
    }
    (output_dir / "accuracy_analysis.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    _write_markdown(report, output_dir / "ACCURACY_ANALYSIS.md")
    _write_csv(
        output_dir / "validation_comparison.csv",
        [_summary_row(item) for item in validation_results],
    )
    _write_csv(output_dir / "final_test_metrics.csv", [_summary_row(final_test)])
    return report
