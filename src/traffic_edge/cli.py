from __future__ import annotations

import argparse
import json
from pathlib import Path

from .demo import generate_demo_video
from .campaign import run_campaign, validate_campaign
from .evaluation import audit_study_manifest, compare_studies, evaluate_files, evaluate_study
from .pipeline import run_pipeline


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Candidate traffic violation Edge-AI prototype")
    subparsers = parser.add_subparsers(dest="command", required=True)

    demo = subparsers.add_parser("generate-demo", help="Create a deterministic synthetic test video")
    demo.add_argument("--output", type=Path, required=True)

    run = subparsers.add_parser("run", help="Run detection, tracking, rule inference and event logging")
    run.add_argument("--input", type=Path, required=True)
    run.add_argument("--output-dir", type=Path, required=True)
    run.add_argument("--config", type=Path, required=True)
    run.add_argument("--backend", choices=("demo", "yolo"), default="yolo")
    run.add_argument("--device", choices=("auto", "cpu", "cuda"), help="Override the inference device in the configuration")

    evaluate = subparsers.add_parser("evaluate", help="Evaluate event time and vehicle location against manual ground truth")
    evaluate.add_argument("--events", type=Path, required=True)
    evaluate.add_argument("--ground-truth", type=Path, required=True)
    evaluate.add_argument("--tolerance", type=float, default=0.5)
    evaluate.add_argument("--spatial-tolerance", type=float, default=0.12)
    evaluate.add_argument("--output", type=Path)

    study = subparsers.add_parser("study", help="Aggregate a held-out multi-video accuracy study")
    study.add_argument("--manifest", type=Path, required=True)
    study.add_argument("--output", type=Path, required=True)

    audit = subparsers.add_parser("audit-study", help="Check dataset completeness and source overlap before evaluation")
    audit.add_argument("--manifest", type=Path, required=True)
    audit.add_argument("--output", type=Path)

    ablation = subparsers.add_parser("compare-studies", help="Compare equivalent detector/tracker study variants")
    ablation.add_argument("--manifest", type=Path, required=True)
    ablation.add_argument("--output", type=Path, required=True)

    campaign = subparsers.add_parser("campaign", help="Validate or run a frozen multi-video model/tracker experiment")
    campaign.add_argument("--manifest", type=Path, required=True)
    campaign.add_argument("--output-dir", type=Path, required=True)
    campaign.add_argument("--dry-run", action="store_true")
    campaign.add_argument("--resume", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "generate-demo":
        generate_demo_video(args.output)
        print(json.dumps({"created": str(args.output.resolve())}, indent=2))
    elif args.command == "run":
        summary = run_pipeline(args.input, args.output_dir, args.config, args.backend, device=args.device)
        print(json.dumps(summary, indent=2))
    elif args.command == "evaluate":
        metrics = evaluate_files(args.events, args.ground_truth, args.tolerance, args.spatial_tolerance)
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        print(json.dumps(metrics, indent=2))
    elif args.command == "study":
        metrics = evaluate_study(args.manifest)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        print(json.dumps(metrics, indent=2))
    elif args.command == "audit-study":
        report = audit_study_manifest(args.manifest)
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(json.dumps(report, indent=2))
    elif args.command == "compare-studies":
        report = compare_studies(args.manifest)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(json.dumps(report, indent=2))
    elif args.command == "campaign":
        report = validate_campaign(args.manifest) if args.dry_run else run_campaign(args.manifest, args.output_dir, args.resume)
        args.output_dir.mkdir(parents=True, exist_ok=True)
        if args.dry_run:
            (args.output_dir / "campaign_readiness.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
