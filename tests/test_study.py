import json
from pathlib import Path
import tempfile
import threading
import time
import unittest
from unittest.mock import patch
from http.server import ThreadingHTTPServer
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import cv2
import numpy as np

from traffic_edge import study, web_ui, diagnostics
from traffic_edge.pipeline import run_pipeline
from traffic_edge.demo import generate_builtin_examples
from traffic_edge.storage import atomic_write_json


class StudyTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        for folder in ("data", "data/annotations", "data/annotation_drafts", "config"):
            (self.root / folder).mkdir(parents=True, exist_ok=True)
        for module, key, value in (
            (study, "ROOT", self.root), (study, "REGISTRY", self.root / "data/study_registry.json"),
            (web_ui, "PROJECT_ROOT", self.root), (web_ui, "DATA_DIR", self.root / "data"),
            (web_ui, "ANNOTATION_DIR", self.root / "data/annotations"),
            (web_ui, "ANNOTATION_DRAFT_DIR", self.root / "data/annotation_drafts"),
        ):
            patcher = patch.object(module, key, value)
            patcher.start()
            self.addCleanup(patcher.stop)
        self.model = self.root / "detector.pt"
        self.model.write_bytes(b"test-model")
        self.config = web_ui.create_config(
            "camera-a", [[0, .35], [1, .35], [1, .7], [0, .7]], .25,
            direction=[[.2, .55], [.8, .55]], enabled_rules={"wrong_way": True}, device="cpu",
        )
        self.counter = 0

    def label(self, time=2.5):
        return {"timestamp_seconds": time, "candidate_event": "candidate_wrong_way_driving",
                "lane_id": "lane-1", "vehicle_point_normalized": [.5, .5],
                "vehicle_bbox_normalized": [.4, .3, .6, .5], "annotation_status": "confirmed"}

    def review(self, name, labels, video=None, duration=10):
        if video is None:
            video = self.root / "data" / f"{name}.mp4"
            self.counter += 1
            writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"mp4v"), 10, (160, 96))
            self.assertTrue(writer.isOpened())
            for _ in range(int(duration * 10)):
                writer.write(np.full((96, 160, 3), self.counter * 25, dtype=np.uint8))
            writer.release()
        truth = self.root / "data/annotations" / f"{name}.json"
        atomic_write_json(truth, labels)
        atomic_write_json(truth.with_suffix(".review.json"), {
            "reviewer": "test-reviewer", "review_complete": True, "review_coverage_percent": 98,
            "duration_seconds": duration, "confirmed_events": len(labels), "negative_video": not labels,
            "source_sha256": study.file_hash(video),
        })
        return video, truth

    def register(self, name, labels, split="validation", group=None):
        video, truth = self.review(name, labels)
        return study.register_case(video, truth, self.config, name=name, source_group=group or name,
                                   camera_id=name, split=split, expected_revision=study.read_registry()["revision"])

    def test_register_negative_and_reject_incomplete_or_changed_inputs(self):
        case = self.register("negative", [])
        self.assertTrue(case["negative_video"])
        self.assertEqual(study.case_issues(case), [])
        truth = study.data_path(case["ground_truth"])
        record = json.loads(truth.with_suffix(".review.json").read_text())
        record["review_coverage_percent"] = 50
        atomic_write_json(truth.with_suffix(".review.json"), record)
        self.assertTrue(study.case_issues(case))
        with self.assertRaisesRegex(ValueError, "98%"):
            study.register_case(study.data_path(case["input_video"]), truth, self.config,
                                name="negative", source_group="negative", camera_id="negative",
                                split="validation", expected_revision=1)

    def test_source_groups_and_camera_ids_cannot_cross_into_test(self):
        self.register("validation-a", [self.label()], group="same-source")
        video, truth = self.review("test-b", [self.label()])
        for group, camera in (("same-source", "camera-b"), ("new-source", "validation-a")):
            with self.subTest(group=group), self.assertRaisesRegex(ValueError, "overlap"):
                study.register_case(video, truth, self.config, name="test-b", source_group=group,
                                    camera_id=camera, split="test", expected_revision=1)
        self.assertEqual(study.read_registry()["revision"], 1)

    def test_stale_writes_and_promoting_development_footage_are_rejected(self):
        case = self.register("development", [self.label()], split="development")
        with self.assertRaisesRegex(ValueError, "changed"):
            study.remove_case(case["id"], 0)
        with self.assertRaisesRegex(ValueError, "cannot be promoted"):
            study.register_case(study.data_path(case["input_video"]), study.data_path(case["ground_truth"]),
                                self.config, name="renamed", source_group="new", camera_id="new",
                                split="test", expected_revision=1)
        study.remove_case(case["id"], 1)
        self.assertEqual(study.read_registry()["cases"], [])
        self.assertTrue(study.data_path(case["ground_truth"]).is_file())

    def test_labels_must_belong_to_an_enabled_lane_and_rule(self):
        label = self.label()
        label["lane_id"] = "missing"
        with self.assertRaisesRegex(ValueError, "labelled lane"):
            self.register("bad-lane", [label])
        label["lane_id"] = "lane-1"
        label["candidate_event"] = "candidate_red_signal_violation"
        with self.assertRaisesRegex(ValueError, "labelled lane"):
            self.register("disabled-rule", [label])

    def test_batches_keep_splits_separate_and_export_real_matching_errors(self):
        self.register("positive", [self.label(2.5), self.label(7)])
        self.register("negative", [])
        self.register("development", [], split="development")
        cases = study.snapshot_cases("validation")
        self.assertEqual(len(cases), 2)

        def pipeline(video, output, config, backend, **kwargs):
            output.mkdir()
            event = self.label(2.5 if video.stem == "positive" else 4)
            event.update(event_id=video.stem, ground_point_normalized=event["vehicle_point_normalized"])
            path = output / "events.jsonl"
            path.write_text(json.dumps(event) + "\n", encoding="utf-8")
            return {"outputs": {"events": str(path)}}

        output = self.root / "runs/study-20261108-120000-abcdef012345"
        with patch.object(study, "run_pipeline", side_effect=pipeline):
            result = study.run_study(cases, output, model=self.model, tracker="bytetrack", device="cpu")
        score = result["metrics"]["micro_average"]
        self.assertEqual([score[k] for k in ("true_positives", "false_positives", "false_negatives")], [1, 1, 1])
        self.assertEqual([score[k] for k in ("precision", "recall", "f1")], [.5, .5, .5])
        self.assertEqual(result["normal_video_count"], 1)
        self.assertEqual({e["kind"] for e in result["errors"]}, {"FP", "FN"})
        self.assertTrue(all((output / e["frame"]).is_file() for e in result["errors"]))
        self.assertTrue((output / "cases.csv").is_file())
        self.assertTrue((output / "errors.csv").is_file())
        self.assertEqual(study.report_directory(output.name), output)
        with self.assertRaises(ValueError):
            study.report_directory("../outside")

    def test_inputs_changed_during_processing_cannot_publish_a_report(self):
        self.register("case", [])
        cases = study.snapshot_cases("validation")
        def pipeline(video, output, config, backend, **kwargs):
            output.mkdir()
            path = output / "events.jsonl"
            path.write_text("", encoding="utf-8")
            self.model.write_bytes(b"changed-model")
            return {"outputs": {"events": str(path)}}
        output = self.root / "runs/changed"
        with patch.object(study, "run_pipeline", side_effect=pipeline), self.assertRaisesRegex(ValueError, "changed"):
            study.run_study(cases, output, model=self.model, tracker="bytetrack", device="cpu")
        self.assertFalse((output / "report.json").exists())

    def test_synthetic_video_runs_through_the_normal_pipeline(self):
        videos = self.root / "data/synthetic"
        generate_builtin_examples(videos)
        label = self.label(.4)
        label["vehicle_point_normalized"] = [.84, .55]
        video, truth = self.review("synthetic", [label], video=videos / "wrong_way.webm", duration=5)
        study.register_case(video, truth, self.config, name="synthetic", source_group="synthetic",
                            camera_id="synthetic", split="validation", expected_revision=0)
        result = study.run_study(study.snapshot_cases("validation"), self.root / "runs/synthetic",
                                 model=self.model, tracker="bytetrack", device="cpu", backend="demo")
        self.assertEqual(result["metrics"]["dataset"]["predicted_events"], 1)
        self.assertEqual(result["metrics"]["micro_average"]["true_positives"], 1)
        case = study.read_registry()["cases"][0]
        folder = self.root / "runs/synthetic" / case["id"]
        summary = json.loads((folder / "summary.json").read_text())
        trace = folder / "decision_trace.jsonl"
        self.assertEqual(summary["decision_trace_sha256"], study.file_hash(trace))
        rows = [json.loads(line) for line in trace.read_text().splitlines()]
        self.assertEqual(len(rows)-1, summary["frames_processed"])
        self.assertTrue(any(t["rules"] for r in rows[1:] for t in r["tracks"]))
        config = json.loads((folder / "config_used.json").read_text())
        config.pop("diagnostics")
        config_path = self.root / "without-trace.json"
        atomic_write_json(config_path, config)
        normal = run_pipeline(video, self.root / "without-trace", config_path, "demo")
        def events(summary):
            return [(e["candidate_event"], e["track_id"], e["timestamp_seconds"])
                    for e in map(json.loads, Path(summary["outputs"]["events"]).read_text().splitlines())]
        self.assertEqual(events(summary), events(normal))
        self.assertNotIn("decision_trace_sha256", normal)

    def test_diagnostic_replay_preserves_metrics_and_verifies_inputs(self):
        case = self.register("miss", [self.label()])
        run_name = "study-20261108-120000-abcdef123456"
        directory = self.root / "runs" / run_name
        folder = directory / case["id"]
        folder.mkdir(parents=True)
        config_path = folder / "config_used.json"
        atomic_write_json(config_path, self.config)
        atomic_write_json(folder / "summary.json", {"config_sha256": study.file_hash(config_path)})
        error = {"event_id": "error-00000", "case_id": case["id"], "kind": "FN",
                 "timestamp_seconds": 2.5, "candidate_event": "candidate_wrong_way_driving",
                 "lane_id": "lane-1", "bbox_normalized": [.4, .3, .6, .5]}
        atomic_write_json(directory / "report.json", {"errors": [error], "metrics": {"false_negatives": 1}})
        atomic_write_json(directory / "study_inputs.json", {"cases": [case], "model": "detector.pt",
                                                            "model_sha256": study.file_hash(self.model)})
        original_report = (directory / "report.json").read_bytes()
        self.assertFalse(diagnostics.diagnose_error(run_name, "error-00000")["available"])
        def replay(video, output, config, backend, **kwargs):
            trace = output / "decision_trace.jsonl"
            trace.write_text(json.dumps({"kind": "header", "schema_version": 1,
                "source_sha256": case["source_sha256"], "config_sha256": study.file_hash(config),
                "configuration": json.loads(config.read_text())})+'\n'+json.dumps({"kind": "frame",
                "frame_index": 26, "time": 2.5, "detections": [], "tracks": [], "signals": {},
                "raw_detection_status": "before_association"})+'\n')
            atomic_write_json(output / "summary.json", {"config_sha256": study.file_hash(config),
                                                        "decision_trace_sha256": study.file_hash(trace)})
        output = directory / "diagnostic-replays" / (case["id"]+"-abc")
        state = web_ui.AppState()
        server = ThreadingHTTPServer(("127.0.0.1", 0), web_ui.make_handler(state))
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        self.addCleanup(thread.join)
        self.addCleanup(server.server_close)
        self.addCleanup(server.shutdown)
        base = f"http://127.0.0.1:{server.server_port}"
        url = base + f"/api/study/diagnosis?run={run_name}&error=error-00000"
        with patch("traffic_edge.pipeline.run_pipeline", side_effect=replay) as pipeline:
            with urlopen(url) as response:
                self.assertFalse(json.load(response)["available"])
            state.processing_lock.acquire()
            try:
                request = Request(base+"/api/study/diagnosis-replay", data=json.dumps({
                    "run": run_name, "error": "error-00000", "device": "cpu"}).encode(),
                    headers={"Content-Type": "application/json"})
                with urlopen(request) as response:
                    job_id = json.load(response)["job_id"]
                self.assertEqual(state.get_job(job_id)["status"], "queued")
                pipeline.assert_not_called()
            finally:
                state.processing_lock.release()
            deadline = time.monotonic()+5
            while state.get_job(job_id)["status"] not in {"complete", "error"} and time.monotonic()<deadline:
                time.sleep(.02)
            self.assertEqual(state.get_job(job_id)["status"], "complete", state.get_job(job_id))
            with urlopen(url) as response:
                result = json.load(response)
            self.assertEqual(result["mode"], "replay")
            self.assertEqual(result["stage"], "detection")
            self.assertTrue(result["anchor_image"].startswith("data:image/jpeg;base64,"))
            with self.assertRaises(HTTPError):
                urlopen(url+"&track=not-an-id")
            diagnostics.replay_error(run_name, "error-00000", output, "cpu")
            self.assertEqual(pipeline.call_count, 1, "Completed trace must be reused")
        self.assertEqual((directory / "report.json").read_bytes(), original_report)
        output = directory / "diagnostic-replays" / (case["id"]+"-"+job_id)
        (output / "decision_trace.jsonl").write_text("tampered")
        with self.assertRaisesRegex(ValueError, "checksum"):
            diagnostics.diagnose_error(run_name, "error-00000")

    def test_replay_with_changed_model_cannot_publish_diagnostics(self):
        case = self.register("miss", [self.label()])
        run_name = "study-20261108-120000-abcdef123456"
        directory = self.root / "runs" / run_name
        directory.mkdir(parents=True)
        atomic_write_json(directory / "report.json", {"errors": [{"event_id": "error-00000", "case_id": case["id"]}]})
        atomic_write_json(directory / "study_inputs.json", {"cases": [case], "model": "detector.pt", "model_sha256": "changed"})
        with self.assertRaisesRegex(ValueError, "weights are missing or changed"):
            diagnostics.replay_error(run_name, "error-00000", directory / "replay", "cpu")
        self.assertFalse((directory / "replay").exists())

    def test_web_api_registers_without_exposing_predictions_or_other_files(self):
        video, truth = self.review("web", [])
        digest, identity = web_ui._annotation_identity(video)
        atomic_write_json(web_ui.ANNOTATION_DRAFT_DIR / f"{identity}.draft.json", {
            "review_complete": True, "source_sha256": digest,
            "final_ground_truth": truth.relative_to(self.root).as_posix(),
        })
        state = web_ui.AppState()
        server = ThreadingHTTPServer(("127.0.0.1", 0), web_ui.make_handler(state))
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        base = f"http://127.0.0.1:{server.server_port}"
        self.addCleanup(thread.join)
        self.addCleanup(server.server_close)
        self.addCleanup(server.shutdown)
        profile = {"camera_id": "web-camera", "source_signature": {"width": 160, "height": 96},
                   "lanes": [{"id": "lane-1", "name": "Lane 1", "polygon": [[0, .35], [1, .35], [1, .7], [0, .7]],
                              "direction": [[.2, .55], [.8, .55]], "enabled_rules": {"wrong_way": True}}]}
        payload = {"video_path": video.relative_to(self.root).as_posix(), "profile": profile,
                   "name": "web", "camera_id": "web-camera", "source_group_id": "web-source",
                   "split": "validation", "revision": 0}
        request = Request(base + "/api/study/cases", data=json.dumps(payload).encode(),
                          headers={"Content-Type": "application/json"})
        with urlopen(request) as response:
            self.assertIn("case_id", json.load(response))
        with urlopen(base + "/api/study/cases") as response:
            data = json.load(response)
        self.assertEqual(data["revision"], 1)
        self.assertTrue(data["cases"][0]["negative_video"])
        self.assertEqual(data["cases"][0]["issues"], [])
        with self.assertRaises(HTTPError):
            urlopen(base + "/api/study/asset?run=../../&name=requirements.txt")

        def pipeline(video, output, config, backend, **kwargs):
            output.mkdir()
            events = output / "events.jsonl"
            events.write_text("", encoding="utf-8")
            return {"outputs": {"events": str(events)}}

        with patch.object(web_ui, "MODEL_PATHS", {"yolov8n": self.model}), \
                patch.object(study, "run_pipeline", side_effect=pipeline) as inference:
            state.processing_lock.acquire()
            try:
                payload = {"split": "validation", "model": "yolov8n", "tracker": "bytetrack", "device": "cpu"}
                request = Request(base + "/api/study/run", data=json.dumps(payload).encode(),
                                  headers={"Content-Type": "application/json"})
                with urlopen(request) as response:
                    job_id = json.load(response)["job_id"]
                self.assertEqual(state.get_job(job_id)["status"], "queued")
                inference.assert_not_called()
            finally:
                state.processing_lock.release()
            deadline = time.monotonic() + 5
            while state.get_job(job_id)["status"] not in {"complete", "error"} and time.monotonic() < deadline:
                time.sleep(.02)
            job = state.get_job(job_id)
            self.assertEqual(job["status"], "complete", job.get("error"))
        state.jobs.clear()  # Completed reports must survive losing in-memory jobs.
        run = job["run_name"]
        with urlopen(base + "/api/study/report?run=" + run) as response:
            report = json.load(response)
        self.assertEqual(report["normal_video_count"], 1)
        self.assertEqual(report["metrics"]["micro_average"]["false_positives"], 0)
        with urlopen(base + "/api/study/asset?run=" + run + "&name=cases.csv") as response:
            self.assertIn(b"true_positives", response.read())
        source = base + "/api/study/source?run=" + run + "&case=" + report["cases"][0]["case_id"]
        with urlopen(Request(source, headers={"Range": "bytes=0-20"})) as response:
            self.assertEqual(response.status, 206)
            self.assertEqual(len(response.read()), 21)


if __name__ == "__main__":
    unittest.main()
