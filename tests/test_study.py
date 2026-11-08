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

from traffic_edge import study, web_ui
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
