import copy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np

from traffic_edge.diagnostics import inspect_trace
from traffic_edge.backends import UltralyticsByteTrackBackend


class DiagnosticTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / "trace.jsonl"
        self.error = {"timestamp_seconds": 1, "candidate_event": "candidate_red_signal_violation",
                      "lane_id": "lane-1", "bbox_normalized": [.4, .3, .6, .5]}
        self.track = {"track_id": 7, "class_name": "car", "bbox": [.4, .3, .6, .5],
                      "provisional": False, "lane_id": "lane-1", "rules": []}

    def row(self, tracks=None, detections=None, time=1, status="before_association"):
        return {"kind": "frame", "frame_index": round(time*10)+1, "time": time,
                "tracks": tracks or [], "detections": detections, "raw_detection_status": status,
                "signals": {"lane-1": {"state": "unknown", "is_red": False}}}

    def inspect(self, rows, selected=None):
        header = {"kind": "header", "schema_version": 1, "configuration": {"yolo": {"confidence": .25}}}
        self.path.write_text("\n".join(json.dumps(r) for r in [header, *rows]), encoding="utf-8")
        return inspect_trace(self.path, self.error, selected)

    def decision(self, **checks):
        self.track["rules"] = [{"rule": "red_light", "lane_id": "lane-1", "checks": checks,
                                "values": {"minimum_age": 3}, "emitted": False}]

    def test_no_surviving_box_and_unassociated_box_are_distinguished(self):
        self.assertEqual(self.inspect([self.row(detections=[])])["stage"], "detection")
        self.assertEqual(self.inspect([self.row(detections=[{"bbox": self.track["bbox"]}])])["stage"], "tracking")

    def test_missing_raw_output_does_not_establish_detector_failure(self):
        result = self.inspect([self.row(status="not_recorded")])
        self.assertEqual(result["stage"], "insufficient_recording")

    def test_ambiguous_vehicle_identity_requires_reviewer_choice(self):
        other = copy.deepcopy(self.track)
        other["track_id"] = 8
        result = self.inspect([self.row([self.track, other])])
        self.assertEqual(result["stage"], "association")
        self.assertIsNone(result["selected_track"])
        self.assertEqual(len(result["candidates"]), 2)
        self.assertEqual(self.inspect([self.row([self.track, other])], 7)["stage"], "lane_assignment")
        with self.assertRaisesRegex(ValueError, "does not appear"):
            self.inspect([self.row([self.track])], 99)

    def test_signal_gates_are_reported_at_the_recorded_crossing(self):
        self.decision(crossed_finite_stop_line=True, signal_red=False, previous_signal_red=False,
                      lane_output_eligible=True, track_mature=True)
        result = self.inspect([self.row([self.track])])
        self.assertEqual(result["stage"], "signal")
        self.assertIn("signal_red", result["blocking_checks"])
        self.assertEqual(result["timeline"][0]["decision"]["values"]["minimum_age"], 3)

    def test_lane_exit_rejection_and_insufficient_history_are_separate(self):
        self.decision(crossed_finite_stop_line=True, lane_output_eligible=False)
        self.assertEqual(self.inspect([self.row([self.track])])["stage"], "lane_assignment")
        self.decision(previous_observation=False, lane_output_eligible=True)
        self.assertEqual(self.inspect([self.row([self.track])])["stage"], "track_history")

    def test_no_crossing_reports_geometry_and_provisional_ids_report_tracking(self):
        self.decision(crossed_finite_stop_line=False, lane_output_eligible=True)
        self.assertEqual(self.inspect([self.row([self.track])])["stage"], "crossing_geometry")
        self.track["provisional"] = True
        self.assertEqual(self.inspect([self.row([self.track])])["stage"], "tracking")

    def test_gaps_are_not_treated_as_proven_identity_switches(self):
        result = self.inspect([self.row([self.track], time=.9), self.row(time=1)], 7)
        self.assertEqual(result["stage"], "tracking_gap")
        self.assertIn("does not establish", result["finding"])

    def test_emitted_event_points_back_to_evaluation_matching(self):
        self.decision(crossed_finite_stop_line=True, lane_output_eligible=True)
        self.track["rules"][0]["emitted"] = True
        self.assertEqual(self.inspect([self.row([self.track])])["stage"], "event_matching")

    def test_detector_callback_copies_boxes_before_tracker_mutation(self):
        class Array:
            def __init__(self, values): self.values = np.asarray(values)
            def cpu(self): return self
            def tolist(self): return self.values.tolist()
        backend = object.__new__(UltralyticsByteTrackBackend)
        backend._diagnostics_enabled = False
        callbacks = []
        backend.model = SimpleNamespace(add_callback=lambda event, callback: callbacks.append((event, callback)))
        backend.enable_diagnostics()
        backend.enable_diagnostics()
        self.assertEqual(len(callbacks), 1)
        boxes = SimpleNamespace(is_track=False, xyxy=Array([[1, 2, 3, 4]]), conf=Array([.9]), cls=Array([2]))
        callbacks[0][1](SimpleNamespace(results=[SimpleNamespace(boxes=boxes)]))
        boxes.xyxy.values[0][0] = 999
        self.assertEqual(backend.raw_detections[0]["bbox"], [1, 2, 3, 4])
        self.assertEqual(backend.raw_detection_status, "before_association")
        backend.raw_detections = None
        boxes.is_track = True
        backend._capture_detections(SimpleNamespace(results=[SimpleNamespace(boxes=boxes)]))
        self.assertIsNone(backend.raw_detections)
        self.assertEqual(backend.raw_detection_status, "callback_after_association")
