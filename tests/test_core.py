from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from traffic_edge.demo import generate_builtin_examples
from traffic_edge.geometry import denormalize_polygon, point_in_polygon
from traffic_edge.lane_assist import suggest_lane_corridor
from traffic_edge.evaluation import audit_study_manifest, evaluate_events, evaluate_study
from traffic_edge.backends import FallbackDetectionTracker
from traffic_edge.campaign import validate_campaign
from traffic_edge.models import TrackObservation
from traffic_edge.pipeline import _fit_overlay_text, build_incidents, overlay_style, run_pipeline
from traffic_edge.rules import DirectedStopLineRule, RegionEntryRule, WrongWayRule
from traffic_edge.signal import RedSignalDetector
from traffic_edge.web_ui import create_config, safe_name


class GeometryTests(unittest.TestCase):
    def test_normalized_polygon_and_containment(self) -> None:
        polygon = denormalize_polygon([[0.5, 0.0], [1.0, 0.0], [1.0, 1.0], [0.5, 1.0]], 100, 100)
        self.assertTrue(point_in_polygon((75, 50), polygon))
        self.assertTrue(point_in_polygon((50, 50), polygon))
        self.assertFalse(point_in_polygon((25, 50), polygon))


class EvidenceOverlayTests(unittest.TestCase):
    def test_overlay_scales_down_for_low_resolution_video(self) -> None:
        small = overlay_style(360, 240)
        large = overlay_style(1920, 1080)
        self.assertLess(float(small["font"]), float(large["font"]))
        self.assertLessEqual(int(small["event_thickness"]), int(large["event_thickness"]))
        self.assertEqual(int(small["text_thickness"]), 1)

    def test_overlay_text_is_ellipsized_to_available_width(self) -> None:
        text = _fit_overlay_text("WRONG WAY | VEHICLE 1000009", 72, 0.30, 1)
        width = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.30, 1)[0][0]
        self.assertLessEqual(width, 72)
        self.assertTrue(text.endswith("..."))

    def test_lane_assistant_fits_two_visible_boundaries(self) -> None:
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        cv2.line(frame, (350, 650), (560, 150), (255, 255, 255), 8)
        cv2.line(frame, (930, 650), (720, 150), (255, 255, 255), 8)
        result = suggest_lane_corridor(frame, [[0.5, 0.9], [0.5, 0.2]])
        self.assertEqual(len(result["polygon"]), 4)
        self.assertEqual(len(result["boundary_lines"]), 2)
        self.assertGreaterEqual(result["confidence"], 0.6)
        self.assertEqual(result["confidence_label"], "geometric_fit_score_not_detection_accuracy")
        self.assertTrue(result["review_required"])
        self.assertEqual(result["boundary_lines"][0][:2], result["polygon"][0])
        self.assertEqual(result["boundary_lines"][0][2:], result["polygon"][1])

    def test_lane_assistant_rejects_lines_outside_semantic_road_mask(self) -> None:
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        cv2.line(frame, (350, 650), (560, 150), (255, 255, 255), 8)
        cv2.line(frame, (930, 650), (720, 150), (255, 255, 255), 8)
        road_mask = np.zeros((720, 1280), dtype=np.uint8)
        road_mask[:80, :80] = 255
        with self.assertRaisesRegex(ValueError, "lane-like lines|boundaries"):
            suggest_lane_corridor(frame, [[0.5, 0.9], [0.5, 0.2]], road_mask)


class IncidentTests(unittest.TestCase):
    def test_same_vehicle_rule_events_are_consolidated_without_merging_other_tracks(self) -> None:
        common = {"camera_id": "camera-a", "lane_id": "lane-1", "lane_name": "Lane 1", "review_flags": []}
        events = [
            {**common, "event_id": "e1", "track_id": 7, "timestamp_seconds": 5.0, "candidate_event": "candidate_red_signal_violation", "evidence_quality_score": 0.8},
            {**common, "event_id": "e2", "track_id": 7, "timestamp_seconds": 5.8, "candidate_event": "candidate_restricted_zone_entry", "evidence_quality_score": 0.7},
            {**common, "event_id": "e3", "track_id": 8, "timestamp_seconds": 5.9, "candidate_event": "candidate_wrong_way_driving", "evidence_quality_score": 0.6},
        ]
        incidents = build_incidents(events)
        self.assertEqual(len(incidents), 2)
        self.assertTrue(incidents[0]["multi_rule"])
        self.assertEqual(incidents[0]["event_count"], 2)
        self.assertEqual(incidents[1]["track_id"], 8)


class RuleTests(unittest.TestCase):
    def test_signal_detector_uses_dominance_and_unknown_state(self) -> None:
        detector = RedSignalDetector((0, 0, 40, 40), minimum_red_ratio=0.01, history_size=3)
        red = np.zeros((40, 40, 3), dtype=np.uint8)
        cv2.circle(red, (20, 20), 7, (0, 0, 255), -1)
        self.assertFalse(detector.update(red))
        self.assertFalse(detector.update(red))
        self.assertTrue(detector.update(red))
        self.assertEqual(detector.last_state, "red")
        blank = np.full((40, 40, 3), 80, dtype=np.uint8)
        self.assertFalse(detector.update(blank))
        self.assertEqual(detector.last_state, "unknown")

        conflict = np.zeros((40, 40, 3), dtype=np.uint8)
        cv2.circle(conflict, (12, 20), 6, (0, 0, 255), -1)
        cv2.circle(conflict, (28, 20), 9, (0, 255, 0), -1)
        self.assertFalse(detector.update(conflict))
        self.assertEqual(detector.last_state, "not_red")

    def test_signal_detector_distinguishes_yellow_housing_from_an_amber_lamp(self) -> None:
        detector = RedSignalDetector((0, 0, 40, 40), history_size=1)
        frame = np.zeros((40, 40, 3), dtype=np.uint8)
        amber = (0, 170, 255)
        frame[:4, :] = amber
        cv2.circle(frame, (20, 12), 3, (0, 0, 255), -1)
        self.assertTrue(detector.update(frame))
        self.assertEqual(detector.last_state, "red")

        frame[4:, :] = 0
        self.assertFalse(detector.update(frame))
        self.assertEqual(detector.last_state, "unknown")

        cv2.circle(frame, (20, 23), 3, amber, -1)
        self.assertFalse(detector.update(frame))
        self.assertEqual(detector.last_state, "not_red")

    def test_fallback_tracker_preserves_an_unconfirmed_moving_detection(self) -> None:
        tracker = FallbackDetectionTracker()
        ids = []
        for x, y in ((100, 150), (104, 144), (108, 138), (112, 132)):
            observation = tracker.update([((x - 10, y - 10, x + 10, y), "car", 0.2)])[0]
            ids.append(observation.track_id)
        self.assertEqual(len(set(ids)), 1)
        self.assertGreaterEqual(ids[0], 1_000_000)

    def test_region_entry_fires_once(self) -> None:
        rule = RegionEntryRule([(50, 0), (100, 0), (100, 100), (50, 100)], "test", "candidate", 2)
        outside = TrackObservation(7, (0, 0, 20, 20), "car", 0.9)
        inside = TrackObservation(7, (60, 0, 80, 20), "car", 0.9)
        self.assertIsNone(rule.evaluate(outside))
        trigger = rule.evaluate(inside)
        self.assertIsNotNone(trigger)
        self.assertIsNone(rule.evaluate(inside))

    def test_region_entry_can_flag_persistent_initial_presence_when_enabled(self) -> None:
        polygon = [(0, 0), (100, 0), (100, 100), (0, 100)]
        rule = RegionEntryRule(
            polygon,
            "restricted",
            "candidate_restricted_zone_entry",
            min_track_age_frames=3,
            trigger_on_initial_presence=True,
        )
        inside = TrackObservation(9, (40, 30, 60, 50), "car", 0.9)
        self.assertIsNone(rule.evaluate(inside))
        self.assertIsNone(rule.evaluate(inside))
        self.assertIsNotNone(rule.evaluate(inside))
        self.assertIsNone(rule.evaluate(inside))

    def test_red_signal_and_direction_are_both_required(self) -> None:
        polygon = [(50, 0), (100, 0), (100, 100), (50, 100)]
        outside = TrackObservation(7, (0, 0, 20, 20), "car", 0.9)
        inside = TrackObservation(7, (60, 0, 80, 20), "car", 0.9)
        green_rule = RegionEntryRule(polygon, "test", "candidate", 2, (1.0, 0.0), 0.5, 5.0)
        self.assertIsNone(green_rule.evaluate(outside, signal_is_red=False))
        self.assertIsNone(green_rule.evaluate(inside, signal_is_red=False))
        wrong_direction_rule = RegionEntryRule(polygon, "test", "candidate", 2, (-1.0, 0.0), 0.5, 5.0)
        self.assertIsNone(wrong_direction_rule.evaluate(outside, signal_is_red=True))
        self.assertIsNone(wrong_direction_rule.evaluate(inside, signal_is_red=True))
        matching_rule = RegionEntryRule(polygon, "test", "candidate", 2, (1.0, 0.0), 0.5, 5.0)
        self.assertIsNone(matching_rule.evaluate(outside, signal_is_red=True))
        self.assertIsNotNone(matching_rule.evaluate(inside, signal_is_red=True))

    def test_directed_stop_line_excludes_side_traffic(self) -> None:
        approach = [(0, 50), (100, 50), (100, 100), (0, 100)]
        controlled = DirectedStopLineRule(approach, ((0, 50), (100, 50)), (50, 90), (0, -1), "approach", "event", 3, 0.75, 5)
        self.assertIsNone(controlled.evaluate(TrackObservation(1, (40, 70, 60, 90), "car", 0.9), True))
        self.assertIsNone(controlled.evaluate(TrackObservation(1, (40, 50, 60, 70), "car", 0.9), True))
        self.assertIsNotNone(controlled.evaluate(TrackObservation(1, (40, 25, 60, 45), "car", 0.9), True))
        side = DirectedStopLineRule(approach, ((0, 50), (100, 50)), (50, 90), (0, -1), "approach", "event", 3, 0.75, 5)
        self.assertIsNone(side.evaluate(TrackObservation(2, (0, 60, 20, 80), "car", 0.9), True))
        self.assertIsNone(side.evaluate(TrackObservation(2, (35, 60, 55, 80), "car", 0.9), True))
        self.assertIsNone(side.evaluate(TrackObservation(2, (70, 60, 90, 80), "car", 0.9), True))

    def test_wrong_way_requires_persistent_opposite_motion(self) -> None:
        polygon = [(0, 0), (100, 0), (100, 100), (0, 100)]
        wrong_way = WrongWayRule(polygon, (1, 0), min_track_age_frames=3, min_motion_pixels=5, consecutive_frames=2)
        trigger = None
        for x in (80, 70, 60, 50):
            trigger = wrong_way.evaluate(TrackObservation(11, (x - 5, 30, x + 5, 50), "car", 0.9)) or trigger
        self.assertIsNotNone(trigger)
        self.assertEqual(trigger.rule_name, "candidate_wrong_way_driving")
        legal = WrongWayRule(polygon, (1, 0), min_track_age_frames=3, min_motion_pixels=5, consecutive_frames=2)
        for x in (20, 30, 40, 50):
            self.assertIsNone(legal.evaluate(TrackObservation(12, (x - 5, 30, x + 5, 50), "car", 0.9)))

    def test_wrong_way_ignores_motion_before_entering_selected_lane(self) -> None:
        polygon = [(50, 0), (100, 0), (100, 100), (50, 100)]
        rule = WrongWayRule(polygon, (1, 0), min_track_age_frames=3, min_motion_pixels=5, consecutive_frames=2)
        for x in (40, 30, 20):
            self.assertIsNone(rule.evaluate(TrackObservation(30, (x - 5, 30, x + 5, 50), "car", 0.9)))
        for x in (60, 70, 80, 90):
            self.assertIsNone(rule.evaluate(TrackObservation(30, (x - 5, 30, x + 5, 50), "car", 0.9)))

    def test_wrong_way_rejects_vehicle_straddling_lane_boundary(self) -> None:
        polygon = [(50, 0), (100, 0), (100, 100), (50, 100)]
        rule = WrongWayRule(polygon, (1, 0), min_track_age_frames=3, min_motion_pixels=5, consecutive_frames=2)
        for centre in (58, 56, 54, 52):
            self.assertIsNone(rule.evaluate(TrackObservation(31, (centre - 20, 30, centre + 20, 50), "car", 0.9)))

    def test_wrong_way_motion_threshold_scales_with_vehicle_height(self) -> None:
        polygon = [(0, 0), (200, 0), (200, 200), (0, 200)]
        rule = WrongWayRule(
            polygon, (1, 0), min_track_age_frames=3, min_motion_pixels=3,
            min_motion_bbox_height_ratio=0.35, consecutive_frames=2,
        )
        # Eight pixels is above the absolute floor but is only detector jitter
        # for this 80-pixel-tall nearby vehicle.
        for x in (100, 96, 92, 88):
            self.assertIsNone(rule.evaluate(TrackObservation(40, (x - 20, 40, x + 20, 120), "car", 0.9)))




class EvaluationTests(unittest.TestCase):
    def test_event_level_metrics(self) -> None:
        predicted = [
            {"event_id": "one", "timestamp_seconds": 3.5, "candidate_event": "lane"},
            {"event_id": "extra", "timestamp_seconds": 8.0, "candidate_event": "lane"},
        ]
        truth = [
            {"timestamp_seconds": 3.6, "candidate_event": "lane"},
            {"timestamp_seconds": 12.0, "candidate_event": "lane"},
        ]
        metrics = evaluate_events(predicted, truth, tolerance_seconds=0.5)
        self.assertEqual(metrics["true_positives"], 1)
        self.assertEqual(metrics["false_positives"], 1)
        self.assertEqual(metrics["false_negatives"], 1)
        self.assertEqual(metrics["f1"], 0.5)

    def test_spatial_matching_distinguishes_vehicles_at_same_time(self) -> None:
        predicted = [
            {"event_id": "side", "timestamp_seconds": 3.5, "candidate_event": "event", "ground_point_normalized": [0.8, 0.5]},
            {"event_id": "target", "timestamp_seconds": 3.5, "candidate_event": "event", "ground_point_normalized": [0.31, 0.52]},
        ]
        truth = [{"timestamp_seconds": 3.5, "candidate_event": "event", "vehicle_point_normalized": [0.3, 0.5]}]
        metrics = evaluate_events(predicted, truth, tolerance_seconds=0.5, spatial_tolerance_normalized=0.1)
        self.assertEqual(metrics["true_positives"], 1)
        self.assertEqual(metrics["false_positives"], 1)
        self.assertEqual(metrics["false_negatives"], 0)

    def test_lane_identity_is_part_of_event_matching(self) -> None:
        predicted = [{"event_id": "lane-one", "timestamp_seconds": 4.0, "candidate_event": "event", "lane_id": "lane-1"}]
        truth = [{"timestamp_seconds": 4.0, "candidate_event": "event", "lane_id": "lane-2"}]
        metrics = evaluate_events(predicted, truth)
        self.assertEqual(metrics["true_positives"], 0)
        self.assertEqual(metrics["false_positives"], 1)
        self.assertEqual(metrics["false_negatives"], 1)

    def test_multi_video_study_reports_rule_metrics_rate_and_intervals(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "events-one.jsonl").write_text(
                "\n".join(json.dumps(item) for item in [
                    {"event_id": "tp", "timestamp_seconds": 2.0, "candidate_event": "wrong_way"},
                    {"event_id": "fp", "timestamp_seconds": 8.0, "candidate_event": "wrong_way"},
                ]) + "\n",
                encoding="utf-8",
            )
            (root / "truth-one.json").write_text(
                json.dumps([{"timestamp_seconds": 2.1, "candidate_event": "wrong_way"}]), encoding="utf-8"
            )
            (root / "events-two.jsonl").write_text("", encoding="utf-8")
            (root / "truth-two.json").write_text(
                json.dumps([{"timestamp_seconds": 4.0, "candidate_event": "wrong_way"}]), encoding="utf-8"
            )
            manifest = {
                "study_name": "test-study",
                "bootstrap_iterations": 100,
                "videos": [
                    {"video_id": "one", "camera_id": "camera-a", "split": "test", "status": "labelled", "duration_seconds": 600, "events_path": "events-one.jsonl", "ground_truth_path": "truth-one.json"},
                    {"video_id": "two", "camera_id": "camera-b", "split": "test", "status": "labelled", "duration_seconds": 600, "events_path": "events-two.jsonl", "ground_truth_path": "truth-two.json"},
                    {"video_id": "dev", "camera_id": "camera-dev", "split": "development", "status": "unlabelled"},
                ],
            }
            manifest_path = root / "manifest.json"
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            result = evaluate_study(manifest_path)
            self.assertEqual(result["micro_average"]["true_positives"], 1)
            self.assertEqual(result["micro_average"]["false_positives"], 1)
            self.assertEqual(result["micro_average"]["false_negatives"], 1)
            self.assertEqual(result["false_positives_per_hour"], 3.0)
            self.assertIsNotNone(result["confidence_intervals_95"])
            self.assertEqual(result["protocol"]["bootstrap_sampling_unit"], "source_group")
            self.assertEqual(result["dataset"]["unique_source_groups"], 2)
            self.assertEqual(result["by_type"]["wrong_way"]["f1"], 0.5)
            self.assertFalse(result["source_overlap_warnings"])

    def test_error_analysis_and_evidence_threshold_diagnostics(self) -> None:
        predicted = [
            {"event_id": "tp", "timestamp_seconds": 2.1, "candidate_event": "wrong_way", "evidence_quality_score": 0.8},
            {"event_id": "fp", "timestamp_seconds": 8.0, "candidate_event": "wrong_way", "evidence_quality_score": 0.3},
        ]
        truth = [
            {"timestamp_seconds": 2.0, "candidate_event": "wrong_way"},
            {"timestamp_seconds": 12.0, "candidate_event": "wrong_way"},
        ]
        result = evaluate_events(predicted, truth)
        self.assertEqual(result["false_positive_events"][0]["event_id"], "fp")
        self.assertEqual(result["unmatched_ground_truth_events"][0]["timestamp_seconds"], 12.0)
        self.assertEqual(result["localization_errors"]["mean_timestamp_error_seconds"], 0.1)
        at_half = next(item for item in result["evidence_threshold_diagnostics"] if item["minimum_evidence_strength"] == 0.5)
        self.assertEqual(at_half["true_positives"], 1)
        self.assertEqual(at_half["false_positives"], 0)

    def test_study_audit_reports_source_overlap_and_incomplete_labels(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "manifest.json"
            path.write_text(json.dumps({"videos": [
                {"video_id": "dev", "camera_id": "camera-a", "split": "development", "status": "unlabelled"},
                {"video_id": "test", "camera_id": "camera-a", "split": "test", "status": "needs_full_labels"},
            ]}), encoding="utf-8")
            report = audit_study_manifest(path)
            self.assertFalse(report["ready_for_final_accuracy_report"])
            self.assertEqual(report["source_group_overlap"], ["camera-a"])
            self.assertEqual(report["incomplete_test_videos"], ["test"])

    def test_campaign_validation_requires_complete_separated_inputs(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for name in ("model-a.pt", "model-b.pt", "video.mp4", "config.json", "truth.json"):
                (root / name).write_bytes(b"[]" if name.endswith(".json") else b"x")
            (root/'config.json').write_text(json.dumps({'enabled_rules': {'red_light': True}}), encoding='utf-8')
            (root/'validation.mp4').write_bytes(b'distinct-validation-source')
            cases = [{
                "video_id": "validation-video",
                "camera_id": "validation-camera",
                "split": "validation",
                "status": "labelled",
                "input_video": "validation.mp4",
                "config": "config.json",
                "ground_truth": "truth.json",
            }] + [
                {"video_id": f"video-{index}", "camera_id": f"camera-{index}", "split": "test", "status": "labelled", "input_video": "video.mp4", "config": "config.json", "ground_truth": "truth.json"}
                for index in range(3)
            ]
            manifest = {
                "variants": [
                    {"name": "a", "model": "model-a.pt", "tracker": "bytetrack"},
                    {"name": "b", "model": "model-b.pt", "tracker": "botsort"},
                ],
                "cases": cases,
            }
            path = root / "campaign.json"
            path.write_text(json.dumps(manifest), encoding="utf-8")
            report = validate_campaign(path)
            self.assertTrue(report["ready"])
            self.assertEqual(report["planned_runs"], 5)
            self.assertEqual(report["validation_case_count"], 1)
            self.assertEqual(report["unique_test_source_groups"], 3)
            self.assertFalse(report["warnings"])


class WebUiTests(unittest.TestCase):
    def test_generated_config_uses_normalized_polygon(self) -> None:
        config = create_config(
            "Camera 01",
            [[0.1, 0.2], [0.8, 0.2], [0.5, 0.9]],
            0.25,
            [[0.1, 0.1], [0.2, 0.2]],
            [[0.5, 0.9], [0.5, 0.2]],
            [[0.2, 0.5], [0.8, 0.5]],
        )
        self.assertEqual(config["camera_id"], "Camera-01")
        self.assertEqual(len(config["forbidden_region"]["normalized_polygon"]), 3)
        self.assertEqual(config["rule"]["event_type"], "candidate_red_signal_violation")
        self.assertEqual(len(config["traffic_signal"]["normalized_rectangle"]), 2)
        self.assertEqual(len(config["controlled_movement"]["normalized_direction"]), 2)
        self.assertEqual(len(config["stop_line"]["normalized_points"]), 2)
        self.assertEqual(config["yolo"]["vehicle_classes"], [2, 5, 7])
        self.assertEqual(config["yolo"]["tracker"], "bytetrack")
        self.assertTrue(config["evidence"]["event_clips"])
        self.assertTrue(config["evidence"]["export_bundle"])
        self.assertEqual(config["wrong_way"]["minimum_inside_seconds"], 0.2)

    def test_generated_config_accepts_botsort_and_rejects_unknown_tracker(self) -> None:
        arguments = (
            "Camera 02",
            [[0.1, 0.2], [0.8, 0.2], [0.5, 0.9]],
            0.25,
            [[0.1, 0.1], [0.2, 0.2]],
            [[0.5, 0.9], [0.5, 0.2]],
            [[0.2, 0.5], [0.8, 0.5]],
        )
        self.assertEqual(create_config(*arguments, tracker_name="botsort")["yolo"]["tracker"], "botsort")
        with self.assertRaisesRegex(ValueError, "Tracker"):
            create_config(*arguments, tracker_name="unknown")

    def test_safe_name(self) -> None:
        self.assertEqual(safe_name("my video.mp4"), "my-video-mp4")

    def test_generated_config_enables_multiple_rules(self) -> None:
        config = create_config(
            "multi-camera",
            [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]],
            0.25,
            [[0.1, 0.1], [0.2, 0.2]],
            [[0.8, 0.5], [0.2, 0.5]],
            [[0.2, 0.5], [0.8, 0.5]],
            [[0.6, 0.0], [1.0, 0.0], [1.0, 1.0], [0.6, 1.0]],
            {"red_light": True, "wrong_way": True, "restricted_zone": True},
        )
        self.assertTrue(all(config["enabled_rules"].values()))
        self.assertEqual(len(config["restricted_zone"]["normalized_polygon"]), 4)
        self.assertNotIn("failure_to_stop", config)
        self.assertEqual(set(config['enabled_rules']), {'red_light', 'wrong_way', 'restricted_zone'})

    def test_generated_config_supports_independent_lanes(self) -> None:
        lanes = [
            {"id": "northbound", "name": "Northbound", "polygon": [[0, .35], [1, .35], [1, .70], [0, .70]], "direction": [[.2, .55], [.8, .55]], "enabled_rules": {"wrong_way": True}},
            {"id": "southbound", "name": "Southbound", "polygon": [[0, .70], [1, .70], [1, 1], [0, 1]], "direction": [[.8, .85], [.2, .85]], "enabled_rules": {"wrong_way": True}},
        ]
        config = create_config("multi-lane", lanes[0]["polygon"], .25, direction=lanes[0]["direction"], enabled_rules=lanes[0]["enabled_rules"], lanes=lanes)
        self.assertEqual([lane["id"] for lane in config["lanes"]], ["northbound", "southbound"])
        self.assertNotEqual(config["lanes"][0]["controlled_movement"], config["lanes"][1]["controlled_movement"])


class SyntheticRegressionTests(unittest.TestCase):
    def test_multiple_lanes_share_one_tracking_pass_but_keep_event_ownership(self) -> None:
        lanes = [
            {"id": "upper", "name": "Upper lane", "polygon": [[0, .35], [1, .35], [1, .70], [0, .70]], "direction": [[.20, .55], [.80, .55]], "enabled_rules": {"wrong_way": True}},
            {"id": "lower", "name": "Lower lane", "polygon": [[0, .70], [1, .70], [1, 1], [0, 1]], "direction": [[.20, .85], [.80, .85]], "enabled_rules": {"wrong_way": True}},
        ]
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            generate_builtin_examples(root)
            config = create_config("multi-lane", lanes[0]["polygon"], .25, direction=lanes[0]["direction"], enabled_rules=lanes[0]["enabled_rules"], lanes=lanes)
            config_path = root / "multi-lane.json"
            config_path.write_text(json.dumps(config), encoding="utf-8")
            summary = run_pipeline(root / "wrong_way.webm", root / "run", config_path, "demo")
            events = [json.loads(line) for line in Path(summary["outputs"]["events"]).read_text(encoding="utf-8").splitlines() if line]
            self.assertEqual(summary["lane_count"], 2)
            self.assertGreater(summary["average_processing_ms_per_frame"], 0)
            self.assertGreater(summary["real_time_factor"], 0)
            self.assertIn("meets_source_frame_rate", summary)
            self.assertEqual(summary["required_fps_for_recommended_headroom"], summary["source_fps"] * 1.25)
            self.assertEqual(len(events), 1)
            self.assertEqual(events[0]["lane_id"], "upper")
            self.assertEqual(summary["event_evidence_clips"], 1)
            self.assertEqual(summary["candidate_incidents"], 1)
            self.assertTrue((root / "run" / events[0]["evidence_clip_pointer"]).is_file())
            self.assertTrue(Path(summary["outputs"]["incidents"]).is_file())
            self.assertTrue(Path(summary["outputs"]["integrity_manifest"]).is_file())
            self.assertTrue(Path(summary["outputs"]["evidence_bundle"]).is_file())
            report_path = Path(summary["outputs"]["run_report"])
            self.assertTrue(report_path.is_file())
            self.assertIn("Event review queue", report_path.read_text(encoding="utf-8"))

    def test_each_generated_fixture_exercises_its_expected_rule(self) -> None:
        scenarios = {
            "red_light": {
                "polygon": [[.35, .52], [.65, .52], [.65, 1], [.35, 1]],
                "signal": [[.465, .045], [.535, .18]], "direction": [[.50, .90], [.50, .25]],
                "stop": [[.35, .50], [.65, .50]], "restricted": [], "event": "candidate_red_signal_violation",
                "rules": {"red_light": True},
            },
            "wrong_way": {
                "polygon": [[0, .35], [1, .35], [1, .70], [0, .70]],
                "signal": [], "direction": [[.20, .55], [.80, .55]], "stop": [], "restricted": [],
                "event": "candidate_wrong_way_driving", "rules": {"wrong_way": True},
            },
            "restricted_zone": {
                "polygon": [], "signal": [], "direction": [], "stop": [],
                "restricted": [[.65, .25], [1, .25], [1, .78], [.65, .78]],
                "event": "candidate_restricted_zone_entry", "rules": {"restricted_zone": True},
            },
        }
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            videos = root / "videos"
            generate_builtin_examples(videos)
            for name, scenario in scenarios.items():
                with self.subTest(name=name):
                    config = create_config(
                        name, scenario["polygon"], 0.25, scenario["signal"], scenario["direction"],
                        scenario["stop"], scenario["restricted"], scenario["rules"],
                    )
                    config_path = root / f"{name}.json"
                    config_path.write_text(json.dumps(config), encoding="utf-8")
                    summary = run_pipeline(videos / f"{name}.webm", root / f"run-{name}", config_path, "demo")
                    self.assertEqual(summary["candidate_events_by_type"], {scenario["event"]: 1})


if __name__ == "__main__":
    unittest.main()
