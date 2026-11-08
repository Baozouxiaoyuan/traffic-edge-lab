from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np

from traffic_edge.models import TrackObservation
from traffic_edge.pipeline import assign_lane_for_point, run_pipeline, create_event_clips


class _FixedBackend:
    def __init__(self, track_id: int, point: tuple[int, int]) -> None:
        self.track_id = track_id
        self.point = point

    def track(self, _frame: np.ndarray) -> list[TrackObservation]:
        x, y = self.point
        return [TrackObservation(self.track_id, (x - 10, y - 20, x + 10, y), "car", 0.8)]


class _MovingFallbackBackend:
    def __init__(self) -> None:
        self.frame_index = 0

    def track(self, _frame: np.ndarray) -> list[TrackObservation]:
        self.frame_index += 1
        y = 105 - self.frame_index * 7
        return [TrackObservation(1_000_000, (70, y - 20, 90, y), "car", 0.8)]


def _write_test_video(path: Path, frames: int = 4) -> None:
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 10.0, (160, 120))
    if not writer.isOpened():
        raise RuntimeError("Could not create reliability-test video")
    try:
        for _ in range(frames):
            writer.write(np.zeros((120, 160, 3), dtype=np.uint8))
    finally:
        writer.release()


def _restricted_lane(
    lane_id: str,
    polygon: list[list[float]],
) -> dict:
    return {
        "id": lane_id,
        "name": lane_id,
        "enabled_rules": {
            "red_light": False,
            "wrong_way": False,
            "failure_to_stop": False,
            "restricted_zone": True,
        },
        "forbidden_region": {
            "name": "controlled_approach_zone",
            "normalized_polygon": polygon,
        },
        "rule": {"event_type": "candidate_red_signal_violation", "min_track_age_frames": 2},
        "restricted_zone": {
            "name": "restricted_vehicle_zone",
            "normalized_polygon": polygon,
            "trigger_on_initial_presence": True,
            "min_track_age_frames": 2,
        },
    }


def _config(lanes: list[dict]) -> dict:
    return {
        "camera_id": "reliability-test-camera",
        "lanes": lanes,
        "evidence": {"event_clips": False, "export_bundle": False},
    }


class LaneAssignmentTests(unittest.TestCase):
    def test_overlap_uses_nearest_axis_and_abstains_on_tie(self) -> None:
        lanes = [
            {
                "id": "left",
                "polygon": [(0, 0), (70, 0), (70, 100), (0, 100)],
                "direction_axis": ((25.0, 90.0), (25.0, 10.0)),
            },
            {
                "id": "right",
                "polygon": [(30, 0), (100, 0), (100, 100), (30, 100)],
                "direction_axis": ((75.0, 90.0), (75.0, 10.0)),
            },
        ]
        lane, status = assign_lane_for_point((40.0, 50.0), lanes)
        self.assertEqual(lane["id"], "left")
        self.assertEqual(status, "assigned_overlap_by_axis")

        lane, status = assign_lane_for_point((50.0, 50.0), lanes)
        self.assertIsNone(lane)
        self.assertEqual(status, "ambiguous_lane_overlap")

        lane, status = assign_lane_for_point((120.0, 50.0), lanes)
        self.assertIsNone(lane)
        self.assertEqual(status, "outside_all_lanes")


class PipelineReliabilityTests(unittest.TestCase):
    def test_evidence_clip_reports_only_existing_context_at_end_of_video(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);video=root/'source.avi';_write_test_video(video,frames=20)
            event={'event_id':'near-end','timestamp_seconds':1.8}
            self.assertEqual(create_event_clips(video,root,[event],pre_seconds=.5,post_seconds=3),1)
            start,end=event['evidence_clip_window_seconds']
            self.assertAlmostEqual(end,2.)
            cap=cv2.VideoCapture(str(root/'near-end-clip.webm'))
            duration=cap.get(cv2.CAP_PROP_FRAME_COUNT)/cap.get(cv2.CAP_PROP_FPS);cap.release()
            self.assertAlmostEqual(end-start,duration,places=3)

    def test_failed_clip_encoder_does_not_advertise_nonexistent_evidence(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);video=root/'source.avi';_write_test_video(video,frames=20)
            event={'event_id':'encoder-failure','timestamp_seconds':1.}
            with patch('traffic_edge.pipeline.cv2.VideoWriter') as factory:
                factory.return_value.isOpened.return_value=False
                self.assertEqual(create_event_clips(video,root,[event]),0)
            self.assertNotIn('evidence_clip_pointer',event)
            self.assertNotIn('evidence_clip_window_seconds',event)

    def test_pipeline_breaks_crossing_state_when_a_confirmed_track_disappears(self) -> None:
        for gap in (False, True):
            with self.subTest(gap=gap), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                video = root/'input.avi'; _write_test_video(video)
                lane = _restricted_lane('lane-1', [[0, .5], [1, .5], [1, 1], [0, 1]])
                lane['enabled_rules'].update(red_light=True, restricted_zone=False)
                lane['rule']['min_track_age_frames'] = 2
                lane['controlled_movement'] = {'normalized_direction': [[.5, .9], [.5, .2]], 'minimum_motion_pixels': 5}
                lane['stop_line'] = {'normalized_points': [[0, .5], [1, .5]]}
                lane['traffic_signal'] = {'normalized_rectangle': [[.01, .01], [.1, .2]]}
                config_path = root/'config.json'; config_path.write_text(json.dumps(_config([lane])))
                positions = (100, 80, None if gap else 70, 50)
                sequence = iter([[] if y is None else [TrackObservation(7, (70, y-20, 90, y), 'car', .9)] for y in positions])
                backend = _FixedBackend(7, (80, 100))
                backend.track = lambda _frame: next(sequence)
                with patch('traffic_edge.pipeline.build_backend', return_value=backend), patch('traffic_edge.pipeline.RedSignalDetector.update', return_value=True):
                    summary = run_pipeline(video, root/'run', config_path, 'yolo')
                self.assertEqual(summary['candidate_events'], 0 if gap else 1)

    def test_zero_decoded_frames_cannot_be_reported_as_successful_analysis(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config_path = root/'config.json'
            config_path.write_text(json.dumps(_config([_restricted_lane('lane-1', [[0, 0], [1, 0], [1, 1], [0, 1]])])))
            with patch('traffic_edge.pipeline.cv2.VideoCapture') as factory, patch('traffic_edge.pipeline.build_backend', return_value=_FixedBackend(1, (80, 80))):
                capture = factory.return_value
                capture.isOpened.return_value = True
                capture.get.side_effect = [160, 120, 10, 4]
                capture.read.return_value = (False, None)
                with self.assertRaisesRegex(ValueError, 'no readable frames'):
                    run_pipeline(root/'bad.avi', root/'run', config_path, 'demo')
                self.assertFalse((root/'run'/'summary.json').exists())
                capture.release.assert_called_once()

    def test_model_initialization_failure_releases_open_video_resources(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config_path = root / 'config.json'
            config_path.write_text(json.dumps(_config([_restricted_lane('lane-1', [[0, 0], [1, 0], [1, 1], [0, 1]])])))
            with patch('traffic_edge.pipeline.cv2.VideoCapture') as capture_factory, patch('traffic_edge.pipeline.cv2.VideoWriter') as writer_factory:
                capture = capture_factory.return_value
                capture.isOpened.return_value = True
                capture.get.side_effect = [160, 120, 10, 4]
                writer_factory.return_value.isOpened.return_value = True
                with patch('traffic_edge.pipeline.build_backend', side_effect=RuntimeError('model failed')):
                    with self.assertRaisesRegex(RuntimeError, 'model failed'):
                        run_pipeline(root/'input.avi', root/'run', config_path, 'yolo')
                capture.release.assert_called_once()
                writer_factory.return_value.release.assert_called_once()

    def test_real_red_light_run_rejects_missing_signal_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            video = root/'input.avi'
            _write_test_video(video)
            lane = _restricted_lane('lane-1', [[0, 0], [1, 0], [1, 1], [0, 1]])
            lane['enabled_rules']['red_light'] = True
            config_path = root/'config.json'
            config_path.write_text(json.dumps(_config([lane])))
            with patch('traffic_edge.pipeline.build_backend') as backend:
                with self.assertRaisesRegex(ValueError, 'controlling-signal region'):
                    run_pipeline(video, root/'run', config_path, 'yolo')
                backend.assert_not_called()

    def test_fallback_tracks_are_reported_but_cannot_trigger_events(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            video = root / "input.avi"
            _write_test_video(video)
            config_path = root / "config.json"
            config_path.write_text(
                json.dumps(_config([_restricted_lane("lane-1", [[0, 0], [1, 0], [1, 1], [0, 1]])])),
                encoding="utf-8",
            )
            with patch("traffic_edge.pipeline.build_backend", return_value=_FixedBackend(1_000_000, (80, 80))):
                summary = run_pipeline(video, root / "run", config_path, "demo")
            self.assertEqual(summary["candidate_events"], 0)
            self.assertEqual(summary["fallback_observations_excluded"], 4)
            self.assertEqual(summary["fallback_track_ids_excluded"], 1)

    def test_persistent_fallback_motion_can_create_review_flagged_wrong_way_candidate(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            video = root / "input.avi"
            _write_test_video(video, frames=8)
            lane = {
                "id": "lane-1",
                "name": "lane-1",
                "enabled_rules": {
                    "red_light": False,
                    "wrong_way": True,
                    "failure_to_stop": False,
                    "restricted_zone": False,
                },
                "forbidden_region": {
                    "name": "controlled_approach_zone",
                    "normalized_polygon": [[0, 0], [1, 0], [1, 1], [0, 1]],
                },
                "rule": {"event_type": "candidate_red_signal_violation", "min_track_age_frames": 3},
                "controlled_movement": {
                    "normalized_direction": [[0.5, 0.1], [0.5, 0.9]],
                    "minimum_motion_pixels": 4,
                },
                "wrong_way": {
                    "minimum_inside_seconds": 0.2,
                    "reference_frame_height": 120,
                    "minimum_motion_pixels": 4,
                    "maximum_alignment": -0.7,
                    "consecutive_frames": 2,
                    "minimum_opposite_step_ratio": 0.6,
                    "enable_tracklet_stitching": False,
                },
            }
            config_path = root / "config.json"
            config_path.write_text(json.dumps(_config([lane])), encoding="utf-8")
            with patch("traffic_edge.pipeline.build_backend", return_value=_MovingFallbackBackend()):
                summary = run_pipeline(video, root / "run", config_path, "demo")
            events = [json.loads(line) for line in (root / "run" / "events.jsonl").read_text(encoding="utf-8").splitlines()]
            self.assertEqual(summary["candidate_events"], 1)
            self.assertEqual(summary["fallback_observations_excluded"], 0)
            self.assertEqual(summary["fallback_wrong_way_track_ids"], 1)
            self.assertIn("fallback_tracking_id", events[0]["review_flags"])

    def test_overlapping_lanes_produce_at_most_one_lane_owned_event(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            video = root / "input.avi"
            _write_test_video(video)
            lanes = [
                _restricted_lane("left", [[0, 0], [.7, 0], [.7, 1], [0, 1]]),
                _restricted_lane("right", [[.3, 0], [1, 0], [1, 1], [.3, 1]]),
            ]
            config_path = root / "config.json"
            config_path.write_text(json.dumps(_config(lanes)), encoding="utf-8")
            with patch("traffic_edge.pipeline.build_backend", return_value=_FixedBackend(7, (64, 80))):
                summary = run_pipeline(video, root / "run", config_path, "demo")
            self.assertEqual(summary["candidate_events"], 1)
            self.assertEqual(summary["candidate_events_by_lane"], {"left": 1})
            self.assertEqual(summary["lane_assignment_diagnostics"]["assigned_overlap_by_centroid"], 4)


if __name__ == "__main__":
    unittest.main()
