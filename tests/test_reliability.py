import unittest
import base64
import json
import tempfile
import threading
from pathlib import Path
from http.server import ThreadingHTTPServer
from urllib.request import urlopen, Request
from urllib.error import HTTPError
from urllib.parse import quote
import cv2
import numpy as np
import traffic_edge.web_ui as web_ui
from traffic_edge.web_ui import (
    AppState, make_handler, PROJECT_ROOT, _review_coverage,
    validate_annotation_quality, register_campaign_case,
)
from traffic_edge.backends import FallbackDetectionTracker
from traffic_edge.geometry import validate_polygon
from traffic_edge.models import TrackObservation
from traffic_edge.rules import WrongWayRule
from traffic_edge.demo import generate_demo_video


class ReliabilityTests(unittest.TestCase):
    def test_annotation_quality_detects_duplicate_timing_and_lane_rule_errors(self):
        lane = {
            'id': 'lane-1', 'name': 'Northbound',
            'enabled_rules': {'wrong_way': True, 'red_light': False},
        }
        base = {
            'timestamp_seconds': 2.0,
            'lane_id': 'lane-1',
            'candidate_event': 'candidate_wrong_way_driving',
            'vehicle_bbox_normalized': [.2, .2, .4, .5],
            'status': 'confirmed',
        }
        labels = [
            base,
            base | {'timestamp_seconds': 2.2},
            base | {
                'timestamp_seconds': 11,
                'candidate_event': 'candidate_red_signal_violation',
                'status': 'uncertain',
            },
        ]
        codes = {issue['code'] for issue in validate_annotation_quality(labels, 10, [lane])}
        self.assertIn('probable_duplicate', codes)
        self.assertIn('timestamp_out_of_range', codes)
        self.assertIn('rule_disabled_for_lane', codes)
        self.assertIn('unresolved_uncertain', codes)

    def test_review_coverage_merges_intervals_and_preserves_gaps(self):
        intervals, coverage = _review_coverage([[0, 4], [3.8, 9.8]], 10)
        self.assertEqual(intervals, [[0.0, 9.8]])
        self.assertAlmostEqual(coverage, 0.98)
        self.assertAlmostEqual(_review_coverage([[0, 4]], 10)[1], 0.4)
        intervals, coverage = _review_coverage([[0, 4], [4.3, 10]], 10)
        self.assertEqual(intervals, [[0.0, 4.0], [4.3, 10.0]])
        self.assertAlmostEqual(coverage, .97, msg='Unwatched seek gaps must not pass the 98% gate')

    def test_annotation_quality_requires_rule_specific_geometry(self):
        label = {
            'timestamp_seconds': 2.0,
            'lane_id': 'lane-1',
            'candidate_event': 'candidate_red_signal_violation',
            'vehicle_bbox_normalized': [.2, .2, .4, .5],
            'status': 'confirmed',
        }
        lane = {
            'id': 'lane-1', 'name': 'Inbound',
            'polygon': [[.1, .9], [.4, .2], [.6, .2], [.9, .9]],
            'direction': [[.5, .9], [.5, .2]],
            'stop_line': [], 'signal_rectangle': [],
            'enabled_rules': {'red_light': True},
        }
        codes = {issue['code'] for issue in validate_annotation_quality([label], 10, [lane])}
        self.assertIn('missing_stop_line', codes)
        self.assertIn('missing_signal_rectangle', codes)
        lane['stop_line'] = [[.2, .5], [.8, .5]]
        lane['signal_rectangle'] = [[.45, .05], [.55, .15]]
        self.assertEqual(validate_annotation_quality([label], 10, [lane]), [])

    def test_annotation_draft_round_trip_and_completion_gate(self):
        with tempfile.TemporaryDirectory(dir=PROJECT_ROOT / 'data') as temporary:
            video = Path(temporary) / 'demo.mp4'
            generate_demo_video(video)
            relative_video = video.relative_to(PROJECT_ROOT).as_posix()
            video_query = '?video_path=' + quote(relative_video, safe='')
            original_draft_dir = web_ui.ANNOTATION_DRAFT_DIR
            original_annotation_dir = web_ui.ANNOTATION_DIR
            web_ui.ANNOTATION_DRAFT_DIR = Path(temporary)
            web_ui.ANNOTATION_DIR = Path(temporary)
            server = ThreadingHTTPServer(('127.0.0.1', 0), make_handler(AppState()))
            worker = threading.Thread(target=server.serve_forever, daemon=True)
            worker.start()
            base = f'http://127.0.0.1:{server.server_port}'
            draft = {
                'video_name': 'demo.mp4', 'video_path': relative_video,
                'reviewer': 'reviewer-a', 'mode': 'ground_truth', 'labels': [],
                'reviewed_intervals': [[0, 5]], 'duration_seconds': 10,
                'current_time_seconds': 5,
            }
            try:
                request = Request(base + '/api/annotation-draft', data=json.dumps(draft).encode(), headers={'Content-Type': 'application/json'})
                with urlopen(request) as response:
                    saved = json.load(response)
                self.assertEqual(saved['review_coverage_percent'], 50.0)
                with urlopen(base + '/api/annotation-draft' + video_query) as response:
                    restored = json.load(response)['draft']
                self.assertEqual(restored['reviewer'], 'reviewer-a')
                self.assertEqual(restored['reviewed_intervals'], [[0.0, 5.0]])
                with urlopen(base + '/api/video-metadata' + video_query) as response:
                    metadata = json.load(response)
                self.assertGreater(metadata['fps'], 0)
                self.assertGreater(metadata['frames'], 0)
                final = draft | {'review_complete': True, 'annotations': [], 'camera_id': 'test'}
                with self.assertRaises(HTTPError) as failure:
                    urlopen(Request(base + '/api/save-ground-truth', data=json.dumps(final).encode(), headers={'Content-Type': 'application/json'}))
                self.assertEqual(failure.exception.code, 400)
                final['reviewed_intervals'] = [[0, 9.8]]
                with urlopen(Request(base + '/api/save-ground-truth', data=json.dumps(final).encode(), headers={'Content-Type': 'application/json'})) as response:
                    completed = json.load(response)
                self.assertAlmostEqual(completed['review_coverage_percent'], 98.0)
                review = json.loads((PROJECT_ROOT / completed['review_metadata']).read_text(encoding='utf-8'))
                self.assertEqual(review['reviewer'], 'reviewer-a')
                self.assertEqual(review['source_sha256'], restored['source_sha256'])
                self.assertTrue(review['negative_video'])
                # Finalization synchronizes the draft's actual coverage too,
                # rather than merely marking an older 50% draft as complete.
                with urlopen(base + '/api/annotation-draft' + video_query) as response:
                    finalized_draft = json.load(response)['draft']
                self.assertTrue(finalized_draft['review_complete'])
                self.assertEqual(finalized_draft['review_coverage_percent'], 98.0)
                current = draft | {'reviewed_intervals': [[0, 9.8]], 'current_time_seconds': 8}
                with urlopen(Request(base + '/api/annotation-draft', data=json.dumps(current).encode(), headers={'Content-Type': 'application/json'})) as response:
                    self.assertTrue(json.load(response)['review_complete'], 'Playback autosave must not undo an unchanged finalized review')
                current['reviewer'] = 'different-reviewer'
                with urlopen(Request(base + '/api/annotation-draft', data=json.dumps(current).encode(), headers={'Content-Type': 'application/json'})) as response:
                    self.assertFalse(json.load(response)['review_complete'], 'Changed review content requires finalization again')
            finally:
                server.shutdown(); server.server_close(); worker.join()
                web_ui.ANNOTATION_DRAFT_DIR = original_draft_dir
                web_ui.ANNOTATION_DIR = original_annotation_dir

    def test_camera_profile_round_trip_and_geometry_validation(self):
        with tempfile.TemporaryDirectory(dir=PROJECT_ROOT / 'config') as temporary:
            original_profile_dir = web_ui.CAMERA_PROFILE_DIR
            web_ui.CAMERA_PROFILE_DIR = Path(temporary)
            server = ThreadingHTTPServer(('127.0.0.1', 0), make_handler(AppState()))
            worker = threading.Thread(target=server.serve_forever, daemon=True)
            worker.start()
            base = f'http://127.0.0.1:{server.server_port}'
            profile = {
                'camera_id': 'North Gate Camera',
                'display_name': 'North Gate Camera',
                'source_signature': {'width': 1920, 'height': 1080, 'aspect_ratio': 16 / 9},
                'calibration_time_seconds': 2.5,
                'lanes': [{
                    'id': 'lane-1', 'name': 'Inbound',
                    'polygon': [[.1, .9], [.4, .2], [.6, .2], [.9, .9]],
                    'stop_line': [], 'signal_rectangle': [],
                    'direction': [[.5, .9], [.5, .2]], 'restricted_polygon': [],
                    'restricted_initial_presence': False,
                    'lane_assist': {'method': 'manual'},
                    'enabled_rules': {'red_light': False, 'wrong_way': True,
                                      'failure_to_stop': False, 'restricted_zone': False},
                }],
                'inference': {'confidence': .25, 'image_size': 960,
                              'model_name': 'yolov8n', 'tracker_name': 'bytetrack'},
            }
            try:
                request = Request(base + '/api/camera-profile', data=json.dumps(profile).encode(), headers={'Content-Type': 'application/json'})
                with urlopen(request) as response:
                    saved = json.load(response)['profile']
                self.assertEqual(saved['profile_id'], 'north-gate-camera')
                with urlopen(base + '/api/camera-profiles') as response:
                    listed = json.load(response)['profiles']
                self.assertEqual(listed[0]['lane_count'], 1)
                with urlopen(base + '/api/camera-profile?id=north-gate-camera') as response:
                    restored = json.load(response)['profile']
                self.assertEqual(restored['lanes'][0]['name'], 'Inbound')
                invalid = profile | {'source_signature': {'width': 1920, 'height': 1080, 'aspect_ratio': 1.0}}
                with self.assertRaises(HTTPError) as failure:
                    urlopen(Request(base + '/api/camera-profile', data=json.dumps(invalid).encode(), headers={'Content-Type': 'application/json'}))
                self.assertEqual(failure.exception.code, 400)
            finally:
                server.shutdown(); server.server_close(); worker.join()
                web_ui.CAMERA_PROFILE_DIR = original_profile_dir

    def test_finalized_review_can_register_after_same_video_run(self):
        with tempfile.TemporaryDirectory(dir=PROJECT_ROOT / 'data') as temporary:
            root = Path(temporary)
            annotations = root / 'annotations'
            annotations.mkdir()
            video = root / 'clip.mp4'
            video.write_bytes(b'video')
            truth = annotations / 'truth.json'
            truth.write_text('[]\n', encoding='utf-8')
            config = root / 'frozen-config.json'
            config.write_text('{}\n', encoding='utf-8')
            (root / 'accuracy_campaign.template.json').write_text(json.dumps({
                'experiment_name': 'test',
                'variants': [{'name': 'baseline', 'model': 'model.pt', 'tracker': 'bytetrack'}],
                'cases': [{
                    'video_id': 'clip', 'source_group_id': 'camera-a', 'split': 'validation',
                    'status': 'needs_full_labels', 'input_video': 'clip.mp4',
                    'config': 'missing.json', 'ground_truth': 'missing.json',
                }],
            }), encoding='utf-8')
            original_data = web_ui.DATA_DIR
            original_annotations = web_ui.ANNOTATION_DIR
            web_ui.DATA_DIR = root
            web_ui.ANNOTATION_DIR = annotations
            job = {
                'status': 'complete', 'video_path': str(video),
                'summary': {'outputs': {'configuration': str(config)}},
            }
            try:
                result = register_campaign_case('camera-a', job, truth)
                self.assertTrue(result.endswith('accuracy_campaign.active.json'))
                active = json.loads((root / 'accuracy_campaign.active.json').read_text(encoding='utf-8'))
                self.assertEqual(active['cases'][0]['status'], 'labelled')
                self.assertEqual(active['cases'][0]['ground_truth'], 'annotations/truth.json')
            finally:
                web_ui.DATA_DIR = original_data
                web_ui.ANNOTATION_DIR = original_annotations

    def test_expired_fallback_cannot_claim_new_vehicle(self):
        tracker = FallbackDetectionTracker(max_missed_frames=2)
        detection = [((10, 10, 30, 30), 'car', .5)]
        first = tracker.update(detection)[0].track_id
        for _ in range(3):
            tracker.update([])
        self.assertNotEqual(first, tracker.update(detection)[0].track_id)

    def test_collapsed_and_crossed_polygons_rejected(self):
        for polygon in ([[0,0],[1,1],[0,1],[1,0]], [[0,0],[.5,.5],[1,1]]):
            with self.assertRaises(ValueError):
                validate_polygon(polygon)
        validate_polygon([[0,0],[1,0],[1,1],[0,1]])

    def test_missing_frames_reset_opposite_direction_history(self):
        rule = WrongWayRule([(0,0),(100,0),(100,100),(0,100)], (1,0),
                            min_track_age_frames=3, min_motion_pixels=5, consecutive_frames=2)
        for frame, x in ((1,80),(2,70),(10,20)):
            rule.begin_frame(frame, {1})
            self.assertIsNone(rule.evaluate(TrackObservation(1,(x-5,30,x+5,50),'car',.9)))

    def test_short_occlusion_preserves_wrong_way_history(self):
        rule = WrongWayRule([(0,0),(100,0),(100,100),(0,100)], (1,0),
                            min_track_age_frames=3, min_motion_pixels=5, consecutive_frames=2,
                            max_occlusion_gap_frames=4)
        trigger = None
        for frame, x in ((1,80),(2,70),(5,60),(6,50)):
            rule.begin_frame(frame, {1})
            trigger = rule.evaluate(TrackObservation(1,(x-5,30,x+5,50),'car',.9)) or trigger
        self.assertIsNotNone(trigger)
        self.assertEqual(rule.short_occlusion_recoveries, 1)
        self.assertEqual(rule.long_gap_resets, 0)

    def test_conservative_tracklet_stitch_survives_id_change(self):
        rule = WrongWayRule(
            [(0, 0), (120, 0), (120, 100), (0, 100)],
            (1, 0),
            min_track_age_frames=4,
            min_motion_pixels=5,
            consecutive_frames=2,
            max_tracklet_stitch_gap_frames=3,
        )
        trigger = None
        for frame, track_id, x in ((1, 1, 100), (2, 1, 90), (3, 1, 80), (5, 2, 60), (6, 2, 50)):
            rule.begin_frame(frame, {track_id})
            observation = TrackObservation(track_id, (x - 6, 30, x + 6, 50), "car", 0.9)
            trigger = rule.evaluate(observation) or trigger
        self.assertIsNotNone(trigger)
        self.assertTrue(trigger.tracklet_stitched)
        self.assertEqual(trigger.previous_track_id, 1)
        self.assertEqual(rule.tracklet_stitches, 1)

    def test_tracklet_stitch_rejects_ambiguous_neighbours(self):
        rule = WrongWayRule(
            [(0, 0), (140, 0), (140, 100), (0, 100)],
            (1, 0),
            min_track_age_frames=6,
            min_motion_pixels=5,
            consecutive_frames=2,
            max_tracklet_stitch_gap_frames=3,
        )
        for frame, positions in ((1, ((1, 80), (2, 100))), (2, ((1, 80), (2, 100)))):
            rule.begin_frame(frame, {track_id for track_id, _x in positions})
            for track_id, x in positions:
                rule.evaluate(TrackObservation(track_id, (x - 6, 30, x + 6, 50), "car", 0.9))
        rule.begin_frame(4, {3})
        self.assertIsNone(rule.evaluate(TrackObservation(3, (84, 30, 96, 50), "car", 0.9)))
        self.assertEqual(rule.tracklet_stitches, 0)

    def test_event_review_and_evaluation_http(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary)
            event={"event_id":"test-event", "timestamp_seconds":5.7,
                   "candidate_event":"candidate_wrong_way_driving", "ground_point_normalized":[.3,.6]}
            events=root/'events.jsonl';events.write_text(json.dumps(event)+'\n', encoding='utf-8')
            state=AppState()
            state.jobs['test']={"status":"complete", "video_path":str(PROJECT_ROOT/'data/test.mp4'),
                "output_dir":str(root), "summary":{"outputs":{"events":str(events)},
                "candidate_events_by_type":{"candidate_wrong_way_driving":1}}}
            server=ThreadingHTTPServer(('127.0.0.1',0),make_handler(state))
            worker=threading.Thread(target=server.serve_forever,daemon=True);worker.start()
            base=f'http://127.0.0.1:{server.server_port}'
            try:
                with urlopen(base+'/api/events?id=test') as response:
                    self.assertEqual(json.load(response)['events'][0]['event_id'],'test-event')
                truth={"candidate_event":event['candidate_event'],"timestamp_seconds":5.7,
                       "vehicle_point_normalized":[.3,.6]}
                payload={"job_id":"test","video_path":"data/test.mp4","labels":[truth]}
                request=Request(base+'/api/evaluate',data=json.dumps(payload).encode(),headers={'Content-Type':'application/json'})
                with urlopen(request) as response:
                    self.assertEqual(json.load(response)['true_positives'],1)
                self.assertEqual(len(list(root.glob('evaluation-*.json'))),1)
                payload['video_path']='data/different.mp4'
                with self.assertRaises(HTTPError) as failure:
                    urlopen(Request(base+'/api/evaluate',data=json.dumps(payload).encode()))
                self.assertEqual(failure.exception.code,400)
            finally:
                server.shutdown();server.server_close();worker.join()

    def test_lane_suggestion_http(self):
        frame = np.zeros((360, 640, 3), dtype=np.uint8)
        cv2.line(frame, (160, 330), (280, 70), (255, 255, 255), 7)
        cv2.line(frame, (480, 330), (360, 70), (255, 255, 255), 7)
        ok, encoded = cv2.imencode('.jpg', frame)
        self.assertTrue(ok)
        payload = {
            'image_data_url': 'data:image/jpeg;base64,' + base64.b64encode(encoded).decode(),
            'direction': [[0.5, 0.92], [0.5, 0.18]],
        }
        server = ThreadingHTTPServer(('127.0.0.1', 0), make_handler(AppState()))
        worker = threading.Thread(target=server.serve_forever, daemon=True)
        worker.start()
        try:
            request = Request(
                f'http://127.0.0.1:{server.server_port}/api/suggest-lane',
                data=json.dumps(payload).encode(),
                headers={'Content-Type': 'application/json'},
            )
            with urlopen(request) as response:
                result = json.load(response)
            self.assertEqual(len(result['polygon']), 4)
            self.assertEqual(result['method'], 'road_mask_painted_marking_corridor_v3')
            self.assertEqual(result['confidence_label'], 'geometric_fit_score_not_detection_accuracy')
            self.assertTrue(result['review_required'])
        finally:
            server.shutdown();server.server_close();worker.join()


if __name__ == '__main__':
    unittest.main()
