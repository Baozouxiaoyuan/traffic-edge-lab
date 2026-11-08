from __future__ import annotations

import unittest
from unittest.mock import patch

import cv2
import numpy as np

from traffic_edge.scene_alignment import align_static_scene
from traffic_edge.scene_calibration import scene_evidence
from traffic_edge.intersection_assist import detect_stop_line_candidates


class SceneEvidenceTests(unittest.TestCase):
    def setUp(self):
        self.frame=np.random.default_rng(20261027).integers(0,256,(240,320,3),dtype=np.uint8)

    def test_background_matching_rejects_camera_pan(self):
        moved=cv2.warpAffine(self.frame,np.float32([[1,0,14],[0,1,0]]),(320,240))
        empty=np.zeros((240,320),np.uint8)
        transform,metadata=align_static_scene(self.frame,moved,empty,empty)
        self.assertIsNone(transform)
        self.assertFalse(metadata['accepted'])

    def test_small_camera_jitter_can_be_aligned(self):
        moved=cv2.warpAffine(self.frame,np.float32([[1,0,2],[0,1,0]]),(320,240))
        empty=np.zeros((240,320),np.uint8)
        transform,metadata=align_static_scene(self.frame,moved,empty,empty)
        self.assertTrue(metadata['accepted'])
        self.assertIsNotNone(transform)
        self.assertAlmostEqual(transform[0,2],2,delta=.3)

    def test_vehicle_roof_paint_predictions_do_not_become_road_markings(self):
        other=self.frame.copy();other[40:80,40:80]=255;other[40:80,130:170]=100
        first=np.full((240,320),13,np.uint8);first[50:60,50:60]=24
        second=np.full((240,320),13,np.uint8);second[50:60,140:150]=24
        with patch('traffic_edge.scene_calibration.segment_scene',side_effect=[(first,{}),(second,{})]):
            labels,metadata=scene_evidence(self.frame,[other],[[40,40,80,80]],[[[130,40,170,80]]])
        self.assertEqual(metadata['temporal_frames_used'],2)
        self.assertFalse(np.any(labels==24))
        self.assertEqual(labels[55,55],13)

    def test_jittered_paint_is_warped_back_to_the_reference_coordinates(self):
        moved=cv2.warpAffine(self.frame,np.float32([[1,0,2],[0,1,0]]),(320,240))
        first=np.full((240,320),13,np.uint8)
        second=first.copy();second[100:160,102:105]=24
        with patch('traffic_edge.scene_calibration.segment_scene',side_effect=[(first,{}),(second,{})]):
            labels,metadata=scene_evidence(self.frame,[moved],[],[[]])
        self.assertEqual(metadata['temporal_frames_used'],2)
        self.assertTrue(np.all(labels[110:150,100:103]==24))
        self.assertFalse(np.any(labels[:,104:]==24))

    def test_changed_camera_view_cannot_contribute_paint(self):
        moved=cv2.warpAffine(self.frame,np.float32([[1,0,14],[0,1,0]]),(320,240))
        first=np.full((240,320),13,np.uint8)
        with patch('traffic_edge.scene_calibration.segment_scene',return_value=(first,{})) as segment:
            labels,metadata=scene_evidence(self.frame,[moved],[],[[]])
        self.assertEqual(segment.call_count,1)
        self.assertEqual(metadata['unstable_frames_skipped'],1)
        self.assertFalse(np.any(labels==24))

    def test_paint_revealed_after_a_vehicle_moves_is_retained(self):
        other=self.frame.copy();other[40:80,40:80]=255
        first=np.full((240,320),13,np.uint8)
        second=first.copy();second[50:60,50:60]=24
        with patch('traffic_edge.scene_calibration.segment_scene',side_effect=[(first,{}),(second,{})]):
            labels,metadata=scene_evidence(self.frame,[other],[[40,40,80,80]],[[]])
        self.assertEqual(labels[55,55],24)
        self.assertEqual(metadata['unrecovered_foreground_pixels'],0)

    def test_persistently_occluded_ground_stays_unknown(self):
        other=self.frame.copy();other[40:80,40:80]=255
        first=np.full((240,320),13,np.uint8)
        second=first.copy();second[50:60,50:60]=24
        with patch('traffic_edge.scene_calibration.segment_scene',side_effect=[(first,{}),(second,{})]):
            labels,metadata=scene_evidence(self.frame,[other],[[40,40,80,80]],[[[40,40,80,80]]])
        self.assertEqual(labels[55,55],255)
        self.assertGreater(metadata['unrecovered_foreground_pixels'],0)

    def test_stop_line_never_bridges_disconnected_parts_of_a_concave_lane(self):
        frame=np.zeros((400,600,3),np.uint8);cv2.line(frame,(0,200),(599,200),(255,255,255),8)
        polygon=[[.1,.1],[.3,.1],[.3,.7],[.7,.7],[.7,.1],[.9,.1],[.9,.9],[.1,.9]]
        candidates=detect_stop_line_candidates(frame,[[.2,.9],[.2,.1]],polygon,line_segments=[[0,200,599,200]])
        self.assertTrue(candidates)
        for candidate in candidates:
            xs=[point[0] for point in candidate['points']]
            self.assertTrue(max(xs)<=.301 or min(xs)>=.699)

    def test_out_of_frame_stop_line_is_clipped_to_visible_image(self):
        frame=np.zeros((400,600,3),np.uint8)
        candidates=detect_stop_line_candidates(frame,[[.5,.9],[.5,.1]],line_segments=[[-100,200,700,200]])
        self.assertTrue(candidates)
        self.assertTrue(all(0<=value<=1 for point in candidates[0]['points'] for value in point))

    def test_zero_length_flow_does_not_create_stop_line_proposals(self):
        frame=np.zeros((400,600,3),np.uint8)
        self.assertEqual(detect_stop_line_candidates(frame,[[.5,.5],[.5,.5]],line_segments=[[0,200,599,200]]),[])


if __name__=='__main__':
    unittest.main()
