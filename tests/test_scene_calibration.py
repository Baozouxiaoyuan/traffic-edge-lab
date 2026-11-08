from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import Mock

import cv2
import numpy as np

from traffic_edge.scene_calibration import merge_signal_detections, detect_signals, signal_reference_evidence
from traffic_edge.marking_curves import lane_regions, trace_markings
from traffic_edge.intersection_assist import detect_stop_line_candidates


class SceneCalibrationTests(unittest.TestCase):
    def test_readable_red_head_is_described_only_for_the_setup_frame(self):
        frame=np.zeros((200,200,3),np.uint8)
        cv2.circle(frame,(100,65),7,(0,0,255),-1)
        evidence=signal_reference_evidence(frame,[[.43,.25],[.57,.65]])
        self.assertEqual(evidence['reference_red_state'],'red')
        self.assertEqual(evidence['scope'],'frozen_setup_frame_only')
        self.assertTrue(evidence['detail_preview'].startswith('data:image/png;base64,'))

    def test_clipped_head_cannot_be_reported_as_readable_red(self):
        frame=np.zeros((200,200,3),np.uint8)
        cv2.circle(frame,(100,9),7,(0,0,255),-1)
        evidence=signal_reference_evidence(frame,[[.43,0],[.57,.4]])
        self.assertEqual(evidence['reference_red_state'],'unknown')
        self.assertEqual(evidence['reason'],'cropped_at_frame_edge')

    def test_enlarging_a_tiny_signal_does_not_invent_state_information(self):
        frame=np.zeros((200,200,3),np.uint8)
        frame[50:56,50:52]=(0,0,255)
        evidence=signal_reference_evidence(frame,[[.25,.25],[.26,.28]])
        self.assertEqual(evidence['reference_red_state'],'unknown')
        self.assertEqual(evidence['reason'],'too_few_pixels')

    def test_nearby_separate_signal_heads_are_not_merged(self):
        boxes = [
            {"rectangle": [[.40,.10],[.41,.14]], "confidence": .8, "frame_index": 0},
            {"rectangle": [[.42,.10],[.43,.14]], "confidence": .9, "frame_index": 0},
            {"rectangle": [[.4005,.10],[.4105,.14]], "confidence": .7, "frame_index": 1},
            {"rectangle": [[.40,.10],[.41,.14]], "confidence": .75, "frame_index": 0},
        ]
        merged = merge_signal_detections(boxes)
        self.assertEqual(len(merged), 2)
        self.assertEqual(sorted(item["observations"] for item in merged), [1,2])

    def test_tiled_detections_are_mapped_to_original_coordinates(self):
        model = Mock()
        model.names = {9:"traffic_light"}
        def result(frame, **kwargs):
            height,width = frame.shape[:2]
            # One detection in the bottom-right tile only.
            if frame[0,0,0] == 1:
                values = [[100,100,120,140]]
            else:
                values = []
            tensor = lambda values: SimpleNamespace(cpu=lambda:SimpleNamespace(tolist=lambda:values))
            return [SimpleNamespace(boxes=SimpleNamespace(xyxy=tensor(values),conf=tensor([.8] if values else []),cls=tensor([9] if values else [])))]
        model.predict.side_effect = result
        frame = np.zeros((600,1000,3),np.uint8)
        frame[228,380,0] = 1
        proposals,meta = detect_signals(model,[frame],640,[9])
        self.assertEqual(meta["views_processed"],5)
        self.assertEqual(len(proposals),1)
        np.testing.assert_allclose(proposals[0]["rectangle"],[[.48,328/600],[.50,368/600]],atol=1e-6)

    def test_visible_empty_parallel_lanes_are_found_without_tracks(self):
        labels = np.full((400,600),13,np.uint8)
        boundaries = [{"points":[[x,0],[x,.9975]],"role":"boundary_curve","length":399} for x in (.2,.4,.6)]
        for x in (120,240,360):
            cv2.line(labels,(x,0),(x,399),24,4)
        lanes = lane_regions(labels,boundaries,{})
        self.assertEqual(len(lanes),2)
        self.assertTrue(all(lane["direction"] == [] for lane in lanes))
        self.assertTrue(all(max(p[1] for p in lane["polygon"])-min(p[1] for p in lane["polygon"]) > .7 for lane in lanes))

    def test_transient_signal_is_rejected_and_duplicate_frames_do_not_add_votes(self):
        model=Mock();model.names={9:'traffic_light'}
        def predict(frame,**kwargs):
            values=[[50,50,60,80]] if frame[0,0,0]==1 else []
            tensor=lambda value:SimpleNamespace(cpu=lambda:SimpleNamespace(tolist=lambda:value))
            return [SimpleNamespace(boxes=SimpleNamespace(xyxy=tensor(values),conf=tensor([.8] if values else []),cls=tensor([9] if values else [])))]
        model.predict.side_effect=predict
        first=np.full((200,200,3),1,np.uint8);second=np.full((200,200,3),2,np.uint8)
        proposals,metadata=detect_signals(model,[first,first.copy(),second],640,[9])
        self.assertEqual(proposals,[])
        self.assertEqual(metadata['frames_sampled'],2)
        self.assertEqual(metadata['rejected_weak_candidates'],1)

    def test_parallel_edges_off_road_do_not_create_lanes(self):
        boundaries = [{"points":[[x,0],[x,.9975]],"role":"boundary_curve","length":399} for x in (.2,.4,.6)]
        self.assertEqual(lane_regions(np.zeros((400,600),np.uint8),boundaries,{}),[])

    def test_curved_marking_is_traced_without_staircase_fragmentation(self):
        labels=np.full((400,600),13,np.uint8)
        path=np.asarray([[90,390],[150,290],[260,190],[400,110]],np.int32)
        cv2.polylines(labels,[path],False,24,7)
        curves=trace_markings(labels,[])
        self.assertTrue(curves)
        self.assertGreater(curves[0]['length'],350)
        self.assertGreater(len(curves[0]['points']),2)

    def test_unseparated_road_touching_three_boundaries_is_not_one_lane(self):
        labels=np.full((400,600),13,np.uint8)
        boundaries=[{"points":[[x,0],[x,.9975]],"role":"boundary_curve","length":399} for x in (.2,.4,.6)]
        self.assertEqual(lane_regions(labels,boundaries,{}),[])

    def test_road_edges_without_paint_do_not_become_stop_lines(self):
        frame = np.zeros((400,600,3),np.uint8)
        frame[200:] = (80,80,80)
        candidates = detect_stop_line_candidates(frame,[[.5,.9],[.5,.1]])
        self.assertEqual(candidates,[])

    def test_crosswalk_excluded_by_semantic_marking_mask(self):
        frame = np.zeros((400,600,3),np.uint8)
        for y in (180,210,240):
            cv2.line(frame,(140,y),(450,y),(255,255,255),8)
        candidates = detect_stop_line_candidates(frame,[[.5,.9],[.5,.1]],marking_mask=np.zeros((400,600),np.uint8))
        self.assertEqual(candidates,[])

    def test_shared_transverse_line_is_clipped_to_each_lane(self):
        frame=np.zeros((400,600,3),np.uint8)
        cv2.line(frame,(50,200),(550,200),(255,255,255),10)
        for low,high in ((.1,.3),(.5,.7)):
            polygon=[[low,.1],[high,.1],[high,.9],[low,.9]]
            lines=detect_stop_line_candidates(frame,[[.2,.9],[.2,.1]],polygon,line_segments=[[50,200,550,200]])
            self.assertTrue(lines)
            for point in lines[0]["points"]:
                self.assertGreaterEqual(point[0],low-.003)
                self.assertLessEqual(point[0],high+.003)

    def test_stop_line_outside_inset_lane_boundary_can_be_proposed(self):
        frame=np.zeros((400,600,3),np.uint8)
        cv2.line(frame,(120,200),(480,200),(255,255,255),8)
        polygon=[[.2,.54],[.8,.54],[.8,.95],[.2,.95]]
        lines=detect_stop_line_candidates(frame,[[.5,.9],[.5,.54]],polygon,polygon_margin_pixels=22)
        self.assertTrue(lines)


if __name__ == "__main__":
    unittest.main()
