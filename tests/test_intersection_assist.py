from __future__ import annotations

import unittest

import cv2
import numpy as np

from traffic_edge.intersection_assist import (
    Trajectory,
    cluster_trajectories,
    detect_stop_line_candidates,
    movement_proposals,
)


class IntersectionAssistTests(unittest.TestCase):
    def test_opposite_vehicle_flows_become_separate_movements(self) -> None:
        tracks = {
            1: [(0.30, 0.90 - index * 0.05, 0.05) for index in range(12)],
            2: [(0.34, 0.92 - index * 0.05, 0.05) for index in range(12)],
            3: [(0.70, 0.25 + index * 0.05, 0.05) for index in range(12)],
            4: [(0.74, 0.23 + index * 0.05, 0.05) for index in range(12)],
        }
        proposals = movement_proposals(tracks)
        self.assertEqual(len(proposals), 2)
        vertical_directions = [item["direction"][1][1] - item["direction"][0][1] for item in proposals]
        self.assertTrue(any(value < 0 for value in vertical_directions))
        self.assertTrue(any(value > 0 for value in vertical_directions))
        self.assertTrue(all(len(item["polygon"]) >= 3 for item in proposals))

    def test_parallel_lanes_can_be_split_by_lateral_offset(self) -> None:
        trajectories = [
            Trajectory(1, [(0.25, 0.9 - i * 0.07) for i in range(10)], [0.035] * 10),
            Trajectory(2, [(0.27, 0.9 - i * 0.07) for i in range(10)], [0.035] * 10),
            Trajectory(3, [(0.55, 0.9 - i * 0.07) for i in range(10)], [0.035] * 10),
            Trajectory(4, [(0.57, 0.9 - i * 0.07) for i in range(10)], [0.035] * 10),
        ]
        groups = cluster_trajectories(trajectories)
        self.assertEqual(sorted(len(group) for group in groups), [2, 2])

    def test_stop_line_candidate_is_perpendicular_to_flow(self) -> None:
        frame = np.zeros((400, 600, 3), dtype=np.uint8)
        cv2.line(frame, (160, 235), (440, 235), (255, 255, 255), 12)
        candidates = detect_stop_line_candidates(
            frame,
            [[0.5, 0.90], [0.5, 0.20]],
            [[0.20, 0.95], [0.25, 0.15], [0.75, 0.15], [0.80, 0.95]],
        )
        self.assertTrue(candidates)
        first, second = candidates[0]["points"]
        self.assertLess(abs(first[1] - second[1]), 0.03)


if __name__ == "__main__":
    unittest.main()
