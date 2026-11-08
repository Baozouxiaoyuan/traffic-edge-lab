from __future__ import annotations

import unittest

from traffic_edge.forward_lane import _select_pair


class ForwardLaneSelectionTests(unittest.TestCase):
    def test_selects_boundaries_that_bracket_direction(self) -> None:
        rows = [0.45, 0.55, 0.65, 0.75, 0.85, 0.95]
        lanes = [
            {"slot": 0, "confidence": 0.8, "points": [[0.38 - 0.08 * (y - 0.45), y] for y in rows]},
            {"slot": 1, "confidence": 0.8, "points": [[0.62 + 0.08 * (y - 0.45), y] for y in rows]},
            {"slot": 2, "confidence": 0.9, "points": [[0.82, y] for y in rows]},
        ]
        left, right, polygon, confidence = _select_pair(lanes, [[0.5, 0.93], [0.5, 0.46]])
        self.assertEqual({left["slot"], right["slot"]}, {0, 1})
        self.assertEqual(len(polygon), 4)
        self.assertGreater(confidence, 0.5)

    def test_abstains_when_no_pair_overlaps_direction_range(self) -> None:
        lanes = [
            {"slot": 0, "confidence": 0.9, "points": [[0.2, y] for y in [0.7, 0.75, 0.8, 0.85, 0.9]]},
            {"slot": 1, "confidence": 0.9, "points": [[0.3, y] for y in [0.7, 0.75, 0.8, 0.85, 0.9]]},
        ]
        with self.assertRaisesRegex(ValueError, "did not find two credible"):
            _select_pair(lanes, [[0.5, 0.95], [0.5, 0.45]])


if __name__ == "__main__":
    unittest.main()
