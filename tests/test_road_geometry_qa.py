from __future__ import annotations

import unittest

import numpy as np

from traffic_edge.road_understanding import assess_polygon_road_overlap


class RoadGeometryQualityTests(unittest.TestCase):
    polygon = [[0.1, 0.1], [0.9, 0.1], [0.9, 0.9], [0.1, 0.9]]

    def test_full_road_support_passes(self) -> None:
        result = assess_polygon_road_overlap(np.full((100, 100), 255, dtype=np.uint8), self.polygon)

        self.assertEqual(result["classification"], "pass")
        self.assertEqual(result["coverage"], 1.0)
        self.assertTrue(result["valid_polygon"])
        self.assertGreater(result["polygon_pixels"], 0)
        self.assertEqual(result["road_pixels_within_polygon"], result["polygon_pixels"])

    def test_borderline_road_support_requires_review(self) -> None:
        road_mask = np.zeros((100, 100), dtype=np.uint8)
        road_mask[:, :70] = 255

        result = assess_polygon_road_overlap(road_mask, self.polygon)

        self.assertEqual(result["classification"], "review")
        self.assertGreaterEqual(result["coverage"], result["thresholds"]["review"])
        self.assertLess(result["coverage"], result["thresholds"]["pass"])

    def test_weak_road_support_fails(self) -> None:
        road_mask = np.zeros((100, 100), dtype=np.uint8)
        road_mask[:, :40] = 255

        result = assess_polygon_road_overlap(road_mask, self.polygon)

        self.assertEqual(result["classification"], "fail")
        self.assertTrue(result["valid_polygon"])
        self.assertLess(result["coverage"], result["thresholds"]["review"])

    def test_invalid_polygons_return_safe_failure(self) -> None:
        road_mask = np.full((100, 100), 255, dtype=np.uint8)
        invalid_polygons = [
            [[0.1, 0.1], [0.9, 0.9]],
            [[0.0, 0.0], [1.0, 1.0], [0.0, 1.0], [1.0, 0.0]],
            [[0.0, 0.0], [1.1, 0.0], [0.0, 1.0]],
            [[0.0, 0.0], [float("nan"), 0.5], [0.0, 1.0]],
        ]

        for polygon in invalid_polygons:
            with self.subTest(polygon=polygon):
                result = assess_polygon_road_overlap(road_mask, polygon)
                self.assertEqual(result["classification"], "fail")
                self.assertFalse(result["valid_polygon"])
                self.assertEqual(result["coverage"], 0.0)
                self.assertTrue(result["reason"])

    def test_invalid_mask_returns_safe_failure(self) -> None:
        result = assess_polygon_road_overlap(np.zeros((10, 10, 3), dtype=np.uint8), self.polygon)

        self.assertEqual(result["classification"], "fail")
        self.assertFalse(result["valid_polygon"])
        self.assertIn("two-dimensional", result["reason"])

    def test_custom_thresholds_are_applied_and_reported(self) -> None:
        road_mask = np.zeros((100, 100), dtype=np.uint8)
        road_mask[:, :70] = 255

        result = assess_polygon_road_overlap(
            road_mask,
            self.polygon,
            pass_threshold=0.65,
            review_threshold=0.40,
        )

        self.assertEqual(result["classification"], "pass")
        self.assertEqual(result["thresholds"], {"pass": 0.65, "review": 0.4})


if __name__ == "__main__":
    unittest.main()
