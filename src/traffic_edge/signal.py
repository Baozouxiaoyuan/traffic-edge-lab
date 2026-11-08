from __future__ import annotations

from collections import deque

import cv2
import numpy as np


class RedSignalDetector:
    """Conservatively estimate a red lamp using colour, shape and temporal evidence.

    A low-information or conflicting crop is reported as ``unknown`` and is never
    treated as red. Colour dominance and connected components help reject
    red housings, reflections and timestamps.
    """

    def __init__(
        self,
        rectangle: tuple[int, int, int, int],
        minimum_red_ratio: float = 0.015,
        history_size: int = 5,
        minimum_colour_ratio: float = 0.003,
        red_dominance_ratio: float = 1.35,
        minimum_blob_ratio: float = 0.001,
    ) -> None:
        self.rectangle = rectangle
        self.minimum_red_ratio = minimum_red_ratio
        self.minimum_colour_ratio = minimum_colour_ratio
        self.red_dominance_ratio = red_dominance_ratio
        self.minimum_blob_ratio = minimum_blob_ratio
        self.history: deque[str] = deque(maxlen=max(1, history_size))
        self.last_ratio = 0.0
        self.last_green_ratio = 0.0
        self.last_amber_ratio = 0.0
        self.last_confidence = 0.0
        self.last_state = "unknown"

    def update(self, frame: np.ndarray) -> bool:
        x1, y1, x2, y2 = self.rectangle
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            self._record("unknown", 0.0)
            return False
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        lower_red = cv2.inRange(hsv, (0, 90, 90), (12, 255, 255))
        upper_red = cv2.inRange(hsv, (168, 90, 90), (179, 255, 255))
        red_mask = cv2.bitwise_or(lower_red, upper_red)
        green_mask = cv2.inRange(hsv, (35, 70, 75), (95, 255, 255))
        amber_mask = cv2.inRange(hsv, (13, 80, 90), (34, 255, 255))
        area = float(crop.shape[0] * crop.shape[1])
        self.last_ratio = cv2.countNonZero(red_mask) / area
        self.last_green_ratio = cv2.countNonZero(green_mask) / area
        self.last_amber_ratio = self._amber_lamp_pixels(amber_mask) / area
        largest_blob_ratio = self._largest_blob_ratio(red_mask, area)
        strongest_other = max(self.last_green_ratio, self.last_amber_ratio)
        colour_total = self.last_ratio + self.last_green_ratio + self.last_amber_ratio
        red_confidence = self.last_ratio / max(colour_total, self.minimum_red_ratio)
        red_evidence = (
            self.last_ratio >= self.minimum_red_ratio
            and largest_blob_ratio >= self.minimum_blob_ratio
            and self.last_ratio >= strongest_other * self.red_dominance_ratio
        )
        non_red_evidence = (
            strongest_other >= self.minimum_colour_ratio
            and strongest_other >= self.last_ratio * self.red_dominance_ratio
        )
        state = "red" if red_evidence else "not_red" if non_red_evidence else "unknown"
        self._record(state, red_confidence if state == "red" else 1.0 - red_confidence)
        valid = [value for value in self.history if value != "unknown"]
        minimum_votes = max(1, (self.history.maxlen + 1) // 2)
        return (
            self.last_state == "red"
            and len(self.history) == self.history.maxlen
            and len(valid) >= minimum_votes
            and valid.count("red") > valid.count("not_red")
        )

    def _record(self, state: str, confidence: float) -> None:
        self.last_state = state
        self.last_confidence = max(0.0, min(1.0, float(confidence)))
        self.history.append(state)

    @staticmethod
    def _amber_lamp_pixels(mask: np.ndarray) -> int:
        # A yellow signal housing can leave a long amber strip along the crop edge.
        # Keep compact amber components so an illuminated lamp still competes with red.
        height, width = mask.shape
        _count, _labels, stats, _centroids = cv2.connectedComponentsWithStats(mask)
        pixels = 0
        for left, top, blob_width, blob_height, area in stats[1:]:
            horizontal_trim = (
                (top == 0 or top + blob_height == height)
                and blob_width >= 0.7 * width
                and blob_width >= 3 * blob_height
            )
            vertical_trim = (
                (left == 0 or left + blob_width == width)
                and blob_height >= 0.7 * height
                and blob_height >= 3 * blob_width
            )
            if not (horizontal_trim or vertical_trim):
                pixels += int(area)
        return pixels

    @staticmethod
    def _largest_blob_ratio(mask: np.ndarray, area: float) -> float:
        count, _labels, stats, _centroids = cv2.connectedComponentsWithStats(mask)
        if count <= 1:
            return 0.0
        return float(max(stats[index, cv2.CC_STAT_AREA] for index in range(1, count))) / area


def denormalize_rectangle(points: list[list[float]], width: int, height: int) -> tuple[int, int, int, int]:
    if len(points) != 2:
        raise ValueError("A signal rectangle requires two corner points")
    xs = [max(0, min(width - 1, round(float(point[0]) * width))) for point in points]
    ys = [max(0, min(height - 1, round(float(point[1]) * height))) for point in points]
    x1, x2 = sorted(xs)
    y1, y2 = sorted(ys)
    if x2 <= x1 or y2 <= y1:
        raise ValueError("The signal rectangle must have a positive area")
    return x1, y1, x2, y2
