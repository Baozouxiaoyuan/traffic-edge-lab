"""Optional forward-view lane-instance inference for calibration assistance.

The model proposes visible lane-marking curves for manual calibration.
Its output does not configure the rule engine.
"""

from __future__ import annotations

import math
import time
from pathlib import Path

import cv2
import numpy as np


MODEL_INPUT_WIDTH = 800
MODEL_INPUT_HEIGHT = 288
GRIDING_NUM = 200
ROW_ANCHORS = np.asarray(
    [121, 131, 141, 150, 160, 170, 180, 189, 199, 209, 219, 228, 238, 248, 258, 267, 277, 287],
    dtype=np.float32,
)
MODEL_PATH = Path(__file__).resolve().parents[2] / "models" / "lane" / "culane_18.onnx"

_NETWORK: cv2.dnn.Net | None = None


def _softmax(values: np.ndarray, axis: int = 0) -> np.ndarray:
    shifted = values - np.max(values, axis=axis, keepdims=True)
    exponent = np.exp(shifted)
    return exponent / np.maximum(np.sum(exponent, axis=axis, keepdims=True), 1e-12)


def _network() -> cv2.dnn.Net:
    global _NETWORK
    if not MODEL_PATH.is_file():
        raise ValueError(
            "The experimental forward-view lane model is not installed. "
            "Expected models/lane/culane_18.onnx."
        )
    if _NETWORK is None:
        try:
            _NETWORK = cv2.dnn.readNetFromONNX(str(MODEL_PATH))
            _NETWORK.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
            _NETWORK.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        except cv2.error as exc:
            raise ValueError(f"The forward-view lane model could not be loaded: {exc}") from exc
    return _NETWORK


def _prepare(frame: np.ndarray) -> np.ndarray:
    resized = cv2.resize(frame, (MODEL_INPUT_WIDTH, MODEL_INPUT_HEIGHT), interpolation=cv2.INTER_AREA)
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    rgb = (rgb - np.asarray([0.485, 0.456, 0.406], dtype=np.float32)) / np.asarray(
        [0.229, 0.224, 0.225], dtype=np.float32
    )
    return np.transpose(rgb, (2, 0, 1))[None, ...]


def _decode(output: np.ndarray) -> list[dict]:
    logits = np.asarray(output)
    if logits.ndim == 4:
        logits = logits[0]
    if logits.shape != (GRIDING_NUM + 1, len(ROW_ANCHORS), 4):
        raise ValueError(f"Unexpected lane-model output shape: {tuple(logits.shape)}")

    logits = logits[:, ::-1, :]
    class_probabilities = _softmax(logits, axis=0)
    foreground = class_probabilities[:-1]
    expected = np.sum(foreground * np.arange(1, GRIDING_NUM + 1)[:, None, None], axis=0)
    winning_class = np.argmax(logits, axis=0)
    point_confidence = np.max(class_probabilities, axis=0)
    expected[winning_class == GRIDING_NUM] = 0

    lanes: list[dict] = []
    for lane_index in range(4):
        points: list[list[float]] = []
        confidences: list[float] = []
        for row_index in range(len(ROW_ANCHORS)):
            position = float(expected[row_index, lane_index])
            if position <= 0:
                continue
            anchor = float(ROW_ANCHORS[len(ROW_ANCHORS) - 1 - row_index])
            x = (position - 1.0) / (GRIDING_NUM - 1.0)
            y = anchor / MODEL_INPUT_HEIGHT
            if 0 <= x <= 1 and 0 <= y <= 1:
                points.append([round(x, 6), round(y, 6)])
                confidences.append(float(point_confidence[row_index, lane_index]))
        if len(points) >= 5:
            points.sort(key=lambda point: point[1])
            lanes.append(
                {
                    "slot": lane_index,
                    "points": points,
                    "point_count": len(points),
                    "confidence": round(float(np.mean(confidences)), 4),
                }
            )
    return lanes


def _x_at_y(lane: dict, y: float) -> float | None:
    points = lane["points"]
    # Allow a short endpoint extension because the fixed row anchors start at
    # y=121/288.  Anything longer would invent a boundary beyond model evidence.
    if not points or y < points[0][1] - 0.065 or y > points[-1][1] + 0.065:
        return None
    for first, second in zip(points, points[1:]):
        if first[1] <= y <= second[1]:
            span = second[1] - first[1]
            ratio = 0.0 if span == 0 else (y - first[1]) / span
            return first[0] + ratio * (second[0] - first[0])
    return points[0][0] if y <= points[0][1] else points[-1][0]


def _direction_x(direction: list[list[float]], y: float) -> float:
    first, second = direction
    span = second[1] - first[1]
    if abs(span) < 1e-6:
        return (first[0] + second[0]) / 2
    ratio = (y - first[1]) / span
    return first[0] + ratio * (second[0] - first[0])


def _select_pair(lanes: list[dict], direction: list[list[float]]) -> tuple[dict, dict, list[list[float]], float]:
    near_y = min(max(float(direction[0][1]), 0.48), 0.96)
    far_y = min(max(float(direction[1][1]), 0.43), near_y - 0.06)
    if direction[0][1] < direction[1][1]:
        near_y, far_y = min(max(float(direction[1][1]), 0.48), 0.96), min(max(float(direction[0][1]), 0.43), 0.90)
    if far_y >= near_y:
        far_y = near_y - 0.08

    target_near = _direction_x(direction, near_y)
    target_far = _direction_x(direction, far_y)
    best: tuple[float, dict, dict, list[list[float]]] | None = None
    for index, first in enumerate(lanes):
        for second in lanes[index + 1 :]:
            first_near, second_near = _x_at_y(first, near_y), _x_at_y(second, near_y)
            first_far, second_far = _x_at_y(first, far_y), _x_at_y(second, far_y)
            if None in {first_near, second_near, first_far, second_far}:
                continue
            left_near, right_near = sorted((float(first_near), float(second_near)))
            left_far, right_far = sorted((float(first_far), float(second_far)))
            near_width, far_width = right_near - left_near, right_far - left_far
            if not (0.035 <= near_width <= 0.55 and 0.018 <= far_width <= 0.38):
                continue
            center_error = abs((left_near + right_near) / 2 - target_near) + abs(
                (left_far + right_far) / 2 - target_far
            )
            bracket_penalty = 0.0
            if not left_near <= target_near <= right_near:
                bracket_penalty += 0.18
            if not left_far <= target_far <= right_far:
                bracket_penalty += 0.12
            confidence_bonus = (first["confidence"] + second["confidence"]) * 0.035
            score = center_error + bracket_penalty - confidence_bonus
            polygon = [
                [left_near, near_y],
                [left_far, far_y],
                [right_far, far_y],
                [right_near, near_y],
            ]
            if best is None or score < best[0]:
                best = (score, first, second, polygon)
    if best is None:
        raise ValueError(
            "The forward-view model did not find two credible lane boundaries around the legal-direction arrow. "
            "Keep the current polygon and fine-edit it manually."
        )
    score, first, second, polygon = best
    if score > 0.30:
        raise ValueError(
            "Lane lines were detected, but none reliably bracket the selected movement. "
            "The model abstained instead of replacing the monitored lane."
        )
    confidence = max(0.0, min(1.0, 1.0 - score / 0.30))
    return first, second, polygon, confidence


def suggest_forward_lane(frame: np.ndarray, direction: list[list[float]]) -> dict:
    if frame is None or frame.size == 0:
        raise ValueError("A valid setup frame is required")
    if not isinstance(direction, list) or len(direction) != 2:
        raise ValueError("Draw the legal-direction arrow before running the forward-view model")
    for point in direction:
        if not isinstance(point, list) or len(point) != 2 or not all(math.isfinite(float(v)) for v in point):
            raise ValueError("The legal-direction arrow is invalid")

    started = time.perf_counter()
    net = _network()
    net.setInput(_prepare(frame))
    lanes = _decode(net.forward())
    elapsed = time.perf_counter() - started
    if len(lanes) < 2:
        raise ValueError(
            "The forward-view model found fewer than two credible lane lines in this CCTV frame. "
            "It was trained for vehicle-mounted forward views, so no polygon was changed."
        )
    left, right, polygon, pair_confidence = _select_pair(lanes, direction)
    return {
        "method": "ufld_culane_forward_view_experimental",
        "model": "Ultra-Fast-Lane-Detection ResNet-18 (CULane)",
        "inference_seconds": round(elapsed, 4),
        "device": "cpu",
        "detected_lanes": lanes,
        "selected_slots": [left["slot"], right["slot"]],
        "selected_boundaries": [left["points"], right["points"]],
        "polygon": [[round(x, 6), round(y, 6)] for x, y in polygon],
        "confidence": round(pair_confidence, 4),
        "review_required": True,
        "domain_warning": (
            "CULane model applied to fixed CCTV. "
            "This proposal is not an accuracy claim and requires human confirmation."
        ),
    }
