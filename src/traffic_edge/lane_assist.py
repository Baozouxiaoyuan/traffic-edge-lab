from __future__ import annotations

from math import hypot

import cv2
import numpy as np

from .geometry import validate_polygon


def suggest_lane_corridor(
    frame: np.ndarray,
    direction_normalized: list[list[float]],
    road_mask: np.ndarray | None = None,
) -> dict:
    """Fit one lane corridor around a user-provided travel direction.

    Uses visible line evidence to propose boundaries for manual review.
    """
    if frame is None or frame.ndim != 3:
        raise ValueError("A colour setup frame is required")
    if len(direction_normalized) != 2:
        raise ValueError("Draw the lane travel direction before auto-fitting")
    height, width = frame.shape[:2]
    start = np.array(
        [float(direction_normalized[0][0]) * width, float(direction_normalized[0][1]) * height],
        dtype=float,
    )
    end = np.array(
        [float(direction_normalized[1][0]) * width, float(direction_normalized[1][1]) * height],
        dtype=float,
    )
    vector = end - start
    lane_length = float(np.linalg.norm(vector))
    if lane_length < max(30.0, 0.08 * hypot(width, height)):
        raise ValueError("Draw a longer lane direction before auto-fitting")
    along = vector / lane_length
    normal = np.array([-along[1], along[0]])

    grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    grey = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(grey)
    grey = cv2.GaussianBlur(grey, (5, 5), 0)
    # Painted lane markings are usually bright/low-saturation white or yellow.
    # Keep this evidence separate from generic edges so curbs, shadows and
    # crosswalks cannot dominate the geometric score merely by being strong.
    hls = cv2.cvtColor(frame, cv2.COLOR_BGR2HLS)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    white = ((hls[:, :, 1] >= 145) & (hls[:, :, 2] <= 135)).astype(np.uint8) * 255
    yellow = cv2.inRange(hsv, np.array([12, 55, 80]), np.array([42, 255, 255]))
    marking_mask = cv2.bitwise_or(white, yellow)
    marking_mask = cv2.morphologyEx(
        marking_mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8), iterations=1
    )
    generic_edges = cv2.Canny(grey, 55, 155)
    marking_edges = cv2.Canny(marking_mask, 35, 110)
    marking_neighbourhood = cv2.dilate(marking_mask, np.ones((7, 7), np.uint8))
    edges = cv2.bitwise_or(marking_edges, cv2.bitwise_and(generic_edges, marking_neighbourhood))
    if road_mask is not None:
        if road_mask.shape != grey.shape:
            road_mask = cv2.resize(road_mask, (width, height), interpolation=cv2.INTER_NEAREST)
        semantic_region = cv2.dilate((road_mask > 0).astype(np.uint8) * 255, np.ones((11, 11), np.uint8))
        edges = cv2.bitwise_and(edges, semantic_region)
        marking_mask = cv2.bitwise_and(marking_mask, semantic_region)
    lines = cv2.HoughLinesP(
        edges,
        1,
        np.pi / 180,
        threshold=max(24, round(min(width, height) * 0.045)),
        minLineLength=max(28, round(hypot(width, height) * 0.055)),
        maxLineGap=max(18, round(hypot(width, height) * 0.035)),
    )
    if lines is None:
        raise ValueError("No stable lane-like lines were found on this frame")

    candidates: list[dict] = []
    for raw in lines[:, 0, :]:
        first = np.array(raw[:2], dtype=float)
        second = np.array(raw[2:], dtype=float)
        segment = second - first
        segment_length = float(np.linalg.norm(segment))
        if segment_length == 0:
            continue
        parallel = abs(float(np.dot(segment / segment_length, along)))
        if parallel < 0.72:
            continue
        t1, t2 = float(np.dot(first - start, along)), float(np.dot(second - start, along))
        if abs(t2 - t1) < 20:
            continue
        s1, s2 = float(np.dot(first - start, normal)), float(np.dot(second - start, normal))
        slope = (s2 - s1) / (t2 - t1)
        intercept = s1 - slope * t1
        centre_offset = slope * (lane_length / 2) + intercept
        if not 0.025 * width <= abs(centre_offset) <= 0.48 * width:
            continue
        overlap = max(0.0, min(max(t1, t2), lane_length) - max(min(t1, t2), 0.0))
        support_canvas = np.zeros_like(marking_mask)
        cv2.line(support_canvas, tuple(raw[:2]), tuple(raw[2:]), 255, max(3, round(width * 0.006)))
        support_pixels = int(np.count_nonzero(support_canvas))
        marking_support = (
            float(np.count_nonzero((support_canvas > 0) & (marking_mask > 0))) / support_pixels
            if support_pixels else 0.0
        )
        score = (
            parallel * segment_length * (0.35 + overlap / lane_length)
            * (0.55 + 1.45 * marking_support)
            / (1 + abs(centre_offset) / width)
        )
        candidates.append(
            {
                "side": -1 if centre_offset < 0 else 1,
                "score": score,
                "slope": slope,
                "intercept": intercept,
                "line": raw.astype(float),
                "parallel": parallel,
                "overlap_ratio": overlap / lane_length,
                "marking_support": marking_support,
            }
        )

    left_candidates = [item for item in candidates if item["side"] == -1]
    right_candidates = [item for item in candidates if item["side"] == 1]
    if not left_candidates or not right_candidates:
        raise ValueError("Two lane boundaries could not be separated; use the drag corridor or a clearer frame")

    # Select a physically plausible pair, not merely the strongest line on each
    # side. This deliberately abstains when road edges, shadows or crosswalks do
    # not form a stable corridor around the user-provided centre direction.
    pair_choices = []
    for left in left_candidates:
        for right in right_candidates:
            left_offsets = [left["intercept"], left["slope"] * lane_length + left["intercept"]]
            right_offsets = [right["intercept"], right["slope"] * lane_length + right["intercept"]]
            if not all(value < -0.01 * width for value in left_offsets):
                continue
            if not all(value > 0.01 * width for value in right_offsets):
                continue
            corridor_widths = [right_offsets[index] - left_offsets[index] for index in range(2)]
            if min(corridor_widths) < 0.04 * width or max(corridor_widths) > 0.75 * width:
                continue
            width_stability = min(corridor_widths) / max(corridor_widths)
            if width_stability < 0.12:
                continue
            left_mid = abs(left["slope"] * lane_length / 2 + left["intercept"])
            right_mid = abs(right["slope"] * lane_length / 2 + right["intercept"])
            centre_balance = min(left_mid, right_mid) / max(left_mid, right_mid)
            coverage = min(left["overlap_ratio"], right["overlap_ratio"])
            parallel = min(left["parallel"], right["parallel"])
            marking_support = min(left["marking_support"], right["marking_support"])
            geometric_fit = (
                0.34 * parallel + 0.24 * coverage + 0.17 * centre_balance
                + 0.10 * width_stability + 0.15 * min(1.0, marking_support * 3)
            )
            pair_choices.append((geometric_fit * (left["score"] + right["score"]), geometric_fit, left, right))
    if not pair_choices:
        raise ValueError("Visible lines did not form a plausible non-crossing lane corridor; use manual drawing")
    _, geometric_fit, left, right = max(pair_choices, key=lambda item: item[0])
    if geometric_fit < 0.60:
        raise ValueError("Lane-line evidence was too weak or incomplete; use manual drawing")
    selected = {-1: left, 1: right}

    def boundary_point(item: dict, longitudinal: float) -> np.ndarray:
        offset = item["slope"] * longitudinal + item["intercept"]
        return start + along * longitudinal + normal * offset

    left_start = boundary_point(selected[-1], 0.0)
    left_end = boundary_point(selected[-1], lane_length)
    right_start = boundary_point(selected[1], 0.0)
    right_end = boundary_point(selected[1], lane_length)
    polygon_pixels = [left_start, left_end, right_end, right_start]
    polygon = [
        [round(float(np.clip(point[0] / width, 0, 1)), 6), round(float(np.clip(point[1] / height, 0, 1)), 6)]
        for point in polygon_pixels
    ]
    validate_polygon(polygon)
    semantic_road_overlap = None
    if road_mask is not None:
        polygon_mask = np.zeros((height, width), dtype=np.uint8)
        cv2.fillPoly(polygon_mask, [np.asarray(polygon_pixels, dtype=np.int32)], 255)
        polygon_pixels_count = int(np.count_nonzero(polygon_mask))
        semantic_road_overlap = (
            float(np.count_nonzero((polygon_mask > 0) & (road_mask > 0))) / polygon_pixels_count
            if polygon_pixels_count else 0.0
        )
        if semantic_road_overlap < 0.55:
            raise ValueError(
                f"The fitted corridor overlaps only {semantic_road_overlap:.0%} of the semantic road mask; draw it manually"
            )
    # Show the fitted boundaries that actually produced the polygon. Previously
    # the UI showed the shorter raw Hough segments, which could appear unrelated
    # to the red corridor after extrapolation.
    boundary_lines = [
        [polygon[0][0], polygon[0][1], polygon[1][0], polygon[1][1]],
        [polygon[3][0], polygon[3][1], polygon[2][0], polygon[2][1]],
    ]
    return {
        "polygon": polygon,
        "boundary_lines": boundary_lines,
        "confidence": round(float(geometric_fit), 3),
        "confidence_label": "geometric_fit_score_not_detection_accuracy",
        "review_required": True,
        "method": "road_mask_painted_marking_corridor_v3",
        "line_evidence": {
            "method": "clahe_colour_mask_canny_hough",
            "left_marking_support": round(float(left["marking_support"]), 3),
            "right_marking_support": round(float(right["marking_support"]), 3),
        },
        "semantic_road_overlap": round(semantic_road_overlap, 4) if semantic_road_overlap is not None else None,
        "warning": "Manual review is mandatory. This is geometric line fitting, not semantic road or lane recognition.",
    }
