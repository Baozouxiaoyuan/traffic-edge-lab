from __future__ import annotations

import base64
import hashlib
import threading
import time
from dataclasses import dataclass

import cv2
import numpy as np

from .geometry import validate_polygon


MODEL_ID = "nvidia/segformer-b0-finetuned-cityscapes-512-1024"
_MODEL_LOCK = threading.Lock()
_MODEL_COMPONENTS = None
_LAST_RESULT = None


# Classify polygon overlap with the predicted road mask. Borderline overlap
# requires review; weak overlap is rejected. These thresholds are not accuracy scores.
ROAD_OVERLAP_PASS_THRESHOLD = 0.85
ROAD_OVERLAP_REVIEW_THRESHOLD = 0.60


def assess_polygon_road_overlap(
    road_mask: np.ndarray,
    polygon_normalized: list[list[float]],
    *,
    pass_threshold: float = ROAD_OVERLAP_PASS_THRESHOLD,
    review_threshold: float = ROAD_OVERLAP_REVIEW_THRESHOLD,
) -> dict:
    """Assess whether normalized camera geometry lies on segmented road.

    The returned ``coverage`` is the fraction of pixels inside the polygon that
    are labelled as road. It is a geometry quality-control signal, not an
    estimate of lane-detection or violation-detection accuracy.

    Default overlap classifications:

    - ``pass``: coverage is at least 0.85;
    - ``review``: coverage is at least 0.60 but below 0.85;
    - ``fail``: coverage is below 0.60, or the mask/polygon is invalid.

    Invalid polygons return a structured ``fail`` result for display in the UI.
    """

    thresholds = {
        "pass": round(float(pass_threshold), 4),
        "review": round(float(review_threshold), 4),
    }

    def failed(reason: str) -> dict:
        return {
            "coverage": 0.0,
            "classification": "fail",
            "valid_polygon": False,
            "polygon_pixels": 0,
            "road_pixels_within_polygon": 0,
            "thresholds": thresholds,
            "reason": reason,
        }

    if not 0.0 <= review_threshold < pass_threshold <= 1.0:
        raise ValueError("Road-overlap thresholds must satisfy 0 <= review < pass <= 1")
    if not isinstance(road_mask, np.ndarray) or road_mask.ndim != 2 or road_mask.size == 0:
        return failed("A non-empty two-dimensional road mask is required.")
    height, width = road_mask.shape
    if height < 2 or width < 2:
        return failed("The road mask is too small for polygon quality assessment.")
    if not isinstance(polygon_normalized, (list, tuple)) or len(polygon_normalized) < 3:
        return failed("Draw at least three distinct polygon corners.")

    try:
        polygon = []
        for point in polygon_normalized:
            if not isinstance(point, (list, tuple)) or len(point) != 2:
                return failed("Every polygon corner must contain normalized x and y values.")
            x, y = float(point[0]), float(point[1])
            if not np.isfinite(x) or not np.isfinite(y) or not (0.0 <= x <= 1.0 and 0.0 <= y <= 1.0):
                return failed("Normalized polygon coordinates must be finite values from 0 to 1.")
            polygon.append([x, y])
        validate_polygon(polygon)
    except (TypeError, ValueError, OverflowError) as exc:
        return failed(str(exc) or "The polygon is invalid.")

    polygon_pixels_xy = np.asarray(
        [
            [round(point[0] * (width - 1)), round(point[1] * (height - 1))]
            for point in polygon
        ],
        dtype=np.int32,
    )
    polygon_mask = np.zeros((height, width), dtype=np.uint8)
    cv2.fillPoly(polygon_mask, [polygon_pixels_xy], 1)
    polygon_pixel_count = int(np.count_nonzero(polygon_mask))
    if polygon_pixel_count == 0:
        return failed("The polygon is too small after conversion to image pixels.")

    road_pixels = np.asarray(road_mask) > 0
    road_within_polygon = int(np.count_nonzero(road_pixels & (polygon_mask > 0)))
    coverage = road_within_polygon / polygon_pixel_count
    if coverage >= pass_threshold:
        classification = "pass"
        reason = "The polygon is strongly supported by the segmented road surface."
    elif coverage >= review_threshold:
        classification = "review"
        reason = "Road support is borderline; confirm the polygon manually before analysis."
    else:
        classification = "fail"
        reason = "Too much of the polygon lies outside the segmented road surface."

    return {
        "coverage": round(float(coverage), 4),
        "classification": classification,
        "valid_polygon": True,
        "polygon_pixels": polygon_pixel_count,
        "road_pixels_within_polygon": road_within_polygon,
        "thresholds": thresholds,
        "reason": reason,
    }


@dataclass
class RoadUnderstandingResult:
    mask: np.ndarray
    mean_road_confidence: float
    road_fraction: float
    inference_seconds: float
    model_id: str
    device: str

    def metadata(self) -> dict:
        return {
            "model": self.model_id,
            "task": "Cityscapes semantic road segmentation",
            "device": self.device,
            "mean_road_confidence": round(self.mean_road_confidence, 4),
            "road_fraction": round(self.road_fraction, 4),
            "inference_seconds": round(self.inference_seconds, 3),
            "review_required": True,
            "limitations": "Pre-trained urban-scene segmentation can fail under unusual CCTV viewpoints, weather, glare or domain shift.",
        }


def _load_model():
    global _MODEL_COMPONENTS
    if _MODEL_COMPONENTS is not None:
        return _MODEL_COMPONENTS
    from transformers import AutoImageProcessor, SegformerForSemanticSegmentation

    import torch

    processor = AutoImageProcessor.from_pretrained(MODEL_ID, use_fast=True)
    model = SegformerForSemanticSegmentation.from_pretrained(MODEL_ID)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()
    labels = {int(index): str(name).lower() for index, name in model.config.id2label.items()}
    road_ids = [index for index, name in labels.items() if name == "road"]
    if not road_ids:
        raise RuntimeError("The selected semantic model does not expose a road class")
    _MODEL_COMPONENTS = (processor, model, road_ids, device)
    return _MODEL_COMPONENTS


def segment_road(frame: np.ndarray) -> RoadUnderstandingResult:
    """Segment road pixels in one calibration frame using SegFormer-B0 Cityscapes."""
    global _LAST_RESULT
    if frame is None or frame.ndim != 3:
        raise ValueError("A colour setup frame is required for road understanding")
    digest = hashlib.sha256(frame.tobytes()).hexdigest()
    with _MODEL_LOCK:
        if _LAST_RESULT is not None and _LAST_RESULT[0] == digest:
            return _LAST_RESULT[1]
        try:
            import torch
            from PIL import Image
        except ImportError as exc:
            raise RuntimeError("Semantic road understanding dependencies are not installed") from exc
        processor, model, road_ids, device = _load_model()
        started = time.perf_counter()
        image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        inputs = {name: value.to(device) for name, value in processor(images=image, return_tensors="pt").items()}
        with torch.inference_mode():
            logits = model(**inputs).logits
            logits = torch.nn.functional.interpolate(
                logits,
                size=frame.shape[:2],
                mode="bilinear",
                align_corners=False,
            )
            probabilities = logits.softmax(dim=1)[0]
            labels = probabilities.argmax(dim=0)
            road_probability = probabilities[road_ids].sum(dim=0)
            road_mask = torch.zeros_like(labels, dtype=torch.bool)
            for road_id in road_ids:
                road_mask |= labels == road_id
        mask = road_mask.cpu().numpy().astype(np.uint8) * 255
        road_values = road_probability[road_mask]
        result = RoadUnderstandingResult(
            mask=mask,
            mean_road_confidence=float(road_values.mean().item()) if road_values.numel() else 0.0,
            road_fraction=float(road_mask.float().mean().item()),
            inference_seconds=time.perf_counter() - started,
            model_id=MODEL_ID,
            device=device,
        )
        _LAST_RESULT = (digest, result)
        return result


def road_overlay_data_url(mask: np.ndarray, frame: np.ndarray | None = None) -> str:
    """Return a review image with the predicted road region tinted green."""
    if frame is None:
        overlay = np.zeros((mask.shape[0], mask.shape[1], 4), dtype=np.uint8)
        overlay[mask > 0] = (60, 200, 90, 110)  # BGRA
    else:
        overlay = frame.copy()
        road_pixels = mask > 0
        tint = np.zeros_like(frame)
        tint[:] = (60, 200, 90)
        overlay[road_pixels] = cv2.addWeighted(frame[road_pixels], 0.55, tint[road_pixels], 0.45, 0)
    ok, encoded = cv2.imencode(".png", overlay)
    if not ok:
        raise RuntimeError("Could not encode the semantic road overlay")
    return "data:image/png;base64," + base64.b64encode(encoded.tobytes()).decode("ascii")
