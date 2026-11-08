from __future__ import annotations

from abc import ABC, abstractmethod
from math import hypot
import os
from pathlib import Path

import cv2
import numpy as np

from .models import TrackObservation
from .compute import resolve_device


class FallbackDetectionTracker:
    """Assign short-lived IDs when ByteTrack has detections but has not confirmed a track yet."""

    def __init__(self, max_missed_frames: int = 8) -> None:
        self.max_missed_frames = max_missed_frames
        self._next_id = 1_000_000
        self._tracks: dict[int, dict[str, object]] = {}

    def update(
        self,
        detections: list[tuple[tuple[int, int, int, int], str, float]],
    ) -> list[TrackObservation]:
        for state in self._tracks.values():
            state["missed"] = int(state["missed"]) + 1
        self._tracks = {key: state for key, state in self._tracks.items()
                        if int(state["missed"]) <= self.max_missed_frames}
        available = set(self._tracks)
        observations: list[TrackObservation] = []
        for bbox, class_name, confidence in sorted(detections, key=lambda item: item[2], reverse=True):
            x1, _y1, x2, y2 = bbox
            ground = ((x1 + x2) / 2.0, float(y2))
            diagonal = hypot(x2 - x1, y2 - _y1)
            candidates = []
            for track_id in available:
                old_area = float(self._tracks[track_id]["area"])
                area = max(1, (x2 - x1) * (y2 - _y1))
                if not 0.2 <= area / old_area <= 5.0:
                    continue
                previous = self._tracks[track_id]["ground"]
                distance = hypot(ground[0] - previous[0], ground[1] - previous[1])
                candidates.append((distance, track_id))
            distance, track_id = min(candidates, default=(float("inf"), -1))
            if distance > max(30.0, diagonal * 2.5):
                track_id = self._next_id
                self._next_id += 1
            else:
                available.remove(track_id)
            self._tracks[track_id] = {"ground": ground, "missed": 0,
                                      "area": max(1, (x2 - x1) * (y2 - _y1))}
            observations.append(TrackObservation(track_id, bbox, class_name, confidence))
        self._tracks = {
            track_id: state
            for track_id, state in self._tracks.items()
            if int(state["missed"]) <= self.max_missed_frames
        }
        return observations


class TrackingBackend(ABC):
    @abstractmethod
    def track(self, frame: np.ndarray) -> list[TrackObservation]:
        raise NotImplementedError


class SyntheticColourTracker(TrackingBackend):
    """Small deterministic backend used only to verify the end-to-end demo."""

    def __init__(self, max_distance: float = 80.0) -> None:
        self.max_distance = max_distance
        self._next_id = 1
        self._centres: dict[int, tuple[float, float]] = {}

    def track(self, frame: np.ndarray) -> list[TrackObservation]:
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, np.array([0, 100, 80]), np.array([179, 255, 255]))
        count, _labels, stats, centroids = cv2.connectedComponentsWithStats(mask)
        detections: list[tuple[tuple[int, int, int, int], tuple[float, float]]] = []
        for component in range(1, count):
            x, y, width, height, area = stats[component]
            if area >= 350:
                detections.append(((int(x), int(y), int(x + width), int(y + height)), tuple(centroids[component])))

        unmatched_ids = set(self._centres)
        assigned: list[tuple[int, tuple[int, int, int, int], tuple[float, float]]] = []
        for bbox, centre in sorted(detections, key=lambda item: item[0][0]):
            candidates = [
                (hypot(centre[0] - self._centres[track_id][0], centre[1] - self._centres[track_id][1]), track_id)
                for track_id in unmatched_ids
            ]
            distance, track_id = min(candidates, default=(float("inf"), -1))
            if distance > self.max_distance:
                track_id = self._next_id
                self._next_id += 1
            else:
                unmatched_ids.remove(track_id)
            assigned.append((track_id, bbox, centre))

        self._centres = {track_id: centre for track_id, _bbox, centre in assigned}
        return [TrackObservation(track_id, bbox, "synthetic_vehicle", 1.0) for track_id, bbox, _centre in assigned]


class UltralyticsByteTrackBackend(TrackingBackend):
    def __init__(
        self,
        model_name: str,
        confidence: float,
        iou: float,
        vehicle_classes: list[int],
        image_size: int = 960,
        tracker_name: str = "bytetrack",
        device: str = "auto",
    ) -> None:
        project_root = Path(__file__).resolve().parents[2]
        config_root = project_root / ".ultralytics-config"
        config_root.mkdir(exist_ok=True)
        os.environ.setdefault("YOLO_CONFIG_DIR", str(config_root))
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise RuntimeError("Install requirements before using the yolo backend") from exc
        self.device = resolve_device(device)
        self.model = YOLO(model_name)
        self.confidence = confidence
        self.iou = iou
        self.vehicle_classes = vehicle_classes
        self.image_size = image_size
        tracker_name = tracker_name.lower().strip()
        if tracker_name not in {"bytetrack", "botsort"}:
            raise ValueError("Tracker must be 'bytetrack' or 'botsort'")
        self.tracker_name = tracker_name
        self.tracker_config = (
            project_root / "config" / "bytetrack_small_objects.yaml"
            if tracker_name == "bytetrack"
            else "botsort.yaml"
        )
        self.fallback_tracker = FallbackDetectionTracker()
        self.raw_detections = None
        self.raw_detection_status = "not_recorded"
        self._diagnostics_enabled = False

    def enable_diagnostics(self) -> None:
        if not self._diagnostics_enabled:
            # Register before the first track() call, which installs the tracker callback.
            self.model.add_callback("on_predict_postprocess_end", self._capture_detections)
            self._diagnostics_enabled = True

    def _capture_detections(self, predictor) -> None:
        boxes = predictor.results[0].boxes
        if boxes is not None and boxes.is_track:
            self.raw_detection_status = "callback_after_association"
            return
        self.raw_detection_status = "before_association"
        self.raw_detections = []
        if boxes is not None:
            for box, score, label in zip(boxes.xyxy.cpu().tolist(), boxes.conf.cpu().tolist(), boxes.cls.cpu().tolist()):
                self.raw_detections.append({"bbox": box, "confidence": score, "class_id": int(label)})

    def track(self, frame: np.ndarray) -> list[TrackObservation]:
        self.raw_detections = None
        self.raw_detection_status = "not_recorded"
        result = self.model.track(
            frame,
            persist=True,
            tracker=str(self.tracker_config),
            conf=self.confidence,
            iou=self.iou,
            classes=self.vehicle_classes,
            imgsz=self.image_size,
            agnostic_nms=True,
            verbose=False,
            device=self.device,
        )[0]
        boxes = result.boxes
        if boxes is None:
            self.fallback_tracker.update([])
            return []
        coordinates = boxes.xyxy.int().cpu().tolist()
        confidences = boxes.conf.cpu().tolist()
        class_ids = boxes.cls.int().cpu().tolist()
        names = result.names
        if boxes.id is None:
            detections = [
                (
                    tuple(int(value) for value in bbox),
                    str(names[class_id]),
                    float(confidence),
                )
                for bbox, confidence, class_id in zip(coordinates, confidences, class_ids)
            ]
            return self.fallback_tracker.update(detections)

        ids = boxes.id.int().cpu().tolist()
        self.fallback_tracker.update([])
        return [
            TrackObservation(
                track_id=int(track_id),
                bbox=tuple(int(value) for value in bbox),
                class_name=str(names[class_id]),
                confidence=float(confidence),
            )
            for track_id, bbox, confidence, class_id in zip(ids, coordinates, confidences, class_ids)
        ]
