from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TrackObservation:
    track_id: int
    bbox: tuple[int, int, int, int]
    class_name: str
    confidence: float

    @property
    def ground_point(self) -> tuple[float, float]:
        """Bottom-centre point approximates where the object touches the road."""
        x1, _y1, x2, y2 = self.bbox
        return ((x1 + x2) / 2.0, float(y2))
