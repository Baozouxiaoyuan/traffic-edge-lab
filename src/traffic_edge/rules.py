from __future__ import annotations

from dataclasses import dataclass
from collections import defaultdict, deque
from math import hypot

from .geometry import point_in_polygon
from .models import TrackObservation


@dataclass(frozen=True)
class RuleTrigger:
    track: TrackObservation
    rule_name: str
    region_name: str
    direction_alignment: float
    signal_is_red: bool
    observations_used: int = 0
    opposite_step_ratio: float | None = None
    tracklet_stitched: bool = False
    previous_track_id: int | None = None


class _ContinuousObservationRule:
    """Do not infer a boundary crossing through missing or unowned observations."""

    _continuity_maps: tuple[str, ...] = ()
    _continuity_sets: tuple[str, ...] = ()

    def begin_frame(self, frame_index: int, visible_ids: set[int]) -> None:
        previous_ids = getattr(self, '_previous_visible_ids', set())
        previous_frame = getattr(self, '_previous_frame_index', frame_index - 1)
        interrupted = previous_ids if frame_index != previous_frame + 1 else previous_ids - visible_ids
        for track_id in interrupted:
            self._reset_continuity(track_id)
        self._previous_visible_ids = set(visible_ids)
        self._previous_frame_index = frame_index

    def _reset_continuity(self, track_id: int) -> None:
        for name in self._continuity_maps:
            getattr(self, name).pop(track_id, None)
        for name in self._continuity_sets:
            getattr(self, name).discard(track_id)


class RegionEntryRule(_ContinuousObservationRule):
    """Trigger on a red signal when a mature track enters in the controlled direction."""

    _continuity_maps = ('_last_inside', '_track_age', '_first_point', '_started_inside')

    def __init__(
        self,
        polygon: list[tuple[int, int]],
        region_name: str,
        event_type: str,
        min_track_age_frames: int = 3,
        controlled_direction: tuple[float, float] | None = None,
        min_direction_alignment: float = 0.55,
        min_motion_pixels: float = 12.0,
        trigger_on_initial_presence: bool = False,
    ) -> None:
        self.polygon = polygon
        self.region_name = region_name
        self.event_type = event_type
        self.min_track_age_frames = min_track_age_frames
        self.controlled_direction = controlled_direction
        self.min_direction_alignment = min_direction_alignment
        self.min_motion_pixels = min_motion_pixels
        self.trigger_on_initial_presence = trigger_on_initial_presence
        self._last_inside: dict[int, bool] = {}
        self._track_age: dict[int, int] = {}
        self._first_point: dict[int, tuple[float, float]] = {}
        self._started_inside: dict[int, bool] = {}
        self._fired: set[int] = set()

    def evaluate(self, track: TrackObservation, signal_is_red: bool = True) -> RuleTrigger | None:
        track_id = track.track_id
        inside = point_in_polygon(track.ground_point, self.polygon)
        previous = self._last_inside.get(track_id)
        self._started_inside.setdefault(track_id, inside)
        age = self._track_age.get(track_id, 0) + 1
        self._track_age[track_id] = age
        self._first_point.setdefault(track_id, track.ground_point)
        self._last_inside[track_id] = inside

        entered = previous is False and inside
        persistent_initial_presence = (
            self.trigger_on_initial_presence
            and self._started_inside[track_id]
            and previous is True
            and inside
        )
        alignment = self._direction_alignment(track_id, track.ground_point)
        direction_matches = self.controlled_direction is None or (
            alignment is not None and alignment >= self.min_direction_alignment
        )
        if (entered or persistent_initial_presence) and signal_is_red and direction_matches and age >= self.min_track_age_frames and track_id not in self._fired:
            self._fired.add(track_id)
            return RuleTrigger(track, self.event_type, self.region_name, alignment or 0.0, signal_is_red,
                               observations_used=age)
        return None

    def _direction_alignment(self, track_id: int, current: tuple[float, float]) -> float | None:
        if self.controlled_direction is None:
            return 1.0
        start = self._first_point[track_id]
        motion = (current[0] - start[0], current[1] - start[1])
        motion_length = hypot(*motion)
        direction_length = hypot(*self.controlled_direction)
        if motion_length < self.min_motion_pixels or direction_length == 0:
            return None
        return (motion[0] * self.controlled_direction[0] + motion[1] * self.controlled_direction[1]) / (
            motion_length * direction_length
        )


def _side(point: tuple[float, float], line_start: tuple[float, float], line_end: tuple[float, float]) -> float:
    return (line_end[0] - line_start[0]) * (point[1] - line_start[1]) - (
        line_end[1] - line_start[1]
    ) * (point[0] - line_start[0])


def _segments_intersect(
    first_start: tuple[float, float],
    first_end: tuple[float, float],
    second_start: tuple[float, float],
    second_end: tuple[float, float],
) -> bool:
    # Orientation signs alone also accept disjoint collinear segments.
    for axis in (0, 1):
        if max(min(first_start[axis], first_end[axis]), min(second_start[axis], second_end[axis])) > min(
            max(first_start[axis], first_end[axis]), max(second_start[axis], second_end[axis])
        ):
            return False
    first_a = _side(first_start, second_start, second_end)
    first_b = _side(first_end, second_start, second_end)
    second_a = _side(second_start, first_start, first_end)
    second_b = _side(second_end, first_start, first_end)
    return first_a * first_b <= 0 and second_a * second_b <= 0


class DirectedStopLineRule(_ContinuousObservationRule):
    """Trigger when a red-light-controlled vehicle crosses a finite stop line from its approach."""

    _continuity_maps = ('_track_age', '_last_point', '_history', '_last_red')
    _continuity_sets = ('_seen_approach',)

    def __init__(
        self,
        approach_polygon: list[tuple[int, int]],
        stop_line: tuple[tuple[float, float], tuple[float, float]],
        direction_start: tuple[float, float],
        controlled_direction: tuple[float, float],
        region_name: str,
        event_type: str,
        min_track_age_frames: int = 5,
        min_direction_alignment: float = 0.75,
        min_motion_pixels: float = 12.0,
    ) -> None:
        self.approach_polygon = approach_polygon
        self.stop_line = stop_line
        self.direction_start = direction_start
        self.controlled_direction = controlled_direction
        self.region_name = region_name
        self.event_type = event_type
        self.min_track_age_frames = min_track_age_frames
        self.min_direction_alignment = min_direction_alignment
        self.min_motion_pixels = min_motion_pixels
        self._track_age: dict[int, int] = {}
        self._last_point: dict[int, tuple[float, float]] = {}
        self._last_red: dict[int, bool] = {}
        self._seen_approach: set[int] = set()
        self._history: dict[int, deque[tuple[float, float]]] = defaultdict(lambda: deque(maxlen=15))
        self._fired: set[int] = set()
        self._approach_side = _side(direction_start, *stop_line)
        if self._approach_side == 0:
            raise ValueError("The direction arrow must start away from the stop line")

    def evaluate(self, track: TrackObservation, signal_is_red: bool = True) -> RuleTrigger | None:
        track_id = track.track_id
        current = track.ground_point
        previous = self._last_point.get(track_id)
        previous_red = self._last_red.get(track_id, False)
        self._last_red[track_id] = signal_is_red
        self._last_point[track_id] = current
        self._history[track_id].append(current)
        age = self._track_age.get(track_id, 0) + 1
        self._track_age[track_id] = age
        if point_in_polygon(current, self.approach_polygon):
            self._seen_approach.add(track_id)
        if previous is None or track_id in self._fired:
            return None

        previous_side = _side(previous, *self.stop_line)
        current_side = _side(current, *self.stop_line)
        crossed_towards_intersection = (
            previous_side * self._approach_side > 0
            and current_side * self._approach_side <= 0
            and _segments_intersect(previous, current, *self.stop_line)
        )
        alignment = self._direction_alignment(track_id)
        direction_matches = alignment is not None and alignment >= self.min_direction_alignment
        if (
            crossed_towards_intersection
            and track_id in self._seen_approach
            and direction_matches
            and signal_is_red
            and previous_red
            and age >= self.min_track_age_frames
        ):
            self._fired.add(track_id)
            return RuleTrigger(track, self.event_type, self.region_name, alignment, signal_is_red,
                               observations_used=len(self._history[track_id]))
        return None

    def _direction_alignment(self, track_id: int) -> float | None:
        history = self._history[track_id]
        if len(history) < 2:
            return None
        motion = (history[-1][0] - history[0][0], history[-1][1] - history[0][1])
        motion_length = hypot(*motion)
        direction_length = hypot(*self.controlled_direction)
        if motion_length < self.min_motion_pixels or direction_length == 0:
            return None
        return (motion[0] * self.controlled_direction[0] + motion[1] * self.controlled_direction[1]) / (
            motion_length * direction_length
        )




class WrongWayRule:
    """Trigger once for stable, predominantly opposite motion inside one selected carriageway."""

    def __init__(
        self,
        polygon: list[tuple[int, int]],
        legal_direction: tuple[float, float],
        region_name: str = "controlled_approach_zone",
        event_type: str = "candidate_wrong_way_driving",
        min_track_age_frames: int = 5,
        maximum_alignment: float = -0.6,
        min_motion_pixels: float = 15.0,
        consecutive_frames: int = 3,
        minimum_opposite_step_ratio: float = 0.7,
        max_occlusion_gap_frames: int = 5,
        enable_tracklet_stitching: bool = True,
        max_tracklet_stitch_gap_frames: int = 12,
        max_stitch_distance_bbox_diagonals: float = 1.5,
        min_motion_bbox_height_ratio: float | None = None,
    ) -> None:
        self.polygon = polygon
        self.legal_direction = legal_direction
        self.region_name = region_name
        self.event_type = event_type
        self.min_track_age_frames = min_track_age_frames
        self.maximum_alignment = maximum_alignment
        self.min_motion_pixels = min_motion_pixels
        self.consecutive_frames = consecutive_frames
        self.minimum_opposite_step_ratio = minimum_opposite_step_ratio
        self.max_occlusion_gap_frames = max(1, max_occlusion_gap_frames)
        self.enable_tracklet_stitching = enable_tracklet_stitching
        self.max_tracklet_stitch_gap_frames = max(1, max_tracklet_stitch_gap_frames)
        self.max_stitch_distance_bbox_diagonals = max(0.5, max_stitch_distance_bbox_diagonals)
        self.min_motion_bbox_height_ratio = min_motion_bbox_height_ratio
        self.short_occlusion_recoveries = 0
        self.long_gap_resets = 0
        self.tracklet_stitches = 0
        self._last_seen_frame: dict[int, int] = {}
        self._frame_index = 0
        self._inside_age: dict[int, int] = {}
        self._history: dict[int, deque[tuple[float, float]]] = defaultdict(lambda: deque(maxlen=24))
        self._wrong_count: dict[int, int] = {}
        self._fired: set[int] = set()
        self._last_bbox: dict[int, tuple[int, int, int, int]] = {}
        self._class_name: dict[int, str] = {}
        self._height_history: dict[int, deque[float]] = defaultdict(lambda: deque(maxlen=24))
        self._stitch_source: dict[int, int] = {}
        self._visible_ids: set[int] = set()

    def evaluate(self, track: TrackObservation) -> RuleTrigger | None:
        track_id = track.track_id
        if not self._vehicle_is_inside(track):
            # Motion outside this carriageway must never influence its direction decision.
            self._inside_age[track_id] = 0
            self._history[track_id].clear()
            self._height_history[track_id].clear()
            self._wrong_count[track_id] = 0
            return None
        self._try_stitch(track)
        self._inside_age[track_id] = self._inside_age.get(track_id, 0) + 1
        self._history[track_id].append(track.ground_point)
        self._height_history[track_id].append(max(1.0, float(track.bbox[3] - track.bbox[1])))
        self._last_bbox[track_id] = track.bbox
        self._class_name[track_id] = track.class_name
        alignment = self._direction_alignment(track_id)
        opposite_ratio = self._opposite_step_ratio(track_id)
        is_wrong_way = (
            alignment is not None
            and alignment <= self.maximum_alignment
            and opposite_ratio >= self.minimum_opposite_step_ratio
        )
        self._wrong_count[track_id] = self._wrong_count.get(track_id, 0) + 1 if is_wrong_way else 0
        if (
            track_id not in self._fired
            and self._inside_age[track_id] >= self.min_track_age_frames
            and self._wrong_count[track_id] >= self.consecutive_frames
        ):
            self._fired.add(track_id)
            return RuleTrigger(
                track,
                self.event_type,
                self.region_name,
                alignment or -1.0,
                False,
                observations_used=len(self._history[track_id]),
                opposite_step_ratio=opposite_ratio,
                tracklet_stitched=track_id in self._stitch_source,
                previous_track_id=self._stitch_source.get(track_id),
            )
        return None

    def begin_frame(self, frame_index: int, visible_ids: set[int]) -> None:
        """Keep short occlusions but reset history after a long, unsafe observation gap."""
        self._frame_index = frame_index
        self._visible_ids = set(visible_ids)
        for track_id in visible_ids:
            previous = self._last_seen_frame.get(track_id)
            if previous is not None:
                gap = frame_index - previous - 1
                had_lane_history = bool(self._history[track_id]) or self._inside_age.get(track_id, 0) > 0
                if gap > self.max_occlusion_gap_frames:
                    self._history[track_id].clear()
                    self._height_history[track_id].clear()
                    self._inside_age[track_id] = 0
                    self._wrong_count[track_id] = 0
                    if had_lane_history:
                        self.long_gap_resets += 1
                elif gap > 0 and had_lane_history:
                    self.short_occlusion_recoveries += 1
            self._last_seen_frame[track_id] = frame_index

    def _try_stitch(self, track: TrackObservation) -> None:
        """Conservatively transfer one recent in-lane tracklet after an ID switch."""
        track_id = track.track_id
        if not self.enable_tracklet_stitching or self._history[track_id] or track_id in self._class_name:
            return
        x1, y1, x2, y2 = track.bbox
        area = max(1.0, float((x2 - x1) * (y2 - y1)))
        diagonal = hypot(x2 - x1, y2 - y1)
        candidates: list[tuple[float, int]] = []
        for old_id, history in self._history.items():
            if old_id == track_id or old_id in self._visible_ids or not history:
                continue
            last_seen = self._last_seen_frame.get(old_id)
            old_bbox = self._last_bbox.get(old_id)
            if last_seen is None or old_bbox is None or self._class_name.get(old_id) != track.class_name:
                continue
            elapsed = self._frame_index - last_seen
            if elapsed < 1 or elapsed > self.max_tracklet_stitch_gap_frames + 1:
                continue
            old_area = max(1.0, float((old_bbox[2] - old_bbox[0]) * (old_bbox[3] - old_bbox[1])))
            if not 0.5 <= area / old_area <= 2.0:
                continue
            points = list(history)
            velocity = (0.0, 0.0)
            if len(points) >= 2:
                recent = points[-4:]
                steps = list(zip(recent, recent[1:]))
                if steps:
                    velocity = (
                        sum(end[0] - start[0] for start, end in steps) / len(steps),
                        sum(end[1] - start[1] for start, end in steps) / len(steps),
                    )
            predicted = (points[-1][0] + velocity[0] * elapsed, points[-1][1] + velocity[1] * elapsed)
            distance = hypot(track.ground_point[0] - predicted[0], track.ground_point[1] - predicted[1])
            threshold = max(8.0, diagonal * self.max_stitch_distance_bbox_diagonals)
            if distance <= threshold:
                candidates.append((distance / threshold, old_id))
        candidates.sort()
        if not candidates or (len(candidates) > 1 and candidates[1][0] <= candidates[0][0] * 1.35):
            return
        _score, old_id = candidates[0]
        self._history[track_id] = deque(self._history[old_id], maxlen=24)
        self._height_history[track_id] = deque(self._height_history[old_id], maxlen=24)
        self._inside_age[track_id] = self._inside_age.get(old_id, 0)
        self._wrong_count[track_id] = self._wrong_count.get(old_id, 0)
        if old_id in self._fired:
            self._fired.add(track_id)
        self._history[old_id].clear()
        self._height_history[old_id].clear()
        self._inside_age[old_id] = 0
        self._wrong_count[old_id] = 0
        self._stitch_source[track_id] = old_id
        self.tracklet_stitches += 1

    def _vehicle_is_inside(self, track: TrackObservation) -> bool:
        """Require the road-contact centre and inset wheel points to remain in the lane polygon."""
        x1, _y1, x2, y2 = track.bbox
        width = x2 - x1
        contact_points = (
            track.ground_point,
            (x1 + width * 0.35, float(y2)),
            (x1 + width * 0.65, float(y2)),
        )
        return all(point_in_polygon(point, self.polygon) for point in contact_points)

    def _direction_alignment(self, track_id: int) -> float | None:
        history = self._history[track_id]
        if len(history) < 2:
            return None
        # Average the beginning and end windows to reduce detector-box jitter.
        window = max(1, min(4, len(history) // 3))
        start = (
            sum(point[0] for point in list(history)[:window]) / window,
            sum(point[1] for point in list(history)[:window]) / window,
        )
        end = (
            sum(point[0] for point in list(history)[-window:]) / window,
            sum(point[1] for point in list(history)[-window:]) / window,
        )
        motion = (end[0] - start[0], end[1] - start[1])
        motion_length = hypot(*motion)
        direction_length = hypot(*self.legal_direction)
        minimum_motion = self._minimum_motion(track_id)
        if motion_length < minimum_motion or direction_length == 0:
            return None
        return (motion[0] * self.legal_direction[0] + motion[1] * self.legal_direction[1]) / (
            motion_length * direction_length
        )

    def _minimum_motion(self, track_id: int) -> float:
        if self.min_motion_bbox_height_ratio is None or not self._height_history[track_id]:
            return self.min_motion_pixels
        heights = sorted(self._height_history[track_id])
        middle = len(heights) // 2
        median_height = heights[middle] if len(heights) % 2 else (heights[middle - 1] + heights[middle]) / 2.0
        return max(self.min_motion_pixels, median_height * self.min_motion_bbox_height_ratio)

    def _opposite_step_ratio(self, track_id: int) -> float:
        history = self._history[track_id]
        if len(history) < 3:
            return 0.0
        direction_length = hypot(*self.legal_direction)
        if direction_length == 0:
            return 0.0
        meaningful = 0
        opposite = 0
        jitter_floor = max(1.0, self.min_motion_pixels / 6.0)
        for previous, current in zip(history, list(history)[1:]):
            motion = (current[0] - previous[0], current[1] - previous[1])
            motion_length = hypot(*motion)
            if motion_length < jitter_floor:
                continue
            meaningful += 1
            alignment = (motion[0] * self.legal_direction[0] + motion[1] * self.legal_direction[1]) / (
                motion_length * direction_length
            )
            if alignment <= self.maximum_alignment:
                opposite += 1
        return opposite / meaningful if meaningful else 0.0
