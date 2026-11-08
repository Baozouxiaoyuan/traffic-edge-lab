from __future__ import annotations

import unittest

from traffic_edge.models import TrackObservation
from traffic_edge.rules import DirectedStopLineRule, RegionEntryRule, _segments_intersect


APPROACH = [(0, 50), (100, 50), (100, 100), (0, 100)]


def observation(y: int, x: int = 50) -> TrackObservation:
    return TrackObservation(1, (x-10, y-20, x+10, y), 'car', .9)


def crossing_rule() -> DirectedStopLineRule:
    return DirectedStopLineRule(APPROACH, ((0, 50), (100, 50)), (50, 90), (0, -1),
                                'lane', 'candidate_red_signal_violation', 2, .75, 5)


class RuleContinuityTests(unittest.TestCase):
    def test_red_onset_during_crossing_is_not_proof_of_red_entry(self):
        rule = crossing_rule()
        self.assertIsNone(rule.evaluate(observation(80), False))
        self.assertIsNone(rule.evaluate(observation(60), False))
        self.assertIsNone(rule.evaluate(observation(45), True))

    def test_continuous_red_crossing_still_triggers(self):
        rule = crossing_rule()
        for frame, y in enumerate((80, 60, 45), 1):
            rule.begin_frame(frame, {1})
            result = rule.evaluate(observation(y), True)
        self.assertIsNotNone(result)

    def test_occluded_crossing_does_not_join_two_separate_observations(self):
        rule = crossing_rule()
        for frame, y in ((1, 80), (2, 60)):
            rule.begin_frame(frame, {1})
            rule.evaluate(observation(y), True)
        rule.begin_frame(3, set())
        rule.begin_frame(4, {1})
        self.assertIsNone(rule.evaluate(observation(45), True))

    def test_skipping_frames_without_empty_callback_also_breaks_continuity(self):
        rule = crossing_rule()
        rule.begin_frame(1, {1}); rule.evaluate(observation(80), True)
        rule.begin_frame(5, {1})
        self.assertIsNone(rule.evaluate(observation(45), True))

    def test_on_line_motion_after_red_onset_is_not_a_new_crossing(self):
        rule = crossing_rule()
        rule.evaluate(observation(80), False)
        rule.evaluate(observation(50), False)
        self.assertIsNone(rule.evaluate(observation(50), True))
        self.assertIsNone(rule.evaluate(observation(45), True))

    def test_region_entry_requires_observed_outside_to_inside_transition(self):
        rule = RegionEntryRule(APPROACH, 'zone', 'restricted', 2)
        rule.begin_frame(1, {1}); rule.evaluate(observation(40))
        rule.begin_frame(2, set())
        rule.begin_frame(3, {1})
        self.assertIsNone(rule.evaluate(observation(60)))
        rule.begin_frame(4, {1}); rule.evaluate(observation(40))
        rule.begin_frame(5, {1})
        self.assertIsNotNone(rule.evaluate(observation(60)))


    def test_finite_segments_do_not_intersect_outside_their_extents(self):
        self.assertFalse(_segments_intersect((110, 50), (130, 50), (0, 50), (100, 50)))
        self.assertFalse(_segments_intersect((50, 110), (50, 130), (50, 0), (50, 100)))
        self.assertTrue(_segments_intersect((90, 50), (110, 50), (0, 50), (100, 50)))
        self.assertTrue(_segments_intersect((50, 60), (50, 40), (0, 50), (100, 50)))
