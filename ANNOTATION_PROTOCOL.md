# Vehicle-level ground-truth annotation protocol

Updated: 4 November 2026

## Purpose

This protocol produces event-level ground truth independently of the rule output. Upstream text, model event predictions and candidate review times are navigation aids only and must never be copied into confirmed ground truth. Detector boxes may be used only as a vehicle-selection shortcut; the reviewer must inspect the frame and manually draw a box when a vehicle is missed.

## Annotation unit

One annotation represents one specific four-wheel vehicle committing one configured rule event in one configured lane. It contains:

- violation type;
- lane identity;
- exact event timestamp;
- vehicle bounding box;
- bottom-centre road-contact point;
- reviewer note and confirmation status.

## Event-time definitions

- **Red-light running:** the first frame in which the vehicle's road-contact point crosses the finite stop line while its controlling signal is red.
- **Wrong-way driving:** the first frame at which persistent opposite-direction travel is visually established inside the configured carriageway, not the first appearance of the vehicle.
- **Restricted-zone entry:** the first frame in which the vehicle's road-contact point enters the configured restricted polygon.

## Required procedure for every video

1. Confirm that the clip is one continuous fixed-camera shot. Reject or split edited montages.
2. Configure the physical lane, legal direction, finite stop line and controlling signal without looking at model events.
3. Watch the complete original video once at normal speed.
4. Use the green timeline coverage as an audit aid. At least 98% measured playback coverage is required by the application; the reviewer remains responsible for checking the complete clip.
5. Review every possible event with source-FPS-aware frame stepping and select the specific vehicle. Click a proposed vehicle box when correct, or draw a manual box when the detector misses it.
6. Mark ambiguous cases as `uncertain`; uncertain cases must be resolved or removed before finalization.
7. Watch the complete video a second time for missed events and normal negative traffic.
8. Run the automated annotation quality check and resolve duplicate, timestamp, bounding-box, uncertainty and lane/rule errors.
9. Enter the reviewer ID, then save completed positive labels or explicitly save a fully reviewed negative video.
10. Run the model only after the ground-truth review is complete. Switch to `Model error analysis` only after finalization; never label from the annotated result video.

## Quality control

- A second reviewer should independently review at least 20% of videos and every uncertain event.
- Report event agreement, timestamp disagreement and vehicle-selection disagreement before adjudication.
- Freeze labels, geometry, event-matching tolerance and test-camera membership before the final experiment.
- Development and validation cameras must not appear in the test split.
- Changes made after viewing test results require a new test split or must be reported as post-hoc analysis.

## Candidate-source warning

The AI City / PhysicalAI Traffic Anomaly Reasoning text is machine-generated and is used only to find potentially useful clips. During local inspection, detector proposals and upstream descriptions were not consistently precise enough to serve as ground truth. The browser therefore exposes their times as review hints and keeps all candidate cases blocked from the campaign until a complete independent review is saved.

## Completion rule

A case may change to `labelled` only when:

- the entire original clip has been reviewed;
- the recorded playback coverage is at least 98%, with no unresolved uncertain labels;
- positive events have vehicle-specific boxes and timestamps, or the video is explicitly confirmed negative;
- the automated quality gate reports no blocking errors;
- reviewer identity, reviewed intervals, video SHA-256 and source path are preserved in the review sidecar;
- camera geometry has been frozen by a completed analysis run;
- the web application has registered both files in `data/accuracy_campaign.active.json`.
