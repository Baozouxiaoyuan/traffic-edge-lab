# Multi-video evaluation plan

## Traffic-rule development and evaluation

Primary source: *Dataset for Traffic Evaluation*, Tiago Silva, Instituto Politécnico de Lisboa.

- DOI: https://doi.org/10.5281/zenodo.14041920
- Intended use stated by the record: monitored-road video for traffic analysis and red-light-running model testing.
- Current local development files include `zenodo_video3.mp4`, `real_examples/zenodo_video4.mp4`, `real_examples/zenodo_red_light_event.webm` and `real_examples/zenodo_rlr_occasions.mp4`.
- These clips are development or demonstration material until complete vehicle-level labels exist. Repeatedly used examples must not be recycled into the final held-out accuracy result.

Each evaluation video requires its own reviewed and frozen configuration. Geometry from one viewpoint must not be copied blindly to another. For each video, record:

1. controlled approach polygon;
2. finite stop line;
3. controlling traffic-light rectangle;
4. legal movement direction;
5. restricted-zone polygon;
6. no-stop stationarity thresholds;
7. manual vehicle-specific labels.

Use one clip for configuration/tuning and different clips for final testing. Report precision, recall and F1 separately for every event type and provide a macro average only as a secondary summary.

## Final scope boundary

Accident detection, motorcycles and fisheye-camera validation are not part of the submitted system. They require different labels and evaluation protocols, and were removed from the final package rather than being presented as unfinished features. The retained dataset plan covers the four implemented rules only.

## Next acquisition: diversity and occlusion

The current built-in gallery is intentionally limited to locally available real clips whose provenance is already recorded. The following sources are acquisition candidates, not bundled project assets:

- UA-DETRAC: 100 real-world traffic videos and more than 140,000 annotated frames, including vehicle boxes and occlusion/truncation attributes. Use it to measure detection and tracking robustness under partial obstruction rather than to validate traffic-light rules.
  - Project page: https://sites.google.com/view/daweidu/projects/ua-detrac
  - Paper: https://arxiv.org/abs/1511.04136
- WTS: more than 1,200 staged traffic-event videos with fixed-overhead and ego views. Access requires following the repository's request process, so it is not currently bundled.
  - Repository: https://github.com/woven-visionai/wts-dataset

Before adding a new video to the gallery, record its licence, original URL, checksum, camera viewpoint, target rule, expected event, known confounders and a camera-specific frozen configuration. A new clip earns a validation badge only after vehicle-specific manual labels and a reproducible run; otherwise it must be marked development or review required.

## Acquisition update: AI City Challenge 2026 shortlist

The project now includes a deliberately small local shortlist from NVIDIA's PhysicalAI Traffic Anomaly Reasoning release for AI City Challenge 2026 Track 3. The official release contains 3,670 real CCTV transportation videos from eight upstream datasets and supplies download tooling rather than redistributing all source videos. Its annotation files are CC BY 4.0, while each source video retains its upstream terms.

- Official dataset card: https://huggingface.co/datasets/nvidia/PhysicalAI-Traffic-Anomaly-Reasoning
- Official challenge description: https://www.aicitychallenge.org/2026-track3/
- Local candidate register: `data/evaluation_candidates.json`
- Final local candidate register and provenance: `data/evaluation_candidates.json`

Seven real candidate files were originally screened. The final package retains only the four fixed, normal-perspective intersection clips accepted for vehicle-level red-light review. The motorway news compilation and two edited compilations were removed because camera changes make them invalid as continuous fixed-camera evaluation inputs.

The AI City training annotations are machine-generated. They are useful for locating candidate footage, but they are not treated as capstone ground truth. Only complete human labels produced from the original video can move a candidate into `data/evaluation_study.template.json` with `status: labelled`.
