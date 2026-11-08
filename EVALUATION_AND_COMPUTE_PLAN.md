# Accuracy, compute and real-time validation plan

## Current evidence status

The prototype has functional regressions, not a statistically valid accuracy result. The NJIT clip has been used during development and must not be reused as a held-out test. An unlabelled zero-event run is not proof that no violation occurred.

## Primary research endpoint

The primary endpoint is vehicle-specific event detection on held-out real video that was not used for development. A prediction matches ground truth only when all available fields agree:

1. candidate-event type;
2. physical lane identity;
3. timestamp within the frozen 0.5-second tolerance;
4. normalized vehicle position within the frozen 0.12 tolerance.

Report, for every rule and overall:

- true positives, false positives and false negatives;
- precision, recall and F1;
- false positives per video hour;
- video-level bootstrap 95% confidence intervals;
- results by camera, day/night, occlusion and vehicle-size strata;
- mean detection latency when an event has a physical onset time.

Detection quality should be reported separately with mAP50 and mAP50-95 when frame-level vehicle boxes are available. Tracking quality should be reported separately with HOTA, IDF1 and ID switches when full trajectories are available. Event F1 remains the capstone's primary measure because accurate boxes alone do not prove accurate traffic-rule decisions.

## Dataset protocol

- Split by camera, never by adjacent video frames.
- Freeze test videos and thresholds before looking at test results.
- Label every eligible positive event, not only model detections.
- Include long normal-traffic intervals so false positives per hour is meaningful.
- Record uncertain labels but exclude them from the primary endpoint until adjudicated.
- Double-review at least 20% of labels and report agreement or adjudication counts.
- Data target: add as many varied, fully labelled positive and negative videos as practical, and report the exact clips, duration and event counts for each rule. If coverage is limited, treat well-supported rules as primary and the others as exploratory extensions.

The campaign manifest is `data/accuracy_campaign.template.json`. Every case must be fully labelled and assigned to either `validation` or `test`. Cameras may not appear in both splits.

Run the controlled campaign with:

```powershell
$env:PYTHONPATH = "$PWD\src"
.\.venv\Scripts\python.exe -m traffic_edge.cli campaign `
  --manifest .\data\accuracy_campaign.json `
  --output-dir .\runs\accuracy-campaign
```

The protocol has two stages. Stage A runs every declared detector/tracker variant on validation videos only and selects by event-level micro F1, then recall, then real-time factor. It freezes the selected variant, manifest hash, model hash and all video/configuration/ground-truth hashes in `selected_variant.json`. Stage B runs only that frozen variant once on held-out test videos. No multi-variant test leaderboard is produced.

Confidence intervals resample whole cameras rather than individual clips, so several clips from one camera are not treated as independent evidence.

## Real-time acceptance criteria

Real time means end-to-end processing FPS is at least the source FPS. A deployable target uses 25% headroom:

- 15 FPS source: at least 18.75 processing FPS;
- 30 FPS source: at least 37.5 processing FPS;
- report average milliseconds per frame, real-time factor, dropped-frame rate and end-to-end alert latency.

The current Windows environment uses CPU-only PyTorch even though an AMD RX 6900 XT is installed. Measured baselines are:

| Run | Source FPS | Processing FPS | Real-time factor | Result |
|---|---:|---:|---:|---|
| NJIT, 1280 inference | 15 | 13.881 | 0.925 | Near real time, but below source rate |
| Zenodo video 4, 960 inference | 30 | 21.416 | 0.714 | Not real time at 30 FPS |

## Compute target

YOLOv8n has approximately 3.2 million parameters and 8.7 GFLOPs at 640-pixel input. Approximate model arithmetic, excluding decode, tracking, rules and encoding, is:

| Input size | Approximate GFLOPs/frame | At 30 FPS |
|---|---:|---:|
| 640 | 8.7 | 0.26 TFLOP/s |
| 960 | 19.6 | 0.59 TFLOP/s |
| 1280 | 34.8 | 1.04 TFLOP/s |

Use measured end-to-end FPS rather than theoretical TOPS for the final device decision.

Recommended benchmark targets:

- Jetson Orin Nano Super 8GB with TensorRT: single-camera edge baseline;
- Jetson Orin NX 16GB: higher-resolution or multi-camera edge option;
- NVIDIA T4 cloud instance: lower-cost cloud baseline;
- NVIDIA L4 cloud instance: preferred cloud real-time candidate.

The XIAO ESP32-S3 Sense is a camera, buffering and transmission node. It is not the compute target for the full YOLOv8 plus tracking pipeline.

## Work estimate

- Freeze protocol and create the manifest: one day.
- Collect independent videos and label them: three to five days for a minimum study.
- Run local and cloud benchmarks: one to two days.
- Error analysis, confidence intervals and report figures: two days.
- Optional detector fine-tuning: additional data-labelling time, one to six GPU hours for an initial small-model experiment, and two to four days for controlled comparisons.

The expected minimum is seven to ten working days. Human labelling and error analysis, not raw GPU runtime, are the critical path.

## Authoritative references

- Ultralytics YOLOv8 specifications: https://docs.ultralytics.com/models/yolov8/
- Ultralytics validation metrics: https://docs.ultralytics.com/modes/val/
- MOTChallenge tracking metrics: https://motchallenge.net/results/CVPR_2019_Tracking_Challenge/
- NVIDIA Jetson Orin specifications: https://www.nvidia.com/en-gb/autonomous-machines/embedded-systems/jetson-orin/
- AWS accelerated-computing instances: https://docs.aws.amazon.com/ec2/latest/instancetypes/ac.html
- Seeed XIAO ESP32-S3 Sense specifications: https://wiki.seeedstudio.com/xiao_esp32s3_getting_started/
