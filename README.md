# Traffic Edge Lab

Traffic Edge Lab is a fixed-camera traffic-video research application. Shared vehicle detection and tracking supply three rules: red-light running, wrong-way driving and restricted-zone entry. Candidates link to source times, lane configurations and exported evidence, and require human review.

## Download and run

Download **Code → Download ZIP** from [the personal GitHub repository](https://github.com/Baozouxiaoyuan/traffic-edge-lab), or clone it:

```sh
git clone https://github.com/Baozouxiaoyuan/traffic-edge-lab.git
cd traffic-edge-lab
```

Install Python 3.11, 3.12 or 3.13, then run this command in the extracted project directory. It creates a local virtual environment, installs CPU dependencies and downloads the official YOLOv8n and YOLO11n detectors.

```sh
python tools/setup.py --compute cpu
```

On Windows, you can double-click `SETUP_CAPSTONE.cmd` for CPU setup, then double-click `START_CAPSTONE.cmd`. On macOS or Linux use `python3 tools/setup.py --compute cpu`, then `sh START_CAPSTONE.sh`. The browser opens at `http://127.0.0.1:8765/`. Each computer runs its own processing and stores its own videos, configurations and results.

## NVIDIA acceleration

On a Windows or Linux computer with an NVIDIA GPU and a compatible driver, run:

```sh
python tools/setup.py --compute cuda
```

Windows also supports `SETUP_CAPSTONE.cmd cuda`. The default CUDA wheel build is `cu126`. You can select `--cuda-build cu130` or `--cuda-build cu132` for compatible newer GPUs and drivers. Consult the [official PyTorch installer](https://pytorch.org/get-started/locally/) for supported hardware and driver requirements.

In **Analysis settings**, choose **Automatic**, **CPU** or **NVIDIA CUDA**. Collision review has its own device selector. Automatic uses the first NVIDIA GPU when CUDA is available and otherwise uses the CPU. An explicit NVIDIA selection requires working CUDA support. Each run records the device actually used. GPU acceleration applies to YOLO detection and the optional collision model. Video decoding, optical flow and rule checks continue on the CPU.

The command-line pipeline also accepts `--device auto`, `--device cpu` or `--device cuda`. Saved JSON configurations can set `yolo.device`. Calibration and annotation can be used on CPU installations.

## Models and footage

To prepare the optional collision model, add `--collision` to the setup command, or run `.venv/Scripts/python.exe tools/prepare_collision_vlm.py` on Windows (`.venv/bin/python` on macOS/Linux). This downloads and verifies the pinned model, approximately 4.3 GB. Other calibration models can be prepared using the tools under `tools/`.

Import footage you are authorised to use. Add `--examples` to the setup command to download two development clips directly from Zenodo with checksum verification, or run `tools/prepare_examples.py` using the virtual environment. The saved Zenodo video4 camera profile and example setup are included. Source recordings, trained weights, personal annotations, thesis documents and generated evidence are stored locally. The repository contains the application, source catalogues and camera presets. The example library shows the clips present on each computer. Third-party video reuse terms are recorded in `data/real_examples/SOURCES.md`, `data/evaluation_candidates.json` and `data/curated_video_library.json`.

1. Select or upload a video.
2. Capture a clear setup frame and configure the monitored lanes.
3. Draw the geometry required by each rule: stop line and controlling signal, legal direction, or restricted zone.
4. Run the analysis and inspect candidate events against the original video.
5. Export evidence or complete independent annotation for evaluation.

Camera profiles are stored in `config/camera_profiles`. Reuse a profile only when camera position, crop and lens are unchanged. Normalized coordinates support resolution changes; a different viewpoint requires calibration. Up to eight lanes are supported. Each observation belongs to at most one lane; ambiguous ownership causes abstention.

## Calibration and perception

The setup assistant combines visible road markings, observed vehicle paths and traffic-light proposals. Confirm proposals before applying them. A vehicle path may cover only part of a lane. The road mask checks overlap and does not identify lane boundaries. Turn arrows and permitted turns require manual verification. Draw missing geometry manually or leave it unconfigured.

The default is YOLOv8n with ByteTrack. YOLO11n and BoT-SORT are available for comparison. The small pilot selected YOLO11n with ByteTrack on validation footage and matched one labelled event in the B06 test replay. Supported vehicle classes are car, bus and truck. Stop-sign compliance, motorcycles, fisheye cameras and roundabouts are outside the active scope.

## Collision review

The separate Collision review workspace examines selected intervals offline. Motion and vehicle tracks propose search regions; a local Qwen3-VL model reviews full-scene and local views. Outputs include candidate intervals, uncertain decisions and original-frame storyboards. This experimental module does not determine legal fault or provide a validated accident-detection accuracy estimate. CPU processing is slower than real time.

On the configured capstone machine, choose a clip under **Task recordings → Collision review**. The example opens at its preset interval. Rear-end collision exercises the candidate output, while the three cards labelled **No-collision control** exercise ordinary traffic.

## Video library and annotation

The library includes development examples and material awaiting evaluation. Development results are not ground truth. Source information and reuse conditions are recorded in the data catalogues and under `data/real_examples`. Earlier low-resolution footage remains only in historical regression records.

Review the original video with predictions hidden. Playback coverage must reach 98%; seeking does not count as viewing. Record lane, event time and vehicle location, resolve uncertain labels, then complete the review with a reviewer ID. Save fully reviewed videos without eligible events as negatives. Drafts are saved by source-video hash.

See [the annotation protocol](ANNOTATION_PROTOCOL.md) and [the evaluation plan](EVALUATION_AND_COMPUTE_PLAN.md). Local development and validation records include several red-light cases. The pilot test split contained one labelled B06 event. A wider accuracy study still needs independently reviewed normal footage and previously unseen cameras.

## Reviewed datasets and batch evaluation

The **Study register** workspace manages reviewed video cases without editing a manifest by hand.

1. Load a recording, finish its independent review in **Ground truth**, and configure the camera geometry in **Lane setup**.
2. In **Study register**, enter a case name, source group, physical camera ID and development, validation or test use. Register the current reviewed video. A completed review with no eligible events is stored as a negative recording.
3. Select one split, detector, tracker and compute device, then choose **Run reviewed batch**. The batch shares the existing processing queue with individual analyses and collision reviews.
4. Inspect overall, per-recording and per-rule TP, FP, FN, Precision, Recall and F1. Open false-positive or missed-event frames, vehicle crops, short clips and the original recording at the event time. Export report JSON, case-metrics CSV and error CSV.

The register rejects source or physical-camera overlap between test and development/validation, detects identical video content, and requires at least 98% original-video review coverage. It saves hashes of videos, labels and review records, and freezes each case's camera geometry. Changed inputs require registration again. A development or validation case cannot be promoted directly into test. Choose fresh test sources and fix the detector/tracker choice before testing.

Completed batch reports remain available after restarting the application. The local register is stored in `data/study_registry.json`, and frozen batch inputs and reports are under `runs/study-*`. Removing a case from the register preserves its source, labels and previous results. Undefined metric denominators appear as a dash in the interface, while the underlying evaluator retains its documented zero convention.

## Command-line use

Run a configured video from PowerShell:

```powershell
$env:PYTHONPATH = "$PWD\src"
.\.venv\Scripts\python.exe -m traffic_edge.cli run `
  --input .\data\zenodo_video3.mp4 `
  --output-dir .\runs\zenodo-video3 `
  --config .\config\zenodo_video3.json `
  --backend yolo
```

Each run saves its configuration, annotated video, event records, consolidated incidents, summary and evidence images. Web runs also generate event clips. `EVIDENCE_MANIFEST.json` records sizes and SHA-256 hashes; `evidence_bundle.zip` packages review evidence. `RUN_REPORT.md` describes the run. Byte-reduction figures cover selected outputs, rather than all generated files or measured network traffic.

Evaluate completed vehicle-specific labels:

```powershell
$env:PYTHONPATH = "$PWD\src"
.\.venv\Scripts\python.exe -m traffic_edge.cli evaluate `
  --events .\runs\RUN_NAME\events.jsonl `
  --ground-truth .\data\annotations\LABELS.json `
  --tolerance 0.5 --spatial-tolerance 0.12
```

Matching uses event type, time and vehicle location, with lane identity required when supplied by both records. For model comparison, populate `data/accuracy_campaign.template.json` and run `traffic_edge.cli campaign --manifest PATH --output-dir PATH --dry-run`. After readiness checks pass, omit `--dry-run` to select on validation data and evaluate the frozen choice on test data. The campaign hashes videos, labels, configurations, weights and implementation inputs; changed inputs cannot reuse cached results.

## Verification

Run `TEST_CAPSTONE.cmd` on Windows for compilation, Python tests, the interface smoke test and JavaScript regression suites. Install Node.js for the JavaScript suites. Tests create temporary fixtures and verify the source catalogue. For a portable synthetic demo, run `python -m traffic_edge.cli generate-demo --output data/demo.mp4` after setting `PYTHONPATH=src`, then use the demo backend with `config/demo.json`. Real-video accuracy requires independent labels and frozen scene configurations.

## Project directories

- `src`: application code.
- `tests`: regression tests for supported behaviour.
- `config`: active presets and saved camera profiles.
- `data`: videos, source records and annotations.
- `models`: local inference weights.
- `tools`: model setup and data preparation utilities.
- `runs`: experiment records and generated evidence.
- `output/submission`: exported documents.
- `.venv`: the Python environment used by the application.

Superseded project material is grouped under `../archive`. Experiment records in `runs` remain at their cited paths.
