# Presentation readiness — 8 November 2026

## Frozen demonstration

Use the existing Zenodo video4 red-light example and its `zenodo-video4-verified-11100` camera profile. The profile and `config/example_setups/zenodo-video4-stress.json` have identical `lanes` and `inference` contents; metadata may differ. Do not select the older top-level `zenodo_video4_tracking_stress.json` CLI wrong-way configuration as a substitute: it has a different purpose.

The presentation reproduction is in `runs/presentation-verified-pickup-20261108/`. It uses the previously verified run's saved configuration, YOLOv8n, ByteTrack and CPU. The 536-frame replay emitted one red-light candidate for truck, track 77, lane 1, at 11.100 s. This is a development replay, not a new independent accuracy sample.

The 225-frame thesis benchmark remains separately recorded under `runs/thesis-submission-20261104/`. Do not assign its 19.24 FPS / 14.193 s / 1.68 MB figures to the new pickup run. The original pilot selection and B06 test replay remain under `runs/accuracy-campaign-zenodo-video3-b06-refined/`.

## Cleanup scope

Removed unused Python imports from `collision_motion.py`, `intersection_assist.py` and `web_ui.py`. The optical-flow source-path root remains required. Existing scripts, model preparation tools, regression tests and cited run records remain in place.

No application features were added in this preparation pass. Do not perform a large split of the embedded `web_ui.py` CSS/JavaScript before presentation: that is a future maintainability task requiring a separate interface regression pass. `run_demo.ps1` remains a useful synthetic end-to-end check and is not dead code. The separate legacy detector-comparison workflow still supports existing campaign records; it cannot be removed merely because the Study register is newer.

## Demonstration route

Show one complete route: recording → saved camera geometry → completed candidate → original evidence → exported configuration/evidence. Have the same completed replay ready offline. Keep collision analysis and detailed evaluation settings for Q&A. Avoid live model downloads, collision inference or a long batch study during the nine-minute talk.

## Evidence boundaries

Development successes demonstrate functionality. The recorded B06 pilot has one labelled test event and prior development exposure. Software regression tests do not add real-world test events. Collision feasibility counts are selected-window model decisions. Dedicated edge-device, power and controlled GPU speedup results have not been established.

The local presentation package is outside this repository under `../presentation/`; it contains the deck, rehearsal guidance and the local demo copy. Third-party source footage is not committed to GitHub.
