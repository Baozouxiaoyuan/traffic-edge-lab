# Real-video provenance

## `njit_wrong_way_car.mp4`

- Original filename: `wrong_way_cam1.mp4`
- Provider: New Jersey Institute of Technology / New Jersey Department of Transportation research demo
- Demonstration page: https://web.njit.edu/~cliu/NJDOT/DEMOS.html
- Direct source: https://web.njit.edu/~cliu/NJDOT/wrong-way/wrong_way_cam1.mp4
- Content: real, normal-perspective fixed highway CCTV containing four-wheel vehicles travelling against the configured carriageway direction
- Local purpose: built-in wrong-way development example for cars, buses and trucks; motorcycles are excluded by the detector configuration
- Source characteristics: 360 x 240, 15 FPS, approximately 58 seconds; the provider's Korean detection overlay is embedded in the source frames

The public demonstration page does not state a reuse licence. Keep this copy for local capstone research and classroom demonstration, cite the provider, and do not redistribute it without confirming permission.

## Removed fisheye research archive

An earlier development copy included the FETV fisheye motorcycle dataset. It was removed from the final package because motorcycles and fisheye-camera validation are outside the submitted scope and the baseline detector was not reliable on its small distorted targets. No FETV result is used as thesis evidence.

## `zenodo_red_light_event.webm`

- Parent file: `rlr_occasions.mp4`
- Dataset: Dataset for Traffic Evaluation, Zenodo record 14041920
- Record: https://zenodo.org/records/14041920
- Parent-file official MD5: `b8cdcc48799eb04df21bff98a33788e5` (verified locally before extraction)
- Extracted source interval: 103.5–111.0 seconds; no scene content was generated or altered
- Local SHA-256: `e5c8eec6d9674350fe22aa3a072eb13cc6d5c0a906e988e8859ef9ba16e6ea1b`
- Local purpose: real red-light development example with red-light, wrong-way and restricted-zone rules executed in one pass
- Review status: one confirmed label at 4.768 s. The current saved preset matched it at 4.933 s and produced one additional red-light candidate without a matching label.

The Zenodo record describes the collection as monitored-road footage for traffic-analysis and red-light-running model tests. The record currently displays no explicit licence value, so retain this excerpt for local capstone research and do not redistribute it without confirming permission.

## `zenodo_video4.mp4`

- Original filename: `video4.mp4`
- Dataset: Dataset for Traffic Evaluation, Zenodo record 14041920
- Record: https://zenodo.org/records/14041920
- Official MD5 verified locally: `cd375768acf4f3aea559462d98816645`
- Local SHA-256: `96e53a5c673ca592bfd6828b87a3c45fb6e49714bfc05ad351c85b6a79428b3a`
- Source characteristics: 1280 x 720, 30 FPS, approximately 17.9 seconds
- Local purpose: a separate real daytime intersection clip for tracking continuity and partial-occlusion stress testing
- Review status: one confirmed pickup label at 11.591 s. The saved red-light setup matched it at 11.100 s in a development replay.

This is a distinct source clip rather than an excerpt of the built-in red-light event. It uses the same Zenodo collection and inherits the record's currently unspecified licence status, so keep it for local capstone research and do not redistribute it without confirming permission.

## `idiap_late_stop.webm` and `idiap_pedestrian_entry.webm`

- Dataset: Idiap Traffic Junction Dataset
- Official page: https://www.idiap.ch/~odobez/RESSOURCES/DataRelease-TrafficJunction.php
- Official annotations: https://www.idiap.ch/~odobez/RESSOURCES/TRAFFIC-JUNCTION-DATA/Annotations.txt
- Source characteristics: 360 x 288, 25 FPS, fixed junction camera
- `idiap_late_stop.webm`: source interval 2:45–3:40; SHA-256 `af8af3086bfc077f2f9dac444f5d0696169d639b5d5f635f65d072d06571be61`
- `idiap_pedestrian_entry.webm`: source interval 36:00–36:30; SHA-256 `3fdeca4f77c5d6d37981d670229d2d24c5a2593c21698135ce513a33a1e78dba`
- Published labels covered by the excerpts: Event A, car stopping after the stop line (2:52–3:32); Event D, car entering a pedestrian area (36:08–37:54)

Idiap permits non-commercial research use and requests citation of Varadarajan and Odobez (ICCV Visual Surveillance Workshop, 2009). The late-stop excerpt is deliberately presented as a hard recall case: the physical stop line is near the image boundary, so a track may begin only after crossing it. The pedestrian-area excerpt enables initial-presence review because a vehicle can first become detectable after entering the selected zone. Both outputs remain candidate events requiring vehicle-specific manual labels.

## AI City Challenge 2026 candidate subset

- Dataset: NVIDIA PhysicalAI Traffic Anomaly Reasoning, AI City Challenge 2026 Track 3
- Official dataset card: https://huggingface.co/datasets/nvidia/PhysicalAI-Traffic-Anomaly-Reasoning
- Official challenge page: https://www.aicitychallenge.org/2026-track3/
- Upstream repositories used locally: `Open-Space-Reasoning/AccidentBench` and `wbfwonderful/Vad-R1` on Hugging Face
- Local directory: `data/real_examples/tar2026_candidates/`
- Machine-readable provenance, hashes and screening decisions: `data/evaluation_candidates.json`

The annotation release is CC BY 4.0, but the videos retain the terms of their upstream datasets. Keep the downloaded files for local research unless upstream redistribution rights are verified. The released training annotations were produced by an automated VLM pipeline, so they are used only to find likely events. They are not accepted as final accuracy ground truth.

Visual screening retained four fixed normal-perspective red-light candidates for manual vehicle-level review. The VAD-R1 motorway item and two edited AccidentBench compilations were removed from the final package because their camera changes make them invalid as continuous fixed-camera evaluation inputs. Their screening decision remains described in `data/DATASET_PLAN.md`; they are not exposed in the application.
