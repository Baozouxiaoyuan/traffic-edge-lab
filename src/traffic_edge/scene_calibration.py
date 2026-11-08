"""Learned static-scene evidence, separate from traffic-rule inference."""
from __future__ import annotations

import base64
import hashlib
import threading
import time
from collections import OrderedDict
from pathlib import Path

import cv2
import numpy as np

MODEL_ID = "facebook/mask2former-swin-large-mapillary-vistas-semantic"
_LOCK = threading.Lock()
_MODEL = None
_CACHE: OrderedDict = OrderedDict()
MODEL_DIR = Path(__file__).resolve().parents[2] / "models" / "calibration"
_DETECTORS = {}
_DETECTOR_LOCK = threading.Lock()


def signal_reference_evidence(frame: np.ndarray, rectangle: list[list[float]]) -> dict:
    """Inspect the frozen frame only; never infer a phase from an unreadable head."""
    from .signal import RedSignalDetector, denormalize_rectangle

    height,width=frame.shape[:2]
    x1,y1,x2,y2=denormalize_rectangle(rectangle,width,height)
    native_width,native_height=x2-x1,y2-y1
    clipped=x1<=1 or y1<=1 or x2>=width-2 or y2>=height-2
    reason='cropped_at_frame_edge' if clipped else 'too_few_pixels' if native_width<6 or native_height<12 else None
    state='unknown'
    if reason is None:
        detector=RedSignalDetector((x1,y1,x2,y2),history_size=1)
        detector.update(frame)
        if detector.last_state=='red':
            state='red'
        else:
            reason='red_not_established'
    padding=max(3,round(native_width*.25))
    crop=frame[max(0,y1-padding):min(height,y2+padding),max(0,x1-padding):min(width,x2+padding)]
    factor=min(6,160/max(crop.shape[:2]))
    preview=cv2.resize(crop,None,fx=factor,fy=factor,interpolation=cv2.INTER_NEAREST)
    ok,encoded=cv2.imencode('.png',preview)
    if not ok:
        raise RuntimeError('Could not encode signal detail')
    return {
        'reference_red_state':state,'reason':reason,'native_size_pixels':[native_width,native_height],
        'detail_preview':'data:image/png;base64,'+base64.b64encode(encoded).decode('ascii'),
        'scope':'frozen_setup_frame_only','review_required':True,
    }


def segment_scene(frame: np.ndarray) -> tuple[np.ndarray, dict]:
    """Return native-size labels; download public weights on first use."""
    global _MODEL
    digest = hashlib.sha256(frame.tobytes()).hexdigest()
    with _LOCK:
        if digest in _CACHE:
            labels, metadata = _CACHE[digest]
            return labels.copy(), dict(metadata, cached=True)
        import torch
        from PIL import Image
        from transformers import AutoImageProcessor, Mask2FormerForUniversalSegmentation

        if _MODEL is None:
            local = MODEL_DIR / "mapillary"
            source = str(local) if (local / "model.safetensors").is_file() else MODEL_ID
            processor = AutoImageProcessor.from_pretrained(source, use_fast=False)
            model = Mask2FormerForUniversalSegmentation.from_pretrained(source, use_safetensors=True)
            model.eval()
            device = "cuda" if torch.cuda.is_available() else "cpu"
            model.to(device)
            _MODEL = processor, model, device
        processor, model, device = _MODEL
        started = time.perf_counter()
        image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        inputs = processor(images=image, size={"shortest_edge": 512, "longest_edge": 1024}, return_tensors="pt")
        with torch.inference_mode():
            outputs = model(**{key: value.to(device) for key, value in inputs.items()})
            labels = processor.post_process_semantic_segmentation(outputs, target_sizes=[frame.shape[:2]])[0]
        labels = labels.cpu().numpy().astype(np.uint8)
        metadata = {
            "model": MODEL_ID, "device": device,
            "inference_seconds": round(time.perf_counter() - started, 3),
            "classes": {str(key): value for key, value in model.config.id2label.items()},
            "cached": False,
            "limitations": "General marking pixels do not classify stop lines or turn arrows. Lane geometry requires review.",
        }
        _CACHE[digest] = labels, metadata
        while len(_CACHE) > 3:
            _CACHE.popitem(last=False)
        return labels.copy(), metadata


def scene_evidence(reference: np.ndarray, samples: list[np.ndarray], reference_boxes: list | None=None,
                   sample_boxes: list[list] | None=None) -> tuple[np.ndarray, dict]:
    """Accumulate only paint seen outside foreground occlusions in aligned frames."""
    from .scene_alignment import foreground_mask, align_static_scene

    if sample_boxes is not None and len(sample_boxes)!=len(samples):
        raise ValueError('Each sampled frame must have its own foreground boxes')
    labels, metadata = segment_scene(reference)
    reference_foreground=foreground_mask(labels.shape,reference_boxes,labels)
    labels[reference_foreground>0]=255
    used,skipped=1,0
    alignments=[]
    seen={hashlib.sha256(reference.tobytes()).hexdigest()}
    for index,frame in enumerate(samples):
        digest=hashlib.sha256(frame.tobytes()).hexdigest()
        if digest in seen:
            continue
        seen.add(digest)
        if frame.shape!=reference.shape:
            skipped+=1
            alignments.append({'accepted':False,'reason':'different_image_shape'})
            continue
        boxes=sample_boxes[index] if sample_boxes is not None else None
        sample_foreground=foreground_mask(frame.shape[:2],boxes)
        transform,alignment=align_static_scene(reference,frame,reference_foreground,sample_foreground)
        alignments.append(alignment)
        if transform is None:
            skipped+=1
            continue
        extra,_=segment_scene(frame)
        sample_foreground=foreground_mask(extra.shape,boxes,extra)
        extra[sample_foreground>0]=255
        extra=cv2.warpAffine(extra,transform,(labels.shape[1],labels.shape[0]),
                             flags=cv2.INTER_NEAREST|cv2.WARP_INVERSE_MAP,
                             borderMode=cv2.BORDER_CONSTANT,borderValue=255)
        unseen=(labels==255)&(extra!=255)
        labels[unseen]=extra[unseen]
        road_surface=np.isin(labels,[13,23,24,255])
        labels[(extra==24)&road_surface&(labels!=23)]=24
        labels[(extra==23)&road_surface]=23
        used+=1
    return labels,dict(metadata,temporal_frames_used=used,unstable_frames_skipped=skipped,
                       occlusion_filtering=True,reference_foreground_pixels=int(reference_foreground.sum()),
                       unrecovered_foreground_pixels=int(np.count_nonzero((reference_foreground>0)&(labels==255))),
                       frame_alignment=alignments)


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    overlap = np.maximum(0, np.minimum(a[1], b[1]) - np.maximum(a[0], b[0]))
    intersection = float(np.prod(overlap))
    return intersection / max(float(np.prod(a[1] - a[0]) + np.prod(b[1] - b[0])) - intersection, 1e-9)


def merge_signal_detections(detections: list[dict]) -> list[dict]:
    """Cluster by overlap, never fixed centre distance (adjacent heads differ)."""
    groups: list[list[dict]] = []
    for detection in sorted(detections, key=lambda item: -item["confidence"]):
        box = np.asarray(detection["rectangle"])
        group = next((g for g in groups if _iou(box, np.asarray(g[0]["rectangle"])) >= 0.35), None)
        if group is None:
            groups.append([detection])
        else:
            group.append(detection)
    proposals = []
    for group in groups:
        frames = {item.get("frame_index", 0) for item in group}
        proposals.append({
            "rectangle": np.median([item["rectangle"] for item in group], axis=0).round(6).tolist(),
            "confidence": round(max(item["confidence"] for item in group), 4),
            "observations": len(frames), "review_required": True,
            "method": "whole_frame_and_overlapping_tiles",
            "observed_labels": sorted({item.get("label", "traffic_light") for item in group}),
        })
    proposals.sort(key=lambda item: (-item["observations"], -item["confidence"]))
    return [dict(item, proposal_id=index + 1) for index, item in enumerate(proposals)]


def detect_signals(model, frames: list[np.ndarray], image_size: int = 960, class_ids: list[int] | None = None) -> tuple[list[dict], dict]:
    """Full frame plus 2x2 overlapping native-image crops with mapped boxes."""
    detections = []
    views_processed = 0
    started = time.perf_counter()
    distinct={hashlib.sha256(frame.tobytes()).hexdigest():frame for frame in frames}
    frames=list(distinct.values())
    for frame_index, frame in enumerate(frames):
        height, width = frame.shape[:2]
        views = [(0, 0, width, height)]
        # Do not manufacture tiny crops from low-information video.
        if width >= 640 and height >= 360:
            tile_w, tile_h = round(width * 0.62), round(height * 0.62)
            views += [(x, y, x + tile_w, y + tile_h)
                      for y in (0, height - tile_h) for x in (0, width - tile_w)]
        for x1, y1, x2, y2 in views:
            result = model.predict(frame[y1:y2, x1:x2], conf=0.15, iou=0.45, classes=class_ids,
                                   imgsz=image_size, verbose=False)[0]
            views_processed += 1
            if result.boxes is None:
                continue
            for box, score, category in zip(result.boxes.xyxy.cpu().tolist(), result.boxes.conf.cpu().tolist(), result.boxes.cls.cpu().tolist()):
                a, b, c, d = box
                # Truncated tile-border boxes are excluded; full-frame context
                # and overlapping neighbours cover these objects instead.
                if len(views) > 1 and (x1 or y1 or x2 != width or y2 != height):
                    if (x1 > 0 and a <= 2) or (y1 > 0 and b <= 2) or (x2 < width and c >= x2-x1-2) or (y2 < height and d >= y2-y1-2):
                        continue
                detections.append({"rectangle": [[(a+x1)/width, (b+y1)/height], [(c+x1)/width, (d+y1)/height]],
                                   "confidence": float(score), "frame_index": frame_index,
                                   "label": str(model.names[int(category)])})
    proposals=merge_signal_detections(detections)
    required=min(2,len(frames))
    height,width=frames[0].shape[:2] if frames else (1,1)
    confirmed=[]
    for proposal in proposals:
        a,b=np.asarray(proposal['rectangle'])*[width,height]
        if proposal['observations']<required or b[0]-a[0]<3 or b[1]-a[1]<5:
            continue
        proposal['proposal_id']=len(confirmed)+1
        confirmed.append(proposal)
    return confirmed, {
        "method": "whole_frame_and_overlapping_tiles", "frames_sampled": len(frames),
        "views_processed": views_processed, "seconds": round(time.perf_counter()-started, 3),
        "association": "unconfirmed",
        "rejected_weak_candidates":len(proposals)-len(confirmed),
        "minimum_distinct_frame_observations":required,
    }


def recognize_signals(fallback_model, frames: list[np.ndarray], image_size: int = 960) -> tuple[list[dict], dict]:
    with _DETECTOR_LOCK:
        path = MODEL_DIR / "traffic_lights_yolov8x.pt"
        if path.is_file():
            from ultralytics import YOLO
            if "signals" not in _DETECTORS:
                _DETECTORS["signals"] = YOLO(str(path))
            model, class_ids = _DETECTORS["signals"], None
        else:
            model, class_ids = fallback_model, [9]
        signals, meta = detect_signals(model, frames, image_size, class_ids)
        meta["model"] = str(path.name) if path.is_file() else "general_COCO_detector"
        meta["specialist_installed"] = path.is_file()
        if path.with_suffix(".pt.provenance.json").is_file():
            import json
            meta["sha256"] = json.loads(path.with_suffix(".pt.provenance.json").read_text())["sha256"]
        return signals, meta


def recognize_arrows(frame: np.ndarray) -> tuple[list[dict], dict]:
    path = MODEL_DIR / "road_markingsyolov8m.pt"
    if not path.is_file():
        reason = "The upstream road-arrow checkpoint failed file validation; arrow classification is unavailable." if path.with_suffix(".pt.rejected.json").exists() else "Specialist road-marking weights are not installed."
        return [], {"available": False, "reason": reason}
    with _DETECTOR_LOCK:
        from ultralytics import YOLO
        if "arrows" not in _DETECTORS:
            _DETECTORS["arrows"] = YOLO(str(path))
        model = _DETECTORS["arrows"]
        result = model.predict(frame, imgsz=1280, conf=.30, verbose=False)[0]
        height, width = frame.shape[:2]
        proposals = []
        for box, score, category in zip(result.boxes.xyxy.cpu().tolist(), result.boxes.conf.cpu().tolist(), result.boxes.cls.cpu().tolist()):
            label = model.names[int(category)]
            proposals.append({"label": label, "rectangle": [[box[0]/width,box[1]/height],[box[2]/width,box[3]/height]],
                              "confidence": round(float(score),4), "review_required": True,
                              "legal_direction_confirmed": False})
        return proposals, {"available": True, "model": path.name, "classes": model.names}


def marking_boundaries(labels: np.ndarray) -> list[dict]:
    """Extract supported marking segments; do not turn all image edges into lanes."""
    height, width = labels.shape
    mask = (labels == 24).astype(np.uint8) * 255
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    thickness = cv2.distanceTransform(mask,cv2.DIST_L2,3)
    lines = cv2.HoughLinesP(mask, 1, np.pi/720, threshold=max(20, round(width*.025)),
                            minLineLength=max(30, round(width*.055)), maxLineGap=round(width*.035))
    candidates = []
    for raw in [] if lines is None else lines[:, 0]:
        x1, y1, x2, y2 = map(float, raw)
        length = float(np.hypot(x2-x1, y2-y1))
        samples = np.linspace([x1,y1],[x2,y2],80).astype(int)
        values = thickness[samples[:,1],samples[:,0]]
        stroke = float(np.median(values[values > 0])*2) if np.any(values > 0) else 2
        candidates.append({"points": [[x1/width, y1/height], [x2/width, y2/height]], "length": length,"stroke_width":stroke})
    candidates.sort(key=lambda item: -item["length"])
    accepted = []
    for item in candidates:
        pts = np.asarray(item["points"]) * [width, height]
        vector = pts[1]-pts[0]
        unit = vector / item["length"]
        redundant = False
        for other in accepted:
            old = np.asarray(other["points"]) * [width, height]
            old_vector = old[1]-old[0]
            if abs(float(np.dot(unit, old_vector / np.linalg.norm(old_vector)))) < .98:
                continue
            distance = abs(float(np.dot(old.mean(axis=0)-pts[0], [-unit[1], unit[0]])))
            projection = (old-pts[0]) @ unit
            separation = max(6,(item["stroke_width"]+other["stroke_width"])*.75+3)
            if distance < separation and max(projection) >= -20 and min(projection) <= item["length"]+20:
                cloud = np.concatenate([pts,old]).astype(np.float32)
                vx,vy,x,y = cv2.fitLine(cloud,cv2.DIST_L2,0,.01,.01).flatten()
                tangent, origin = np.asarray([vx,vy]),np.asarray([x,y])
                distances = (cloud-origin)@tangent
                merged = np.asarray([origin+tangent*min(distances),origin+tangent*max(distances)])
                other["points"] = (merged/[width,height]).tolist()
                other["length"] = float(max(distances)-min(distances))
                other["stroke_width"] = max(other["stroke_width"],item["stroke_width"])
                redundant = True
                break
        if not redundant:
            accepted.append(dict(item, review_required=True, method="learned_marking_pixels"))
        if len(accepted) >= 40:
            break
    # A long transverse stripe touched by several longitudinal markings is a
    # junction boundary, not another longitudinal lane boundary. This also
    # prevents pairing the stop line with an arrow shaft as a fictitious lane.
    for item in accepted:
        item["role"] = "unclassified_marking"
        a = np.asarray(item["points"])*[width,height]
        vector = a[1]-a[0]
        length = np.linalg.norm(vector)
        if length < np.hypot(width,height)*.20:
            continue
        unit = vector/length
        contacts = []
        for other in accepted:
            if other is item:
                continue
            b = np.asarray(other["points"])*[width,height]
            direction = b[1]-b[0]
            if abs(np.dot(unit,direction/np.linalg.norm(direction))) > .98:
                continue
            for point in b:
                offset = point-a[0]
                t = float(np.dot(offset,unit))
                distance = abs(float(np.dot(offset,[-unit[1],unit[0]])))
                if -10 <= t <= length+10 and distance < max(14,item["stroke_width"]*3):
                    if all(abs(t-old) > 35 for old in contacts):
                        contacts.append(t)
        if len(contacts) >= 2:
            item["role"] = "transverse_marking"
    return accepted


def scene_overlay(frame: np.ndarray, labels: np.ndarray | None, boundaries: list[dict],
                  signals: list[dict], lanes: list[dict] | None = None) -> str:
    overlay = frame.copy()
    if labels is not None:
        for label, colour in [(24, (0, 220, 255)), (23, (255, 130, 80)), (2, (190, 100, 255))]:
            pixels = labels == label
            overlay[pixels] = (.35 * overlay[pixels] + .65 * np.asarray(colour)).astype(np.uint8)
    height, width = frame.shape[:2]
    for index, boundary in enumerate(boundaries):
        pts = np.round(np.asarray(boundary["points"]) * [width, height]).astype(int)
        cv2.line(overlay, tuple(pts[0]), tuple(pts[1]), (0, 255, 255), 2)
    for index, signal in enumerate(signals):
        pts = np.round(np.asarray(signal["rectangle"]) * [width, height]).astype(int)
        cv2.rectangle(overlay, tuple(pts[0]), tuple(pts[1]), (40, 255, 80), 2)
        cv2.putText(overlay, f"S{index+1}", (pts[0, 0], max(14, pts[0, 1]-5)), cv2.FONT_HERSHEY_SIMPLEX, .55, (40,255,80), 2)
    for index, lane in enumerate(lanes or []):
        pts = np.round(np.asarray(lane["polygon"]) * [width, height]).astype(np.int32)
        cv2.polylines(overlay, [pts], True, (255, 180, 0), 2)
        centre = pts.mean(axis=0).astype(int)
        cv2.putText(overlay, f"L{index+1}", tuple(centre), cv2.FONT_HERSHEY_SIMPLEX, .7, (255,180,0), 2)
        for candidate in lane.get("stop_line_candidates",[])[:1]:
            line=np.round(np.asarray(candidate["points"])*[width,height]).astype(int)
            cv2.line(overlay,tuple(line[0]),tuple(line[1]),(255,40,255),3)
            cv2.putText(overlay,f"T{index+1}",tuple(line.mean(axis=0).astype(int)),cv2.FONT_HERSHEY_SIMPLEX,.55,(255,40,255),2)
    ok, encoded = cv2.imencode(".jpg", overlay, [cv2.IMWRITE_JPEG_QUALITY, 90])
    if not ok:
        raise RuntimeError("Could not encode scene preview")
    return "data:image/jpeg;base64," + base64.b64encode(encoded).decode("ascii")
