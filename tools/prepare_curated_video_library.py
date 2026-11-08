"""Materialize visually screened, continuous source intervals without adding scene content.

Requires imageio-ffmpeg for this offline preparation tool only. The application
does not download videos or need this dependency at runtime.
"""
from __future__ import annotations

import concurrent.futures
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from traffic_edge.storage import atomic_write_json
from traffic_edge.scope import ACTIVE_SCENE_TYPES, ACTIVE_RULES


def digest(path: Path) -> str:
    result = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            result.update(block)
    return result.hexdigest()


def inspect(path: Path) -> dict:
    capture = cv2.VideoCapture(str(path))
    try:
        if not capture.isOpened():
            raise ValueError(f'Cannot decode {path}')
        fps = capture.get(cv2.CAP_PROP_FPS)
        frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        if not np.isfinite(fps) or fps <= 0 or frames <= 0:
            raise ValueError(f'Missing video timing: {path}')
        return {'width': int(capture.get(3)), 'height': int(capture.get(4)),
                'fps': round(fps, 6), 'frames': frames,
                'duration_seconds': round(frames/fps, 6)}
    finally:
        capture.release()


def materialize(item: dict, refresh: bool = False) -> dict:
    import imageio_ffmpeg

    if item.get('scene_type') not in ACTIVE_SCENE_TYPES or item.get('rule') not in ACTIVE_RULES:
        raise ValueError(f'Video is outside the current validated scene/rule scope: {item["id"]}')
    source = ROOT/item['source_path']
    destination = ROOT/item['path']
    destination.parent.mkdir(parents=True, exist_ok=True)
    before = inspect(source)
    start, end = item['source_interval_seconds']
    if start < 0 or end <= start or end > before['duration_seconds']+.1:
        raise ValueError(f'Invalid source interval: {item["id"]}')
    if refresh or not destination.exists():
        temporary = destination.with_suffix('.preparing.mp4')
        command = [imageio_ffmpeg.get_ffmpeg_exe(), '-hide_banner', '-loglevel', 'error',
                   '-ss', str(start), '-i', str(source), '-t', str(end-start),
                   '-map', '0:v:0', '-an', '-c:v', 'libx264', '-threads', '2',
                   '-preset', 'fast', '-crf', '18', '-pix_fmt', 'yuv420p',
                   '-movflags', '+faststart', '-y', str(temporary)]
        subprocess.run(command, check=True, capture_output=True)
        temporary.replace(destination)
    actual = inspect(destination)
    if (actual['width'], actual['height']) != (before['width'], before['height']):
        raise ValueError('A curated video must retain the source dimensions')
    if abs(actual['duration_seconds']-(end-start)) > .3:
        raise ValueError(f'Export duration changed: {item["id"]}')
    # Decode the complete export, not just its first thumbnail.
    subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), '-v', 'error', '-xerror',
                    '-i', str(destination), '-f', 'null', '-'],
                   check=True, capture_output=True)
    capture = cv2.VideoCapture(str(destination))
    tiles = []
    poster = destination.with_suffix('.jpg')
    try:
        for index, second in enumerate(np.linspace(.1, actual['duration_seconds']-.15, 12)):
            capture.set(cv2.CAP_PROP_POS_MSEC, float(second*1000))
            ok, frame = capture.read()
            if not ok:
                raise ValueError(f'Cannot review exported frame: {item["id"]} at {second}')
            if index == 1:
                # A clean still, without synthetic evidence boxes or labels.
                scale = min(1, 960/frame.shape[1])
                cv2.imwrite(str(poster), cv2.resize(frame, None, fx=scale, fy=scale))
            tile = cv2.resize(frame, (480, 270))
            band = np.zeros((28, 480, 3), np.uint8)
            cv2.putText(band, f'{second:.2f}s', (8, 20), 0, .5, (255,255,255), 1)
            tiles.append(np.vstack([band, tile]))
    finally:
        capture.release()
    sheet = ROOT/'runs'/'video-curation'/'exports'/(item['id']+'.jpg')
    sheet.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(sheet), np.vstack([np.hstack(tiles[i:i+3]) for i in range(0,12,3)]))
    return dict(item, **actual, sha256=digest(destination), source_sha256=digest(source),
                poster_path=poster.relative_to(ROOT).as_posix(),
                export_transform='continuous temporal trim; H.264 transcode; no crop, upscale, denoise, overlays or synthetic content',
                full_decode_verified=True, training_ready=False, evaluation_ready=False,
                annotation_status='pending_full_vehicle_level_review', split='unassigned')


def main() -> None:
    plan_path = ROOT/'data'/'curated_video_plan.json'
    plan = json.loads(plan_path.read_text(encoding='utf-8'))
    catalogue_path = ROOT/'data'/'curated_video_library.json'
    previous = json.loads(catalogue_path.read_text(encoding='utf-8')) if catalogue_path.exists() else {}
    previous = {item['id']: item for item in previous.get('videos', [])}

    def prepare(item):
        old = previous.get(item['id'], {})
        changed = any(old.get(key) != item.get(key) for key in ('source_path', 'source_interval_seconds'))
        return materialize(item, refresh=changed)

    with concurrent.futures.ThreadPoolExecutor(2) as pool:
        videos = list(pool.map(prepare, plan['videos']))
    catalogue = {key: value for key, value in plan.items() if key != 'videos'}
    catalogue['videos'] = videos
    atomic_write_json(catalogue_path, catalogue)
    print(json.dumps({'screened_clips': len(videos),
                      'source_groups': len({x['source_id'] for x in videos}),
                      'minutes': round(sum(x['duration_seconds'] for x in videos)/60, 2),
                      'all_exports_decoded': all(x['full_decode_verified'] for x in videos)}, indent=2))


if __name__ == '__main__':
    main()
