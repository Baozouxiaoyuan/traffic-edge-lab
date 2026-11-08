"""Fetch public research checkpoints with resumable verified byte ranges."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import time
import zipfile

import requests

ROOT = Path(__file__).resolve().parents[1]
SOURCES = {
    "signals": ("traffic_lights_yolov8x.pt", "https://bwsyncandshare.kit.edu/s/ZLtoyf4q8nTLL7Y/download/traffic_lights_yolov8x.pt"),
    "arrows": ("road_markingsyolov8m.pt", "https://bwsyncandshare.kit.edu/s/5YQPPoz4mPSe3j9/download/road_markingsyolov8m.pt"),
    "markings": ("mapillary/model.safetensors", "https://huggingface.co/facebook/mask2former-swin-large-mapillary-vistas-semantic/resolve/main/model.safetensors"),
}


def fetch(name: str, workers: int = 8) -> None:
    relative, url = SOURCES[name]
    path = ROOT / "models" / "calibration" / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.with_suffix(path.suffix+".invalid").exists() or path.with_suffix(path.suffix+".rejected.json").exists():
        raise RuntimeError(f"The previously downloaded {name} checkpoint failed validation. Verify an alternative upstream checkpoint before retrying.")
    if name == "markings":
        for filename in ("config.json", "preprocessor_config.json"):
            if not (path.parent/filename).is_file():
                response = requests.get("https://huggingface.co/facebook/mask2former-swin-large-mapillary-vistas-semantic/resolve/main/"+filename, timeout=40)
                response.raise_for_status()
                (path.parent/filename).write_bytes(response.content)
    if path.is_file():
        print(f"Already installed: {path.name}", flush=True)
        return
    probe = requests.get(url+"?range_probe=1", headers={"Range": "bytes=0-0"}, timeout=40, stream=True)
    try:
        probe.raise_for_status()
        if probe.status_code != 206:
            raise RuntimeError("Server does not support bounded range downloads")
        total = int(probe.headers["Content-Range"].split("/")[-1])
    finally:
        probe.close()
    partial = path.with_suffix(path.suffix + ".ranges")
    partial.mkdir(exist_ok=True)
    chunk = 4*1024*1024
    count = (total+chunk-1)//chunk
    prefix = path.with_suffix(path.suffix+".partial")
    if prefix.is_file() and prefix.stat().st_size < total:
        with prefix.open("rb") as source:
            for index in range(prefix.stat().st_size//chunk):
                data = source.read(chunk)
                target = partial/str(index)
                if not target.is_file():
                    target.write_bytes(data)

    def part(index):
        start, end = index*chunk, min(total, (index+1)*chunk)-1
        dest = partial / str(index)
        if dest.is_file() and dest.stat().st_size == end-start+1:
            return
        for attempt in range(4):
            try:
                # Unique query prevents intermediaries caching one range for all requests.
                separator = "&" if "?" in url else "?"
                response = requests.get(url+separator+f"part={index}", headers={"Range": f"bytes={start}-{end}"}, timeout=(20,90))
                response.raise_for_status()
                expected_range = f"bytes {start}-{end}/{total}"
                if response.status_code != 206 or response.headers.get("Content-Range") != expected_range or len(response.content) != end-start+1:
                    raise RuntimeError(f"Incorrect content range for part {index}")
                dest.write_bytes(response.content)
                return
            except (requests.RequestException, RuntimeError):
                if attempt == 3:
                    raise
                time.sleep(1+attempt)

    print(f"Fetching {name}: {total/1024/1024:.1f} MiB / {count} parts", flush=True)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for done, _ in enumerate(pool.map(part, range(count)), 1):
            if done % 10 == 0 or done == count:
                print(f"{name}: {done}/{count} parts", flush=True)
    digest = hashlib.sha256()
    temporary = path.with_suffix(path.suffix+".assembling")
    with temporary.open("wb") as stream:
        for index in range(count):
            data = (partial/str(index)).read_bytes()
            digest.update(data)
            stream.write(data)
    if temporary.stat().st_size != total:
        raise RuntimeError("Assembled checkpoint length mismatch")
    if name != "markings":
        try:
            with zipfile.ZipFile(temporary) as archive:
                # Torch archives need an intact pickle header as well as a
                # central directory; a damaged all-zero prefix can look like TAR.
                archive.read(archive.infolist()[0])
        except (zipfile.BadZipFile,IndexError) as exc:
            temporary.replace(path.with_suffix(path.suffix+".invalid"))
            raise RuntimeError(f"Upstream {name} checkpoint is damaged: {exc}") from exc
    temporary.replace(path)
    path.with_suffix(path.suffix+".provenance.json").write_text(json.dumps({
        "source_url": url, "bytes": total, "sha256": digest.hexdigest(),
        "source_project": "https://github.com/KASTEL-MobilityLab/traffic-light-detection" if name != "markings" else "https://huggingface.co/facebook/mask2former-swin-large-mapillary-vistas-semantic",
    }, indent=2), encoding="utf-8")
    for item in partial.iterdir():
        item.unlink()
    partial.rmdir()
    prefix.unlink(missing_ok=True)
    print(f"Installed {path.name}: SHA256 {digest.hexdigest()}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("models", nargs="+", choices=list(SOURCES))
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    for name in args.models:
        fetch(name, max(1,min(32,args.workers)))
