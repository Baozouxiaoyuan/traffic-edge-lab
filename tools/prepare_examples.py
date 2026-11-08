"""Fetch two development clips directly from their cited Zenodo source."""
from __future__ import annotations

import hashlib
import os
from pathlib import Path
import tempfile

import requests

ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = (
    ("video3.mp4", "data/zenodo_video3.mp4", "9dae5722a3294b37f4747ffd45c639d3"),
    ("video4.mp4", "data/real_examples/zenodo_video4.mp4", "cd375768acf4f3aea559462d98816645"),
)


def main() -> None:
    print("Downloading original Zenodo clips for local review. Source terms are recorded in data/real_examples/SOURCES.md.")
    for name, relative, expected in EXAMPLES:
        target = ROOT / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            with target.open("rb") as stream:
                actual = hashlib.file_digest(stream, "md5").hexdigest()
            if actual != expected:
                raise ValueError(f"Existing file differs from the source checksum. Preserve or move it before retrying: {target}")
            print(f"Verified: {target}")
            continue
        descriptor, temporary = tempfile.mkstemp(dir=target.parent, suffix=".incomplete")
        try:
            with os.fdopen(descriptor, "wb") as output:
                url = f"https://zenodo.org/records/14041920/files/{name}?download=1"
                with requests.get(url, stream=True, timeout=(30, 60)) as response:
                    response.raise_for_status()
                    for chunk in response.iter_content(1024 * 1024):
                        output.write(chunk)
            with open(temporary, "rb") as stream:
                actual = hashlib.file_digest(stream, "md5").hexdigest()
            if actual != expected:
                raise ValueError(f"Source checksum mismatch for {name}")
            os.replace(temporary, target)
            print(f"Downloaded and verified: {target}")
        finally:
            Path(temporary).unlink(missing_ok=True)


if __name__ == "__main__":
    main()
