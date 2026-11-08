"""Replace saved JSON only after a complete file has been written."""
from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any


def atomic_write_json(destination: Path, value: Any) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', newline='\n',
                                         dir=destination.parent, prefix=destination.name+'.',
                                         suffix='.tmp', delete=False) as stream:
            temporary = Path(stream.name)
            json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
            stream.write('\n')
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
