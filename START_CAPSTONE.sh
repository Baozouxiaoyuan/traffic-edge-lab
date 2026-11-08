#!/bin/sh
set -eu
cd -- "$(dirname -- "$0")"
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
if [ ! -x .venv/bin/python ]; then
    echo "Run python3 tools/setup.py --compute cpu first."
    exit 1
fi
exec .venv/bin/python -m traffic_edge.web_ui "$@"
