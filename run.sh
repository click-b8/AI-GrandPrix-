#!/bin/bash
# Convenience script to run with MuJoCo viewer on macOS
# Usage: ./run.sh race.py [args...]  or  ./run.sh train.py [args...]
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
VENV_PYTHON="$SCRIPT_DIR/venv/bin/python3.11"
MJPYTHON_SCRIPT="$SCRIPT_DIR/venv/bin/mjpython"

cd "$SCRIPT_DIR"

if [ "$1" = "train.py" ]; then
    # Training doesn't need GUI, use plain python
    "$VENV_PYTHON" "$@"
else
    # Race/demo needs MuJoCo viewer, use mjpython trampoline
    "$VENV_PYTHON" -c "
import sys, os
sys.argv = ['mjpython'] + sys.argv[1:]
exec(open('$MJPYTHON_SCRIPT').read())
" "$@"
fi
