#!/bin/bash
# Launcher for MuJoCo viewer (workaround for spaces in path)
DIR="$(cd "$(dirname "$0")" && pwd)"
export PATH="$DIR/.venv/bin:$PATH"
exec "$DIR/.venv/bin/python3" "$DIR/.venv/bin/mjpython" "$DIR/race.py" --model "$DIR/trained_swift/best_model/best_model.zip" --slow 1.5
