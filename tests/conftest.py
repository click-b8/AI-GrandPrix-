"""Import paths for the test suite.

The portfolio restructure relocated the entry-point scripts out of the repo root:
`flight_report.py` and `track.py` moved to `analysis/`, and the DCL hardware adapters
moved to `archive/dcl-hardware/`. The tests still import them by bare module name --
deliberately, since the archived-era integration tests are still exercised rather than
deleted -- so those directories go on sys.path here.

`archive/rl-training/` is here for the same reason at one remove: dcl_adapter imports
`vision_model`, which lives with the training code.

Path setup only: no fixtures, no collection hooks, no behaviour.
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

for _sub in ("", "analysis",
             os.path.join("archive", "dcl-hardware"),
             os.path.join("archive", "rl-training")):
    _p = os.path.join(ROOT, _sub) if _sub else ROOT
    if _p not in sys.path:
        sys.path.insert(0, _p)
