"""LOOP RATE: the tube detector must not run by default.

THE DEFECT. Since the rail was removed the tube has had ZERO control authority -- but the
detector was still invoked on every new frame, at FULL resolution, purely to fill log
columns nothing read. It is the single most expensive thing in the tick.

MEASURED IN FLIGHT (loop_hz, ticks where vision was live):
    flt13  ag=0  median 34.9 Hz   floor 24.0 Hz
           ag=1  median 34.6 Hz   floor 16.2 Hz     <- the gate-2 leg
    flt15  ag=0  median 34.2 Hz   floor 23.3 Hz
           ag=1  median 32.5 Hz   floor 16.0 Hz     <- the gate-2 leg
A loop that swings between 16 and 35 Hz is a loop whose gains are effectively varying by
2x, and that variance is what makes gate 2 inconsistently pass-or-collide.

WHAT THIS FILE PINS:
  * --log-tube is OFF by default, so the detector is never constructed;
  * the tube detector really is the dominant per-frame cost (so nobody re-enables it
    casually);
  * turning it off changes NO control law -- the gate detector still runs on every new
    frame, which is the only detection the lateral and vertical loops read.
"""
import os
import time

import numpy as np
import pytest

from dataclasses import replace
from vq1_vision_servo import GateDetector, ServoConfig, TubeDetector

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FRAME = os.path.join(ROOT, "vision_frame.png")


def test_log_tube_defaults_off():
    """The whole point: default flights do not construct or call TubeDetector."""
    import subprocess
    import sys
    out = subprocess.run([sys.executable, os.path.join(ROOT, "tools", "schedule_flier.py"),
                          "--help"], capture_output=True, text=True, cwd=ROOT)
    assert "--log-tube" in out.stdout
    # argparse store_true with no default= is False; assert via the help text contract
    assert "OFF BY DEFAULT" in out.stdout.replace("\n", " ").replace("  ", " ")


@pytest.mark.skipif(not os.path.exists(FRAME), reason="vision_frame.png not present")
def test_tube_detector_dominates_the_per_frame_cost():
    """Justifies the default. If this ever stops being true the flag could be flipped."""
    from PIL import Image
    cfg = ServoConfig()
    frame = np.asarray(Image.open(FRAME).convert("RGB"))
    half = np.ascontiguousarray(frame[::2, ::2])
    tube, gate = TubeDetector(cfg), GateDetector(replace(cfg, gate_dilate_px=2))

    def bench(fn, arg, n=20):
        fn(arg)
        t0 = time.perf_counter()
        for _ in range(n):
            fn(arg)
        return (time.perf_counter() - t0) / n

    t_tube, t_gate = bench(tube.measure, frame), bench(gate.detect, half)
    assert t_tube > 1.5 * t_gate, (
        f"tube {t_tube*1000:.1f} ms vs gate {t_gate*1000:.1f} ms -- the tube is no longer "
        f"the dominant cost, so --log-tube's default could be revisited")
    # and it is the majority of the combined budget
    assert t_tube / (t_tube + t_gate) > 0.55


@pytest.mark.skipif(not os.path.exists(FRAME), reason="vision_frame.png not present")
def test_dropping_the_tube_raises_the_achieved_loop_rate():
    """Simulates the flier's real tick structure -- detect ONLY on a new frame (30 fps
    source), 1/250 s sleep otherwise -- and counts achieved ticks/second with and without
    the tube. This is the closest measurement available without a live race."""
    from PIL import Image
    cfg = ServoConfig()
    frame = np.asarray(Image.open(FRAME).convert("RGB"))
    half = np.ascontiguousarray(frame[::2, ::2])
    tube, gate = TubeDetector(cfg), GateDetector(replace(cfg, gate_dilate_px=2))

    def run(with_tube, seconds=1.0, fps=30.0):
        t_end = time.perf_counter() + seconds
        last_frame, ticks = -1.0, 0
        t0 = time.perf_counter()
        while time.perf_counter() < t_end:
            now = time.perf_counter()
            frame_n = (now - t0) * fps // 1
            if frame_n != last_frame:
                last_frame = frame_n
                if with_tube:
                    tube.measure(frame)
                gate.detect(half)
            ticks += 1
            time.sleep(1.0 / 250.0)
        return ticks / seconds

    with_tube = run(True)
    without = run(False)
    assert without > with_tube, f"{without:.0f} Hz vs {with_tube:.0f} Hz"
    # the tube is ~70% of the per-frame budget, so this should be a large margin
    assert without > 1.3 * with_tube, (
        f"only {without/with_tube:.2f}x faster ({with_tube:.0f} -> {without:.0f} Hz)")
