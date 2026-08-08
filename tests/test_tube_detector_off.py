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
  * what the detector now COSTS, measured -- see the cost test, whose finding has
    reversed since the rail rebuild: it is no longer the dominant per-frame cost, and
    at half resolution it is cheaper than the gate detection the loop already pays for;
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
# Fixture lives with the tests: the source frame sits in archive/frames/, which is
# gitignored bulk data, so a clone would otherwise skip every detector test.
FRAME = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                     "fixtures", "vision_frame.png")


def test_log_tube_defaults_off():
    """The whole point: default flights do not construct or call TubeDetector."""
    import subprocess
    import sys
    out = subprocess.run([sys.executable, os.path.join(ROOT, "tools", "schedule_flier.py"),
                          "--help"], capture_output=True, text=True, cwd=ROOT)
    assert "--log-tube" in out.stdout
    # argparse store_true with no default= is False; assert via the help text contract
    assert "OFF BY DEFAULT" in out.stdout.replace("\n", " ").replace("  ", " ")


def _bench(fn, arg, n=20):
    fn(arg)
    t0 = time.perf_counter()
    for _ in range(n):
        fn(arg)
    return (time.perf_counter() - t0) / n


@pytest.mark.skipif(not os.path.exists(FRAME), reason="vision_frame.png not present")
def test_the_rail_detector_is_no_longer_the_dominant_per_frame_cost():
    """THE COST ARGUMENT HAS EXPIRED -- recorded here rather than left as a stale claim.

    The old fill-and-area detector ran an HSV conversion over the full frame and cost
    13.5 ms against the gate detector's 5.5, which is what made --log-tube expensive
    enough to default off and what drove the 16 Hz loop floor. The rebuilt RAIL detector
    thresholds on an integer chroma difference instead -- no HSV -- and fits two
    quadratics over a few hundred row samples: measured on this frame, 8.0 ms at full
    resolution and 3.4 ms at half, against the gate detector's 6.1 ms.

    So at half resolution the rail detector is now CHEAPER than the gate detection the
    loop already pays for on every frame. That does not by itself justify flipping
    --log-tube's default (that is a flight decision, and the detector is still
    validation-only), but the "71% of the vision budget" reasoning no longer applies and
    must not be quoted as if it did.
    """
    from PIL import Image
    cfg = ServoConfig()
    frame = np.asarray(Image.open(FRAME).convert("RGB"))
    half = np.ascontiguousarray(frame[::2, ::2])
    tube, gate = TubeDetector(cfg), GateDetector(replace(cfg, gate_dilate_px=2))

    t_full = _bench(tube.measure, frame)
    t_half = _bench(tube.measure, half)
    t_gate = _bench(gate.detect, half)
    assert t_half < t_gate, (
        f"rail@half {t_half*1000:.1f} ms vs gate {t_gate*1000:.1f} ms -- the rail "
        f"detector was measured cheaper than the gate detector; if that regressed, the "
        f"cost of running it every frame needs re-arguing")
    assert t_full < 2.0 * t_gate, (
        f"rail@full {t_full*1000:.1f} ms vs gate {t_gate*1000:.1f} ms")


@pytest.mark.skipif(not os.path.exists(FRAME), reason="vision_frame.png not present")
def test_the_rail_detector_is_resolution_invariant():
    """Half-res is only a usable economy if it measures the same thing. Both must LOCK
    and agree on the lateral offset -- the quantity any future steering law would read."""
    from PIL import Image
    cfg = ServoConfig()
    frame = np.asarray(Image.open(FRAME).convert("RGB"))
    half = np.ascontiguousarray(frame[::2, ::2])
    det = TubeDetector(cfg)
    a, b = det.measure(frame), det.measure(half)
    assert a.found and b.found, "the rail lock must survive half-resolution"
    assert abs(a.u_tube - b.u_tube) < 0.01, (
        f"offset disagrees across resolution: {a.u_tube:+.4f} vs {b.u_tube:+.4f}")


@pytest.mark.skipif(not os.path.exists(FRAME), reason="vision_frame.png not present")
def test_dropping_the_tube_raises_the_achieved_loop_rate():
    """Simulates the flier's real tick structure -- detect ONLY on a new frame (30 fps
    source), 1/250 s sleep otherwise -- and counts achieved ticks/second with the rail
    detector off, at half resolution, and at full. The closest measurement available
    without a live race, and what justifies half res being the default under
    --rail-vert, which runs the detector every frame on the gate-2 approach."""
    from PIL import Image
    cfg = ServoConfig()
    frame = np.asarray(Image.open(FRAME).convert("RGB"))
    half = np.ascontiguousarray(frame[::2, ::2])
    tube, gate = TubeDetector(cfg), GateDetector(replace(cfg, gate_dilate_px=2))

    def run(mode, seconds=1.0, fps=30.0):
        """mode: 'none' | 'half' | 'full' -- what the rail detector is fed, if anything."""
        t_end = time.perf_counter() + seconds
        last_frame, ticks = -1.0, 0
        t0 = time.perf_counter()
        while time.perf_counter() < t_end:
            now = time.perf_counter()
            frame_n = (now - t0) * fps // 1
            if frame_n != last_frame:
                last_frame = frame_n
                if mode == "full":
                    tube.measure(frame)
                elif mode == "half":
                    tube.measure(half)
                gate.detect(half)
            ticks += 1
            time.sleep(1.0 / 250.0)
        return ticks / seconds

    # BEST OF N, interleaved. This is a wall-clock benchmark, so a background process
    # stealing the CPU during one arm makes it report a difference that is not there --
    # measured, that is exactly how it fails when the whole suite runs in parallel
    # (38 vs 46 Hz against ~157/132 standalone). Taking the best sample of each arm
    # measures the machine at its least contended, and interleaving keeps a slow patch
    # from landing entirely on one arm.
    best = {"none": 0.0, "half": 0.0, "full": 0.0}
    for _ in range(3):
        for mode in ("none", "half", "full"):
            best[mode] = max(best[mode], run(mode, seconds=0.4))
    none, half_res, full_res = best["none"], best["half"], best["full"]
    # The half-vs-full claim is taken from the DIRECT per-call bench rather than from the
    # simulated loop: a tight timing loop is far less sensitive to another process
    # stealing the CPU than a wall-clock rate is, and the achieved-rate version of this
    # comparison failed intermittently under full-suite load even at best-of-3.
    t_half_call, t_full_call = _bench(tube.measure, half), _bench(tube.measure, frame)
    # Running the rail detector still costs SOMETHING -- it is not free, and a change
    # that made it free would mean it stopped running.
    assert none > half_res > 0, f"{none:.0f} vs {half_res:.0f} Hz"
    # HALF RES IS THE POINT: --rail-vert runs this every frame on the g1->g2 leg, which
    # is the gate-2 approach, so the resolution default is a flight-safety choice and
    # not a micro-optimisation. Measured: ~179 Hz clean, ~157 half, ~132 full.
    assert t_half_call < t_full_call, (
        f"half res ({t_half_call*1000:.1f} ms/call) is not cheaper than full "
        f"({t_full_call*1000:.1f} ms) -- the --rail-full-res default would need "
        f"rethinking")
    assert full_res > 0, "the full-res arm did not run"
    # The "how much loop rate does it cost" claim is asserted from the PER-CALL bench,
    # not from the achieved rate. An achieved-rate ratio is only meaningful on an idle
    # box: with the simulator running on the same machine the absolute rates collapse
    # (measured 72-92 Hz against ~157 Hz standalone) and the ratio fails for reasons
    # that have nothing to do with the detector. The per-call cost is the invariant --
    # it is what actually determines the loop impact, and it is stable under load.
    assert t_half_call < 0.5 * (1.0 / 30.0), (
        f"rail detector at half res is {t_half_call*1000:.1f} ms/call, more than half a "
        f"30 fps frame budget; it was ~3.4 ms when the half-res default was chosen")
