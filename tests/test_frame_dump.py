"""FRAME DUMP (--dump-frames-leg1) -- the diagnostic that answers "why does the rail
lock collapse on the g1->g2 leg?" with pictures instead of inference.

TWO PROPERTIES MATTER AND BOTH ARE LOAD-BEARING:

1. IT MUST NOT TOUCH THE FLIGHT. A 640x360 PNG costs ~29.5 ms to encode and each sample
   is two of them plus a detector call -- ~68 ms against a ~28 ms tick. Written inline
   that stalls the control loop for more than two ticks, on the g1->g2 leg, which IS the
   gate-2 approach: the leg whose loop-rate variance (35 Hz median, 16 Hz floor) already
   makes gate 2 inconsistently pass-or-collide. Dumping frames must not be the reason a
   flight fails, or the frames describe a flight nobody wanted to fly. So the control
   loop only hands over a reference, and a full queue DROPS rather than blocks.

2. IT MUST CATCH THE TRANSITION. The interesting moment is the last second BEFORE gate 1
   and the first frames after. Those approach frames are buffered and written
   retroactively, once active_gate actually reaches 1 -- not guessed at from a rising
   size_frac, which fires on every glimpse of a gate and misses the transition whenever
   the detector blinks.
"""
import os
import time

import numpy as np
import pytest
from PIL import Image

from vq1_vision_servo import ServoConfig, TubeDetector, draw_rail_overlay
from tools.schedule_flier import FrameDumper, build_parser

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# Fixture lives with the tests: the source frame sits in archive/frames/, which is
# gitignored bulk data, so a clone would otherwise skip every detector test.
FRAME = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                     "fixtures", "vision_frame.png")

real_frame = pytest.mark.skipif(not os.path.exists(FRAME),
                                reason="vision_frame.png not present")


def a_frame():
    return np.asarray(Image.open(FRAME).convert("RGB"))


def drain(d, timeout=20.0):
    return d.stop(drain_s=timeout)


# ---------------------------------------------------------------------------------
# Discipline: off by default, log-only
# ---------------------------------------------------------------------------------
def test_frame_dump_is_off_by_default():
    d = vars(build_parser().parse_args([]))
    assert d["dump_frames_leg1"] is False
    assert d["dump_frames_every"] == 0.25
    assert d["dump_frames_pre"] == 1.0
    assert d["dump_frames_dir"] == "leg1_frames"


# ---------------------------------------------------------------------------------
# It must not stall the control loop
# ---------------------------------------------------------------------------------
@real_frame
def test_offer_is_cheap_enough_for_the_control_loop(tmp_path):
    """offer() runs on the flight's hot path. It must be a clock check and a queue put --
    orders of magnitude below the ~29.5 ms it costs to actually write the PNG."""
    d = FrameDumper(str(tmp_path), ServoConfig(), every_s=0.0)
    frame = a_frame()
    d.offer("ag1", 0.0, frame)                       # warm
    t0 = time.perf_counter()
    for i in range(200):
        d.offer("ag1", i * 0.001, frame)
    per = (time.perf_counter() - t0) / 200
    assert per < 0.002, f"offer() took {per*1000:.2f} ms/call on the control loop"


@real_frame
def test_a_full_queue_drops_instead_of_blocking(tmp_path):
    """THE SAFETY PROPERTY. The writer is never started here, so the queue fills and
    stays full: offer() must return promptly and count drops rather than block the
    flight waiting on a disk."""
    d = FrameDumper(str(tmp_path), ServoConfig(), every_s=0.0, queue_max=4)
    frame = a_frame()
    t0 = time.perf_counter()
    accepted = sum(d.offer("ag1", i * 0.01, frame) for i in range(50))
    elapsed = time.perf_counter() - t0
    assert accepted == 4, f"queue cap not honoured: {accepted} accepted"
    assert d.dropped == 46
    assert elapsed < 0.5, f"offer() blocked for {elapsed:.2f}s on a full queue"


@real_frame
def test_the_rate_limit_holds(tmp_path):
    """--dump-frames-every must actually bound the sample rate, or the writer drowns."""
    d = FrameDumper(str(tmp_path), ServoConfig(), every_s=0.25, queue_max=1000)
    frame = a_frame()
    # 4 s of 30 fps frames offered -> ~16 accepted at 0.25 s spacing, not 120
    accepted = sum(d.offer("ag1", i / 30.0, frame) for i in range(120))
    assert 14 <= accepted <= 18, f"{accepted} samples accepted from 4 s at 0.25 s spacing"


# ---------------------------------------------------------------------------------
# It must catch the transition through gate 1
# ---------------------------------------------------------------------------------
@real_frame
def test_pre_gate_frames_are_written_only_once_gate_1_is_reached(tmp_path):
    """The ring holds the approach; nothing is committed until ag actually reaches 1."""
    d = FrameDumper(str(tmp_path), ServoConfig(), every_s=0.25, pre_s=1.0)
    frame = a_frame()
    for i in range(60):                       # 3 s of approach at 20 fps
        d.hold_pre_gate(i * 0.05, frame)
    assert d.q.qsize() == 0, "approach frames were written before gate 1 was passed"
    n = d.flush_pre_gate()
    assert n >= 4, f"only {n} approach frames committed"
    assert d.q.qsize() == n


@real_frame
def test_the_ring_keeps_the_LAST_second_not_the_first(tmp_path):
    """It must be the second BEFORE the gate. A ring that kept the oldest frames would
    hand back the start of the approach, which is the least interesting moment."""
    d = FrameDumper(str(tmp_path), ServoConfig(), every_s=0.25, pre_s=1.0)
    frame = a_frame()
    for i in range(100):                      # 5 s of approach
        d.hold_pre_gate(i * 0.05, frame)
    times = [t for t, _ in d.ring]
    assert times, "ring is empty"
    assert min(times) > 3.0, (
        f"ring holds t={min(times):.2f}s onward -- that is the START of the approach, "
        f"not the last second before the gate")
    assert max(times) == pytest.approx(4.95, abs=0.3)


# ---------------------------------------------------------------------------------
# What it writes
# ---------------------------------------------------------------------------------
@real_frame
def test_writes_raw_and_overlay_pairs_plus_a_manifest(tmp_path):
    d = FrameDumper(str(tmp_path), ServoConfig(), every_s=0.0)
    d.start()
    frame = a_frame()
    for i in range(3):
        d.offer("ag1", i * 0.25, frame)
    wrote, dropped = drain(d)
    assert (wrote, dropped) == (3, 0)

    files = os.listdir(str(tmp_path))
    assert "manifest.csv" in files
    raw = sorted(f for f in files if f.endswith("_raw.png"))
    ovl = sorted(f for f in files if f.endswith("_ovl.png"))
    assert len(raw) == len(ovl) == 3, f"expected 3 pairs, got {len(raw)}/{len(ovl)}"

    # the RAW file is the untouched frame -- it must not have the overlay drawn on it
    assert np.array_equal(np.asarray(Image.open(os.path.join(tmp_path, raw[0]))), frame)
    # and the overlay must differ from it
    over = np.asarray(Image.open(os.path.join(tmp_path, ovl[0])))
    assert not np.array_equal(over, frame)

    body = open(os.path.join(tmp_path, "manifest.csv")).read().strip().splitlines()
    assert body[0].startswith("t,tag,lock,u_tube,v_converge")
    assert len(body) == 4                                  # header + 3 samples
    assert body[1].split(",")[2] == "1", "the real frame should log as locked"


@real_frame
def test_filenames_carry_the_lock_state_so_the_folder_can_be_scanned(tmp_path):
    """The point of the dump is spotting WHERE the lock dies. Both outcomes must be
    distinguishable without opening a single file."""
    d = FrameDumper(str(tmp_path), ServoConfig(), every_s=0.0)
    d.start()
    d.offer("ag1", 1.0, a_frame())                              # locks
    d.offer("ag1", 2.0, np.full((360, 640, 3), 20, np.uint8))   # cannot lock
    drain(d)
    names = os.listdir(str(tmp_path))
    assert any("t0001.000_LOCK_" in n for n in names), names
    assert any("t0002.000_NOLOCK_" in n for n in names), names


@real_frame
def test_the_dumped_overlay_is_the_same_drawing_the_offline_tool_produces(tmp_path):
    """One renderer, so a frame dumped mid-race and a frame checked offline are directly
    comparable -- otherwise 'it looked fine in the tool' proves nothing about the race."""
    d = FrameDumper(str(tmp_path), ServoConfig(), every_s=0.0)
    d.start()
    frame = a_frame()
    d.offer("ag1", 0.0, frame)
    drain(d)
    ovl = [f for f in os.listdir(str(tmp_path)) if f.endswith("_ovl.png")][0]
    written = np.asarray(Image.open(os.path.join(tmp_path, ovl)))
    cfg = ServoConfig()
    expected = draw_rail_overlay(frame, TubeDetector(cfg).measure(frame), cfg)
    assert np.array_equal(written, expected)


@real_frame
def test_the_overlay_never_modifies_the_frame_the_flight_is_using(tmp_path):
    """The writer shares the control loop's frame array by reference (get_latest_frame
    already hands out a private copy). If the drawer mutated it in place, the flight
    would be detecting on a frame with green lines painted across it."""
    frame = a_frame()
    before = frame.copy()
    cfg = ServoConfig()
    draw_rail_overlay(frame, TubeDetector(cfg).measure(frame), cfg)
    assert np.array_equal(frame, before), "draw_rail_overlay mutated its input"


def test_lock_state_shows_in_the_overlay_without_reading_a_filename():
    """Green band = lock, red = not. The at-a-glance signal when flicking through a
    folder of a few hundred frames."""
    cfg = ServoConfig()
    det = TubeDetector(cfg)
    dark = np.full((360, 640, 3), 20, np.uint8)
    bad = draw_rail_overlay(dark, det.measure(dark), cfg)
    assert tuple(bad[0, 0]) == (255, 60, 60)
    if os.path.exists(FRAME):
        good = draw_rail_overlay(a_frame(), det.measure(a_frame()), cfg)
        assert tuple(good[0, 0]) == (0, 255, 0)
