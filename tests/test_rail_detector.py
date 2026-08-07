"""RAIL DETECTOR -- the course path is two cyan LINES, and area is not evidence.

THE DEFECT BEING FIXED. The old detector masked cyan and asked what FRACTION of the
frame it covered. That is the wrong question about a thin bright curve: a rail that is
one or two pixels wide is completely visible and still has essentially zero area, so the
area test returned "not found" on frames where the path is plainly visible to a human.
The sub-4% lock rate on the gate-3 leg was therefore a FALSE NEGATIVE -- a broken
question, not a weak signal.

WHAT THE REBUILD DOES: threshold on CHROMA (min(G,B) - R, which is ~0 on grey scenery at
any brightness and large only on saturated cyan), take the OUTERMOST run per row as the
two rail boundaries, and fit each rail as a robust quadratic x = f(y). Validity is then
fit quality + rail separation + row count. area_frac is still reported, and gates
nothing.

Measured on the one saved raw FPV frame: LOCK, 197 rows, 0.89 px fit residual, lateral
offset +0.003 (the drone is centred), rails converging at v=-0.35.
"""
import os

import numpy as np
import pytest
from PIL import Image

from vq1_vision_servo import ServoConfig, TubeDetector

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FRAME = os.path.join(ROOT, "vision_frame.png")

real_frame = pytest.mark.skipif(not os.path.exists(FRAME),
                                reason="vision_frame.png not present")


def frame():
    return np.asarray(Image.open(FRAME).convert("RGB"))


def synth_rails(w=640, h=360, x_conv=0.5, halfspan=0.42, width=3, y_conv=0.15,
               bg=25, colour=(40, 230, 255)):
    """Two THIN cyan rails converging at (x_conv*w, y_conv*h) -- the real feature, with
    none of the area a fill would have. Rail width defaults to 3 px, i.e. ~0.5% of the
    frame masked: an area threshold of any useful size rejects this outright."""
    img = np.full((h, w, 3), bg, np.uint8)
    yv, xc = y_conv * h, x_conv * w
    for y in range(int(yv), h):
        t = (y - yv) / max(h - yv, 1)
        for sgn in (-1, 1):
            x = int(xc + sgn * halfspan * w * t)
            x0, x1 = x - width // 2, x + width // 2 + 1
            if x1 <= 0 or x0 >= w:
                continue          # rail is off-frame on this row -- draw nothing.
            #                       (Clipping with max(0, x0) alone would let a negative
            #                       x1 slice backwards and paint half the row.)
            img[y, max(0, x0):min(w, x1)] = colour
    return img


# --------------------------------------------------------------------------------
# The headline: thin rails lock, and they do so at an area that fails any area gate
# --------------------------------------------------------------------------------
def test_thin_rails_lock_despite_negligible_area():
    """THE WHOLE POINT. Rails 3 px wide mask well under 1% of the frame. The old
    --tube-steer-area-min was 0.015; this locks with area far below it."""
    m = TubeDetector(ServoConfig()).measure(synth_rails())
    assert m.found, "thin rails must LOCK -- this is the false negative being fixed"
    assert m.area_frac < 0.015, (
        f"area {m.area_frac:.4f} -- if thin rails now exceed the old area gate this "
        f"test no longer demonstrates anything")
    assert abs(m.u_tube) < 0.02, f"centred rails should read ~0 offset, got {m.u_tube:+.3f}"


def test_lateral_offset_tracks_the_path_sideways_with_correct_sign():
    """+ offset must mean the path is RIGHT of frame centre, and scale sensibly."""
    det = TubeDetector(ServoConfig())
    # halfspan 0.30 keeps BOTH rails inside the frame at these offsets, so this measures
    # the offset itself rather than how the detector handles a rail leaving the frame.
    left = det.measure(synth_rails(x_conv=0.35, halfspan=0.30))
    right = det.measure(synth_rails(x_conv=0.65, halfspan=0.30))
    assert left.found and right.found
    assert left.u_tube < -0.05 and right.u_tube > 0.05
    assert left.u_tube == pytest.approx(-right.u_tube, abs=0.02)   # symmetric


def test_locks_while_a_rail_runs_off_the_edge_of_the_frame():
    """The case that matters most in flight: the drone is off to one side, so the near
    end of one rail has left the frame. Rows that show only one rail are skipped rather
    than half-fitted, and the lock survives on the rows that show both -- degrading in
    row count, not in accuracy. A detector that dropped the lock here would go blind
    exactly when the steering error is largest."""
    det = TubeDetector(ServoConfig())
    for x_conv, expect in ((0.20, -0.60), (0.35, -0.30), (0.75, +0.50)):
        m = det.measure(synth_rails(x_conv=x_conv, halfspan=0.42))
        assert m.found, f"lost lock at x_conv={x_conv} (rows={m.n_rows})"
        assert m.u_tube == pytest.approx(expect, abs=0.03), (
            f"x_conv={x_conv}: offset {m.u_tube:+.3f}, expected ~{expect:+.2f}")
        assert m.fit_rms < 2.0


def test_vanishing_point_moves_with_the_convergence_row():
    """The vertical output is the row where the rails MEET. Push the convergence down
    the frame and the reported value must follow, positive = below centre."""
    det = TubeDetector(ServoConfig())
    high = det.measure(synth_rails(y_conv=0.10))
    low = det.measure(synth_rails(y_conv=0.40))
    assert high.found and low.found
    assert high.v_converge < low.v_converge
    assert high.converge_ok and low.converge_ok
    # and it is roughly the right row: v is normalised (y - H/2)/(H/2)
    assert low.v_converge == pytest.approx(2 * 0.40 - 1.0, abs=0.15)


def test_no_lock_on_an_empty_frame():
    dark = np.full((360, 640, 3), 20, np.uint8)
    assert not TubeDetector(ServoConfig()).measure(dark).found


def test_grey_clutter_does_not_lock():
    """The chroma test must reject BRIGHT GREY -- buildings and the horizon grid are
    bright but unsaturated, and a value/luminance threshold would admit them."""
    img = np.full((360, 640, 3), 20, np.uint8)
    rng = np.random.default_rng(0)
    for _ in range(60):
        y, x = int(rng.integers(0, 340)), int(rng.integers(0, 620))
        v = int(rng.integers(180, 255))
        img[y:y + 18, x:x + 18] = (v, v, v)          # bright, but R == G == B
    assert not TubeDetector(ServoConfig()).measure(img).found


def test_the_red_gate_does_not_lock():
    """The gate is the other bright thing in frame. It must not read as rails."""
    img = np.full((360, 640, 3), 20, np.uint8)
    img[120:240, 260:380] = (235, 40, 40)
    assert not TubeDetector(ServoConfig()).measure(img).found


def test_a_single_bright_streak_is_not_a_path():
    """One rail is not a corridor: without two separated boundaries there is no centre
    to steer to, so this must NOT lock."""
    img = np.full((360, 640, 3), 20, np.uint8)
    img[100:360, 300:304] = (40, 230, 255)
    m = TubeDetector(ServoConfig()).measure(img)
    assert not m.found, f"a lone streak locked (sep={m.rail_sep:.3f})"


def test_clutter_beside_the_rails_is_rejected_by_the_robust_fit():
    """A few bright cyan blobs off to the side must not bend the fit -- that is what
    the MAD rejection is for. The offset must stay put."""
    det = TubeDetector(ServoConfig())
    clean = det.measure(synth_rails())
    dirty_img = synth_rails()
    rng = np.random.default_rng(1)
    for _ in range(12):                      # speckle well outside the corridor
        y, x = int(rng.integers(120, 350)), int(rng.integers(0, 60))
        dirty_img[y:y + 4, x:x + 4] = (40, 230, 255)
    dirty = det.measure(dirty_img)
    assert dirty.found, "clutter broke the lock entirely"
    assert abs(dirty.u_tube - clean.u_tube) < 0.03, (
        f"clutter moved the offset {clean.u_tube:+.3f} -> {dirty.u_tube:+.3f}")


def test_curvature_sign_says_which_way_the_course_bends():
    """A path whose far end sits right of its near end bends RIGHT ahead."""
    img = np.full((360, 640, 3), 20, np.uint8)
    h, w = 360, 640
    for y in range(120, h):
        t = (y - 120) / (h - 120)
        centre = 0.72 - 0.22 * t              # far (small y) is RIGHT of near
        for sgn in (-1, 1):
            x = int(centre * w + sgn * 0.35 * w * t)
            img[y, max(0, x - 2):x + 2] = (40, 230, 255)
    m = TubeDetector(ServoConfig()).measure(img)
    assert m.found and m.curvature > 0.05, f"curvature {m.curvature:+.3f} should be > 0"


# --------------------------------------------------------------------------------
# The real frame
# --------------------------------------------------------------------------------
@real_frame
def test_real_frame_locks_and_is_well_fitted():
    m = TubeDetector(ServoConfig()).measure(frame())
    assert m.found, "the one real FPV frame we have must lock"
    assert m.fit_rms < 3.0, f"fit residual {m.fit_rms:.2f} px -- not tracking real rails"
    assert m.n_rows > 100, f"only {m.n_rows} rows contributed"
    assert m.rail_sep > 0.2, f"rail separation {m.rail_sep:.3f} implausibly small"


@real_frame
def test_real_frame_reads_as_centred_on_the_path():
    """Eyeballed against rail_overlays/vision_frame_overlay.png: the drone sits on the
    centreline in this frame, so the offset must be ~0. A detector that locked onto one
    rail, or onto the corridor haze, would not land here."""
    m = TubeDetector(ServoConfig()).measure(frame())
    assert abs(m.u_tube) < 0.05, f"offset {m.u_tube:+.4f} on a visually centred frame"


@real_frame
def test_real_frame_area_would_have_failed_a_solidity_gate_on_the_gate3_leg():
    """Context for why area was abandoned. Even on this GOOD frame -- rails filling the
    lower frame, path dead ahead -- the masked area is small; on the gate-3 leg it
    measured 0.0000 at the median. Validity cannot rest on that number."""
    m = TubeDetector(ServoConfig()).measure(frame())
    assert m.found
    assert m.area_frac < 0.10


@real_frame
def test_offset_follows_a_known_lateral_shift():
    """Translate the real frame by a known amount: the reported offset must move by the
    same amount in normalised units. This checks sign AND scale on real pixels."""
    det = TubeDetector(ServoConfig())
    base = frame()
    W = base.shape[1]
    m0 = det.measure(base)
    for px in (-100, -50, 50, 100):
        sh = np.roll(base, px, axis=1)
        if px > 0:
            sh[:, :px] = 0
        else:
            sh[:, px:] = 0
        m = det.measure(sh)
        assert m.found, f"lost lock at shift {px:+d}px"
        assert m.u_tube == pytest.approx(m0.u_tube + 2.0 * px / W, abs=0.02), (
            f"shift {px:+d}px: offset {m.u_tube:+.4f}")


@real_frame
def test_half_resolution_measures_the_same_path():
    """The detector must be resolution-invariant, so the loop can buy it cheaply."""
    det = TubeDetector(ServoConfig())
    full, half = det.measure(frame()), det.measure(np.ascontiguousarray(frame()[::2, ::2]))
    assert full.found and half.found
    assert full.u_tube == pytest.approx(half.u_tube, abs=0.01)
    assert full.v_converge == pytest.approx(half.v_converge, abs=0.05)


def test_the_lateral_law_is_disarmed_pending_validation():
    """DISCIPLINE. The detector changed what u_tube MEANS, so the steering law tuned on
    the old signal must not fly until its gains are re-derived. --tube-lateral parses but
    refuses to run."""
    import subprocess
    import sys
    r = subprocess.run([sys.executable, os.path.join(ROOT, "tools", "schedule_flier.py"),
                        "--coast-tube", "--tube-lateral", "--go-timeout", "1"],
                       capture_output=True, text=True, cwd=ROOT, timeout=120)
    assert r.returncode != 0, "--tube-lateral must refuse to fly while disarmed"
    assert "DISARMED" in (r.stdout + r.stderr)
