"""RAIL DETECTOR VALIDATION -- measurement only, no control wiring.

Runs the rebuilt TubeDetector over saved FPV frames, prints per-frame lock / lateral
offset / vertical (vanishing point), writes a PNG overlay with the fitted rail curves
drawn on the frame, and reports the lock rate.

WHY AN OVERLAY: the numbers alone cannot tell you whether the fit is on the REAL rails
or on some bright clutter that happens to fit a parabola. Drawing the curves back onto
the frame is the check that the detector tracks the thing a human sees.

    python tools/tube_rail_check.py                    # every saved frame
    python tools/tube_rail_check.py --shift-sweep      # + synthetic lateral shifts

The shift sweep translates a real frame by a known number of pixels and checks the
reported offset follows it. That does NOT create new viewpoints -- it cannot stand in
for real flight frames -- but it does test that the measurement has the right sign and
scale, and that the threshold survives the path sitting off-centre.
"""
import argparse
import os
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vq1_vision_servo import (ServoConfig, TubeDetector,    # noqa: E402
                              draw_rail_overlay)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# The RAW FPV frames. out.png / scratch_tube_overlay.png are excluded on purpose:
# they are old MASK RENDERS of vision_frame.png (the corridor painted flat green),
# not camera frames, so detecting "rails" in them would measure nothing real.
RAW_FRAMES = ["vision_frame.png"]
NOT_RAW = {"out.png", "scratch_tube_overlay.png", "trajectory_plot.png"}


def load(path):
    return np.asarray(Image.open(path).convert("RGB"))


def overlay(frame, m, cfg, path_out):
    """Render via the SHARED drawer (vq1_vision_servo.draw_rail_overlay), which is the
    same code the in-flight --dump-frames-leg1 writer uses, so a frame dumped mid-race
    and a frame checked here are drawn identically."""
    Image.fromarray(draw_rail_overlay(frame, m, cfg)).save(path_out)
    return path_out


def report(name, m):
    flag = "LOCK  " if m.found else "no-lock"
    print(f"  {name:<34} {flag}  offset={m.u_tube:+.4f}  vert={m.v_converge:+.4f}"
          f"{'' if m.converge_ok else '(fallback)'}  curv={m.curvature:+.4f}"
          f"  sep={m.rail_sep:.3f}  rows={m.n_rows:3d}  rms={m.fit_rms:.2f}px"
          f"  area={m.area_frac:.4f}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("frames", nargs="*", help="frames to test (default: saved raw ones)")
    ap.add_argument("--shift-sweep", action="store_true",
                    help="also test synthetic lateral shifts of each frame")
    ap.add_argument("--outdir", default=os.path.join(ROOT, "rail_overlays"))
    args = ap.parse_args()

    cfg = ServoConfig()
    det = TubeDetector(cfg)
    os.makedirs(args.outdir, exist_ok=True)

    frames = args.frames or [os.path.join(ROOT, f) for f in RAW_FRAMES]
    frames = [f for f in frames if os.path.exists(f)]
    if not frames:
        print("no frames found")
        return 1

    print(f"\n=== RAIL DETECTOR over {len(frames)} raw frame(s) ===")
    print(f"    threshold min(G,B)-R >= {cfg.rail_score_min:.0f}, "
          f"max(G,B) >= {cfg.rail_val_min:.0f}; look-ahead row "
          f"{cfg.rail_lookahead:.2f}H; NO area gate")
    locks = 0
    for f in frames:
        m = det.measure(load(f))
        locks += bool(m.found)
        report(os.path.basename(f), m)
        out = overlay(load(f), m, cfg,
                      os.path.join(args.outdir,
                                   os.path.basename(f).replace(".png", "_overlay.png")))
        print(f"  {'':<34} overlay -> {os.path.relpath(out, ROOT)}")

    total, total_lock = len(frames), locks
    if args.shift_sweep:
        print("\n--- synthetic lateral shifts (offset must follow the shift) ---")
        print("    (same viewpoint, translated: tests sign/scale + threshold margin,")
        print("     NOT a substitute for real frames from other parts of the course)")
        for f in frames:
            base = load(f)
            W = base.shape[1]
            m0 = det.measure(base)
            for px in (-120, -80, -40, 40, 80, 120):
                sh = np.roll(base, px, axis=1)
                # np.roll wraps; blank the wrapped edge so it cannot fake a rail
                if px > 0:
                    sh[:, :px] = 0
                else:
                    sh[:, px:] = 0
                m = det.measure(sh)
                total += 1
                total_lock += bool(m.found)
                exp = m0.u_tube + 2.0 * px / W       # normalised units
                err = m.u_tube - exp
                ok = "ok " if (m.found and abs(err) < 0.05) else "BAD"
                print(f"  shift {px:+5d}px  {ok}  offset={m.u_tube:+.4f} "
                      f"expected={exp:+.4f}  err={err:+.4f}  "
                      f"{'LOCK' if m.found else 'no-lock'}")

    print(f"\nLOCK RATE: {total_lock}/{total} = {100.0 * total_lock / total:.1f}%")
    print("NOTE: only one genuine raw FPV frame is saved in this repo "
          "(out.png / scratch_tube_overlay.png are old mask renders of it), so the "
          "raw-frame lock rate is over n=1. Capture frames from the gate-3 leg to "
          "make this number mean anything about the flight.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
