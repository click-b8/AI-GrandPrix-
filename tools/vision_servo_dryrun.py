#!/usr/bin/env python3
"""Offline dry-run + calibration for the hardcoded vision-servo controller.

NO sim, NO network — feed it a frame and it prints what the detector sees and
what the controller would command. This is the tool you use ON THE SIM BOX to
calibrate gate detection: capture a real FPV frame (tools/imu_check.py saves one
as vision_frame.png), run this against it, and adjust the HSV thresholds until
the overlay masks the gate and the centroid crosshair sits on it.

    # against a real saved frame:
    python tools/vision_servo_dryrun.py --image vision_frame.png --overlay out.png

    # tweak thresholds live until the overlay looks right:
    python tools/vision_servo_dryrun.py --image f.png --hue-lo 5 --hue-hi 45 \
        --sat-min 0.45 --val-min 0.35 --overlay out.png

    # no frame handy? synthesise one to sanity-check the pipeline:
    python tools/vision_servo_dryrun.py --synthetic 0.75 0.4

Prints: detection (found, u_err, v_err, size, area) and the command dict
(throttle/roll/pitch/yaw) for a LEVEL IMU. Signs: u_err + = gate right,
v_err + = gate low.
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vq1_vision_servo import ServoConfig, VisionServoController, TubeDetector  # noqa: E402


def load_image(path):
    from PIL import Image
    img = Image.open(path).convert("RGB")
    return np.array(img, dtype=np.uint8)


def synthetic(cx_frac, cy_frac, w=640, h=360):
    frame = np.full((h, w, 3), 25, dtype=np.uint8)
    half = int(0.18 * h / 2)
    cx, cy = int(cx_frac * w), int(cy_frac * h)
    frame[max(0, cy - half):cy + half, max(0, cx - half):cx + half] = (255, 140, 0)
    return frame


def write_overlay(frame, det, mask, out_path):
    from PIL import Image
    vis = (frame.astype(np.float32) * 0.45).astype(np.uint8)   # dim original
    vis[mask] = (0, 255, 0)                                     # masked -> green
    if det.found:
        cx, cy = int(det.cx), int(det.cy)
        h, w = frame.shape[0], frame.shape[1]
        vis[max(0, cy - 6):cy + 6, max(0, cx - 1):cx + 2] = (255, 0, 0)  # crosshair
        vis[max(0, cy - 1):cy + 2, max(0, cx - 6):cx + 6] = (255, 0, 0)
        vis[h // 2 - 1:h // 2 + 2, :] = np.maximum(vis[h // 2 - 1:h // 2 + 2, :], 60)
        vis[:, w // 2 - 1:w // 2 + 2] = np.maximum(vis[:, w // 2 - 1:w // 2 + 2], 60)
    Image.fromarray(vis).save(out_path)
    print(f"[overlay] wrote {out_path}  (green=masked, red=centroid, grey=image centre)")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--image", help="path to a real FPV frame (PNG/JPG)")
    src.add_argument("--synthetic", nargs=2, type=float, metavar=("CX", "CY"),
                     help="synthesise a gate at fractional (cx, cy), e.g. 0.75 0.4")
    ap.add_argument("--overlay", help="write a mask/centroid overlay PNG here")
    ap.add_argument("--tube", action="store_true",
                    help="measure the CYAN GUIDANCE TUBE (TubeDetector) instead of the gate")
    # threshold overrides (all optional; default to ServoConfig)
    ap.add_argument("--hue-lo", type=float)
    ap.add_argument("--hue-hi", type=float)
    ap.add_argument("--sat-min", type=float)
    ap.add_argument("--val-min", type=float)
    # tube-band overrides (only used with --tube)
    ap.add_argument("--tube-hue-lo", type=float)
    ap.add_argument("--tube-hue-hi", type=float)
    ap.add_argument("--tube-sat-min", type=float)
    ap.add_argument("--tube-val-min", type=float)
    ap.add_argument("--dim", type=float, default=1.0,
                    help="scale frame brightness by this (robustness check, e.g. 0.7)")
    ap.add_argument("--no-bright", action="store_true",
                    help="disable the bright/near-white fallback mask")
    args = ap.parse_args()

    cfg = ServoConfig()
    if args.hue_lo is not None: cfg.gate_hue_lo = args.hue_lo
    if args.hue_hi is not None: cfg.gate_hue_hi = args.hue_hi
    if args.sat_min is not None: cfg.gate_sat_min = args.sat_min
    if args.val_min is not None: cfg.gate_val_min = args.val_min
    if args.no_bright: cfg.gate_use_brightness_fallback = False
    if args.tube_hue_lo is not None: cfg.tube_hue_lo = args.tube_hue_lo
    if args.tube_hue_hi is not None: cfg.tube_hue_hi = args.tube_hue_hi
    if args.tube_sat_min is not None: cfg.tube_sat_min = args.tube_sat_min
    if args.tube_val_min is not None: cfg.tube_val_min = args.tube_val_min

    frame = load_image(args.image) if args.image else synthetic(*args.synthetic)
    if args.dim != 1.0:
        frame = np.clip(frame.astype(np.float32) * args.dim, 0, 255).astype(np.uint8)

    if args.tube:
        td = TubeDetector(cfg)
        m = td.measure(frame)
        print(f"frame: {frame.shape[1]}x{frame.shape[0]}  dim={args.dim}  "
              f"TUBE band: hue[{cfg.tube_hue_lo},{cfg.tube_hue_hi}] "
              f"sat>={cfg.tube_sat_min} val>={cfg.tube_val_min}")
        print(f"\nTUBE: found={m.found}  area_frac={m.area_frac:.4f}")
        print(f"  u_lower={m.u_lower:+.3f} (+=tube right of us)   "
              f"u_upper={m.u_upper:+.3f}   curvature={m.curvature:+.3f} (+=bends right ahead)")
        if args.overlay:
            import types
            write_overlay(frame, types.SimpleNamespace(found=False), td._mask(frame), args.overlay)
        return
    print(f"frame: {frame.shape[1]}x{frame.shape[0]}  "
          f"thresholds: hue[{cfg.gate_hue_lo},{cfg.gate_hue_hi}] "
          f"sat>={cfg.gate_sat_min} val>={cfg.gate_val_min} "
          f"bright_fallback={cfg.gate_use_brightness_fallback}")

    ctl = VisionServoController(cfg)
    det = ctl.detector.detect(frame)
    mask = ctl.detector._mask(frame)
    print(f"\nDETECTION: found={det.found}")
    if det.found:
        print(f"  u_err={det.u_err:+.3f} (+right)   v_err={det.v_err:+.3f} (+low)")
        print(f"  size_frac={det.size_frac:.3f}   area_frac={det.area_frac:.4f}"
              f"   centroid=({det.cx:.0f},{det.cy:.0f})")
    else:
        print(f"  (mask covered {mask.mean()*100:.2f}% of pixels; "
              f"below min_area_frac={cfg.min_area_frac} or empty)")

    level = {"gravity_frd": (0.0, 0.0, 9.81), "velocity": (0.0, 0.0, 0.0)}
    cmd = ctl.command(frame, level, active_gate=0)
    print(f"\nCOMMAND (level IMU): throttle={cmd['throttle']:.3f}  "
          f"roll={cmd['roll']:+.3f}  pitch={cmd['pitch']:+.3f}  yaw={cmd['yaw']:+.3f}")

    if args.overlay:
        write_overlay(frame, det, mask, args.overlay)


if __name__ == "__main__":
    main()
