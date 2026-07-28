"""Go/no-go evaluator for the tube+gate hybrid pitch-hold flights.

    python tools/compare_pitch_runs.py base.csv hold.csv

Run A (base.csv) = --no-pitch-hold  : pure-coast baseline, zero new authority.
Run B (hold.csv) = --hold-pitch-from-seg 1 : hold engages at gate 1.

Checks the three gates that must ALL pass before the gate->thrust v-trim is
enabled on flight 2:

  1. est_pitch holds within ~+/-5 deg of the setpoint on segs 1-5 in B,
     vs large drift in A.
  2. pitch_n is not PINNED at the rate clamp for sustained stretches
     (a brief clamp at engagement is expected and fine).
  3. active_gate reaches 1 at the same time in both runs (the hold did not
     disturb the proven gate-1 pass).

If all three pass it then fits PITCH_REF from the (v_err, est_pitch) columns:
with pitch held, v_err should be ~affine in est_pitch, and the intercept is the
pitch at which v_err reads true. Reports the fit AND its quality -- a weak fit
means the residual is not yet a clean function of pitch and the trim should
stay off.
"""
import argparse
import csv
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from vq1_vision_servo import SPAWN_PITCH_DEG, MAX_BODY_RATE, ServoConfig  # noqa: E402

try:
    from config import FPV_FOV
except Exception:
    FPV_FOV = 58.72
HALF_FOV_DEG = FPV_FOV / 2.0


def load(path):
    with open(path, encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        raise SystemExit(f"{path}: empty")
    out = []
    for r in rows:
        try:
            out.append({
                "t": float(r["t"]), "seg": int(r["seg"]), "ag": int(r["active_gate"]),
                "thrust": float(r["thrust"]), "gate_found": int(r["gate_found"]),
                "v_err": float(r["v_err"]), "size_frac": float(r["size_frac"]),
                "cy": float(r["cy"]), "estP": float(r["est_pitch_deg"]),
                "gyroP": float(r["gyro_pitch"]), "desP": float(r["des_pitch_deg"]),
                "pitch_n": float(r["pitch_n"]), "tube_found": int(r["tube_found"]),
                "estR": float(r["est_roll_deg"]), "hold_on": int(r["hold_on"]),
                "loop_hz": float(r.get("loop_hz", 0) or 0),
                "det_hz": float(r.get("det_hz", 0) or 0),
            })
        except (KeyError, ValueError) as e:
            raise SystemExit(f"{path}: bad row -- {e}")
    return out


def first_ag(rows, n):
    for r in rows:
        if r["ag"] >= n:
            return r["t"]
    return None


def rate_summary(rows, label):
    lp = [r["loop_hz"] for r in rows if r["loop_hz"] > 0]
    dt = [r["det_hz"] for r in rows if r["loop_hz"] > 0]
    if not lp:
        print(f"  {label}: no rate samples (run shorter than 1 s?)")
        return
    lp.sort(); dt.sort()
    print(f"  {label}: loop_hz min/med/max = {lp[0]:.0f}/{lp[len(lp)//2]:.0f}/{lp[-1]:.0f}"
          f"   det_hz min/med/max = {dt[0]:.0f}/{dt[len(dt)//2]:.0f}/{dt[-1]:.0f}")


def seg_pitch_table(rows, label):
    print(f"\n  {label} est_pitch by segment:")
    print("    seg   n     min      max     mean    span   hold")
    for s in range(6):
        sr = [r for r in rows if r["seg"] == s]
        if not sr:
            continue
        ep = [r["estP"] for r in sr]
        mean = sum(ep) / len(ep)
        held = sum(r["hold_on"] for r in sr) / len(sr)
        print(f"    {s:<4}{len(sr):<6}{min(ep):+7.1f} {max(ep):+8.1f} {mean:+8.1f} "
              f"{max(ep)-min(ep):7.1f}   {held*100:3.0f}%")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("base", help="Run A csv (--no-pitch-hold)")
    ap.add_argument("hold", help="Run B csv (--hold-pitch-from-seg 1)")
    ap.add_argument("--setpoint", type=float, default=SPAWN_PITCH_DEG)
    ap.add_argument("--tol", type=float, default=5.0, help="criterion-1 band (deg)")
    ap.add_argument("--clamp-frac", type=float, default=0.25,
                    help="criterion-2: max fraction of held ticks allowed at the clamp")
    ap.add_argument("--ag-tol", type=float, default=1.0,
                    help="criterion-3: allowed |dt| on the ag->1 tick (s)")
    a = ap.parse_args()

    A, B = load(a.base), load(a.hold)
    cfg = ServoConfig()
    clamp = cfg.ol_max_rate_rad_s / MAX_BODY_RATE

    print("=" * 78)
    print(f"A (baseline) {a.base}: {len(A)} ticks, {A[-1]['t']:.1f}s, "
          f"max seg {max(r['seg'] for r in A)}, max ag {max(r['ag'] for r in A)}")
    print(f"B (hold)     {a.hold}: {len(B)} ticks, {B[-1]['t']:.1f}s, "
          f"max seg {max(r['seg'] for r in B)}, max ag {max(r['ag'] for r in B)}")
    print("\nLOOP RATE (mock predicted ~35/27 -- this is the real number):")
    rate_summary(A, "A")
    rate_summary(B, "B")

    seg_pitch_table(A, "A")
    seg_pitch_table(B, "B")

    # ---- criterion 1: hold arrests drift on segs 1-5 -------------------------
    print("\n" + "-" * 78)
    A15 = [r["estP"] for r in A if r["seg"] >= 1]
    B15 = [r["estP"] for r in B if r["seg"] >= 1 and r["hold_on"]]
    if not B15:
        print("1. FAIL -- no held ticks at seg>=1 in B (did the run reach seg 1?)")
        c1 = False
    else:
        devB = max(abs(p - a.setpoint) for p in B15)
        spanA = (max(A15) - min(A15)) if A15 else float("nan")
        c1 = devB <= a.tol
        print(f"1. {'PASS' if c1 else 'FAIL'} -- B max |est_pitch - {a.setpoint:.1f}| = "
              f"{devB:.1f} deg (tol {a.tol:.1f}); A drift span on seg>=1 = {spanA:.1f} deg")
        if A15 and spanA < a.tol:
            print("      NOTE: baseline barely drifted -- this run does not exercise the "
                  "hold; the comparison proves little.")

    # ---- criterion 2: pitch_n not pinned ------------------------------------
    held = [r for r in B if r["hold_on"]]
    if not held:
        print("2. FAIL -- no held ticks in B")
        c2 = False
    else:
        pinned = [r for r in held if abs(r["pitch_n"]) >= clamp * 0.999]
        frac = len(pinned) / len(held)
        # longest consecutive pinned stretch, in seconds
        longest = cur = 0.0
        prev = None
        for r in held:
            if abs(r["pitch_n"]) >= clamp * 0.999:
                cur += (r["t"] - prev) if prev is not None else 0.0
                longest = max(longest, cur)
            else:
                cur = 0.0
            prev = r["t"]
        c2 = frac <= a.clamp_frac
        print(f"2. {'PASS' if c2 else 'FAIL'} -- pitch_n at clamp (+/-{clamp:.3f}) on "
              f"{frac*100:.0f}% of held ticks (max {a.clamp_frac*100:.0f}%), "
              f"longest sustained stretch {longest:.2f}s")

    # ---- criterion 3: ag->1 unchanged ---------------------------------------
    tA, tB = first_ag(A, 1), first_ag(B, 1)
    if tA is None or tB is None:
        c3 = False
        print(f"3. FAIL -- ag->1 not reached (A: {tA}, B: {tB}). "
              f"{'Baseline never passed gate 1, so the hold cannot be blamed.' if tA is None else ''}")
    else:
        c3 = abs(tA - tB) <= a.ag_tol
        print(f"3. {'PASS' if c3 else 'FAIL'} -- ag->1 at A={tA:.2f}s B={tB:.2f}s "
              f"(d={tB-tA:+.2f}s, tol {a.ag_tol:.1f}s)")

    # ---- PITCH_REF fit -------------------------------------------------------
    print("-" * 78)
    fit = [r for r in B if r["gate_found"] and r["hold_on"]]
    if len(fit) < 20:
        print(f"PITCH_REF fit: SKIPPED -- only {len(fit)} held ticks with a gate "
              f"(need >=20).")
    else:
        xs = [r["estP"] for r in fit]
        ys = [r["v_err"] for r in fit]
        n = len(xs)
        mx, my = sum(xs) / n, sum(ys) / n
        sxx = sum((x - mx) ** 2 for x in xs)
        sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
        if sxx < 1e-9:
            print("PITCH_REF fit: SKIPPED -- est_pitch is constant over the fit window "
                  "(hold worked TOO well to identify the slope). Use the geometric "
                  f"prior: 1 deg pitch = {1/HALF_FOV_DEG:.4f} v_err.")
        else:
            slope = sxy / sxx
            icpt = my - slope * mx
            syy = sum((y - my) ** 2 for y in ys)
            r2 = (sxy ** 2 / (sxx * syy)) if syy > 1e-12 else float("nan")
            pitch_ref = (-icpt / slope) if abs(slope) > 1e-9 else float("nan")
            print(f"PITCH_REF fit over {n} held ticks with a gate:")
            print(f"  v_err = {slope:+.5f} * est_pitch_deg {icpt:+.5f}    R^2 = {r2:.3f}")
            print(f"  geometric prior (1/half-FOV, {HALF_FOV_DEG:.2f} deg) = "
                  f"{1/HALF_FOV_DEG:+.5f} per deg  -> measured/prior = "
                  f"{slope*HALF_FOV_DEG:+.2f}x")
            print(f"  => PITCH_REF (v_err crosses 0) = {pitch_ref:+.2f} deg")
            if r2 < 0.5:
                print("  WARNING: R^2 < 0.5 -- v_err is NOT yet a clean function of "
                      "pitch. Do not enable the trim on this fit.")

    print("=" * 78)
    ok = c1 and c2 and c3
    print(f"VERDICT: {'GO' if ok else 'NO-GO'} for enabling the gate v-trim (flight 2).")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
