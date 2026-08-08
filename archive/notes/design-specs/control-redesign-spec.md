# VQ1 control redesign — FF + PID + LPF (implementation spec)

Goal: stop tuning blind constants; make every adjustment analytical. Diagnostic tool:
`archive/notes/design-specs/protocol.py` — run on any `filtN.csv` for per-gate closest-approach error.
IMPORTANT: protocol.py samples at the **peak-size (closest-approach) tick** of each gate
window, NOT the ag-transition tick. That is why gate-1 reads v_err ≈ −0.11 (window min at
closest approach) while the ag→1 tick shows v_err ≈ +0.5. Use protocol.py's number.

## STATUS after flt5 (post-gate1-descent 0.026): GATE 2 PASSED (active_gate→2)
Frontier is gate 3. Per-gate closest-approach (protocol.py):
- gate1: v_err −0.11, u_err −0.03 — clean.
- gate2: v_err +0.30 (a bit high, barely passed). High because the gate-1 climb ballooned
  the drone up — Priority 3 integral should absorb this.
- gate3: LOST to lateral under-damping. u_err tracked to ≈0 at mid-range (t=9.3) but the
  drone carried leftward roll velocity from the gate-2 exit; at close range the gate swept
  right u: 0.0→+0.19→+0.34→+0.53→+0.78, faster than the −11° max bank → missed left.

## PRIORITY 1 — Lateral derivative damping (DONE, pending gain approval)
Implemented: `du_lat = LATERAL_SIGN*sign*(u_f − u_f2)/tau`, `trim = clamp(k_gate_bank*u_lat
− kd_lat*du_lat)`, flag `--kd-lat`. Two implementation notes from the replay analysis, both correct
and APPROVED:
1. **Default gain = 0.30, NOT 0.8.** The filt5 ag=2 replay: kd_lat=0.8 makes D
   larger than P at the close-range rush (D/P=1.08) and RAISES peak reversal 24.8→187.6
   deg/s. 0.30 is the largest gain that keeps peak ≤ pure-P (24.5) while cutting clamp
   saturation 25/40→19/40. **Set the default to 0.30.**
2. **u_f2 keeps converging during a HOLD** (unlike v_f2 which freezes). Freezing both
   stages pins du at a stale value → command slam (436 deg/s) when the gate returns.
   Correct deviation — keep it. NOTE: v_f2 has the same latent defect; fix it in a later
   slot (low impact today because kd_v acts behind a thrust clamp).
Acceptance: refly + protocol.py — gate-3 u_err should stop sweeping past ~0.4 at sz>0.3.
Note the replay only predicts; the flight is ground truth. If 0.30 damps but gate 3 still
sweeps, that's the geometry/speed limit → bring in Change D (slowdown), don't just raise
kd_lat past 0.30 (it amplifies).

## PRIORITY 2 — Kill the open-loop lateral veer (flt4; latent)
Any feedforward lateral bank decays to level when the gate is lost > `--ff-lat-hold-s`
(0.4s), τ=`--ff-lat-decay-s` (0.5s). flt5 didn't veer only because leg1 is ag==1-gated and
ag reached 2; the next missed gate re-exposes it.

## PRIORITY 3 — Integral on vertical PID (removes hand-bisect + gate-1 coupling)
`--ki-v` (default 0.02): integrate LPF v_err while a gate is live, clamp ±0.03, reset on
no-gate. Self-nulls gate 2's +0.30 residual.

## PRIORITY 4 — Per-leg FF table (replaces blind post-gate1-descent + leg1 bank)
From course_gates_cm.json, per leg (ag i ⇒ leg i): small vertical thrust trim + lateral
sidestep bank sized to the leg's lateral delta. Gated per leg, decayed per Priority 2.
| leg | fwd | lateral | drop | flightpath |
|---|---|---|---|---|
| START→G1 | 23.7m | −2.1 (L) | 5.1m | +12.1° |
| G1→G2 | 27.9m | +3.7 (R) | 8.6m | +17.1° |
| G2→G3 | 37.4m | −6.3 (L) | 10.9m | +16.2° |
| G3→G4 | 24.4m | +4.3 (R) | 0.8m | +1.9° |
| G4→FIN | 24.0m | −3.6 (L) | 0.6m | +1.5° |
(Hand-tuned leg1 +5° LEFT is geometrically BACKWARDS for G1→G2 which goes right.)

## CHANGE D (separate experiment) — gentle slowdown via near-equilibrium pitch hold
`pitch_n=0.0` today → speed fixed by coast (~8 m/s); --cruise can't slow it. Hold pitch at
−15° (~3° nose-up from the −17.8° equilibrium) with SOFT gain (kp≈1.0, not the 3.0 that
tumbled at −10°) + rate damping: `--pitch-hold-deg -15 --pitch-kp 1.0`. Goal ~8→~6.5 m/s.
Isolated test AFTER Priority 1; abort if |roll| grows (tumble signature).

## Confirmed plant (use, don't re-derive)
Hover ≈ 0.250 (|a|=1g). ol_thrust_hi ceiling 0.34 (up-auth dead past 0.10). thrust→vert
accel ≈ 0.53 m/s²/1%. Vertical sensitivity dV/d(post-gate1-descent) = −126/unit.

## Do NOT regress
Keep shared LPF (τ=0.2) + far-gate reject (size<0.055). Verify gate 1 (protocol gate#1
v_err ≈ −0.1) and gate 2 (ag→2) still pass after each change.

## Sequence
1. Priority 1 lands at kd_lat=0.30 → refly → protocol.py: gate-3 u_err should stop sweeping.
2. Priority 3 (integral) → gate 2 centers without touching descent.
3. Priority 4 (per-leg FF) → gates 3–6.
4. Change D (slowdown) as robustness layer, isolated.
Every refly: protocol.py → read table → one analytical change. No guessing.
