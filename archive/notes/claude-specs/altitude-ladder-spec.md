# VQ1 altitude ladder — per-leg vertical feedforward (spec for Claude Code)

## Why (flt8 result — the pitch slowdown is DEAD)
`--pitch-hold-deg -15` failed hard: estP ran -17.8 -> +25.6 deg (nose-up runaway), drone
fell before gate 1, ag never left 0. Two coupled failures, both from moving pitch off the
coast equilibrium:
1. Accel-based pitch estimate corrupts under deceleration (decel reads as nose-down ->
   positive-feedback runaway nose-up).
2. Tilting the camera up makes the gate read low (v_err -> +0.71) -> vision-vertical loop
   commands descent -> drone sinks.
PERMANENT RULE: pitch stays at coast equilibrium (-17.8). It is the ONLY attitude where
v_err means altitude, not altitude+pitch. Do NOT re-arm any pitch hold. Keep Change D's
code but OFF.

## What to build — the per-leg altitude ladder
Today post_gate1_descent applies ONE value (0.026) to every leg ag>=1. That over-descends
the flat final legs (G3->G4, G4->FIN ~1.9/1.5 deg slope) and flies the drone into the
ground past gate 3. Replace with a per-leg table indexed by active_gate.

Table (from course_gates_cm.json slopes; anchor = proven G1->G2 0.026 at 17.1 deg):
descent_bias[ag] ~= 0.026 * (leg_slope / 17.1)
  ag0 (START->G1, 12.1): 0.000  (leave as-is: const 0.275 proven START pass)
  ag1 (G1->G2, 17.1):    0.026
  ag2 (G2->G3, 16.2):    0.025
  ag3 (G3->G4, 1.9):     0.003
  ag4 (G4->FIN, 1.5):    0.002
Compute at startup from JSON slopes (derive, don't hardcode). Add --descent-scale (default
1.0) one-knob trim and --descent-bias-legN per-leg overrides.

Behavior:
- On ag->i, immediately switch baseline to hover - descent_bias[i].
- Gate-vert trims around it when a gate is visible (as now).
- Between gates (no gate): HOLD the leg FF baseline. Do NOT coast to level, do NOT chase a
  far gate.
- Log active descent_bias[ag] in the tick line + CSV.

Do NOT touch: pitch (coast, Change D off), LPF, far-gate reject, lateral, kd_lat=0.

## Tuning loop
Fly -> python claude/protocol.py filtN.csv -> each gate closest-approach v_err:
  v_err > +0.1 (high): raise that leg bias.  v_err < -0.1 (low): lower it.
  Delta bias ~= v_err/126 per leg. One leg at a time. Anchor G1->G2=0.026 is proven.

## Speed
Pitch slowdown is out. Get gates with the ladder at coast speed first (a clean gate-2 exit
at the right altitude should fix the gate-3 lateral entry). If speed still bites later, the
only robust slowdown is a complementary-filter pitch estimate (gyro-integrated, accel-
corrected) -- a separate later build, not now.

## ADDENDUM (flt9) — LATERAL must follow the TUBE (the blue path = the rails)
flt9 passed gate 2 again, then veered off before gate 3 — and the CSV shows why. On the
gate-3 approach the lateral loop steers on GATE u_err, which sweeps from parallax at close
range while the TUBE stays smooth:
  t=10.4  GATE u_err +0.20   TUBE u +0.17
  t=10.6  GATE u_err +0.42   TUBE u +0.23
  t=11.0  GATE u_err +0.95   TUBE u +0.37   <- gate slams bank to -11 clamp, tube barely moved
The gate is a decoy that lunges sideways as you close on it; the tube is the rail.

BUILD: lateral steers on the TUBE (u_tube + curvature lead), like the original coast-tube
"LATERAL=TUBE(closed)" design, NOT gate-centering.
- des_roll = clamp(k_bank * (u_tube + k_tube_lead * curvature)) when the tube is solid.
- Use GATE u_err ONLY as a small fine-trim when a gate is LARGE and near-centred (sz high,
  |u_err| small); NEVER let a large/sweeping gate u_err drive the bank — that's the decoy.
- Tube lost/weak: hold last tube command briefly, then coast straight. No slam.

WHY IT CONNECTS TO THE LADDER: on the gate-3 leg the tube is weakly seen (area_frac ~0.007)
because the drone rides above the line. The altitude ladder puts it at the right altitude ->
tube area grows -> tube-following becomes reliable. Get on the line (ladder), stay on the
line (tube). Build the ladder FIRST (or together); verify tube area_frac rises on the
gate-3 leg once altitude is right.

This is also the real "slow & steady": following a continuous line is inherently steadier
than chasing a sweeping gate. Pitch slowdown stays DEAD (flt8). The rails are the fix.
