# Controller — the coast-tube pipeline

`tools/schedule_flier.py --coast-tube` is the competition controller. It is one file by design: a single control loop whose every experiment is a default-off CLI flag, so any historical configuration can be reproduced from a command line. The frozen qualifier configuration is recorded in [../results/overnight-2026-08-03/REPRODUCE.md](../results/overnight-2026-08-03/REPRODUCE.md).

## Control loop (per tick)

1. **Ingest** — MAVLink attitude + race state; vision frame → detector → raw `u_err` (lateral, + = gate right of frame center), `v_err` (vertical, + = drone high), `size_frac`.
2. **Filter** — low-pass to `u_f`, `v_f`, `sz_f`; dirty-derivative `du_f = Δu_f/dt` on the filtered signal.
3. **Lateral law** — desired bank = schedule bank (per-leg) + post-gate hold + vision servo `k_gate_bank · u_f`, clamped per-leg (Gate-3 leg: +0/−9°). The **commit latch** (below) fades the servo out at close range.
4. **Vertical law** — thrust = base thrust + per-leg descent bias + PD vertical trim `−(k_thrust_v·v_f + kd_v·dv)` within asymmetric authority, then the **Gate-3 terminal descent** and **blind hold** stages, then slew limiting.
5. **Emit** — `SET_ATTITUDE_TARGET` (roll command, thrust; pitch is uncontrolled by design), and a ~94-column CSV tick row.

## The mechanisms that got us through Gates 1–2

- **Altitude ladder / per-leg descent biases** (`--descent-bias-legN`): the backbone vertical plan; legs 0–2 tuned, legs 3–5 never reached tuning.
- **Post-gate hold** (`--post-gate-hold-s/decay`, `--post-gate1-bank`, `--post-gate2-bank`): a decaying scheduled bank after each gate crossing, replacing vision during the gate-to-gate gap.
- **The −0.6° held-bank lever** (`--gate2-hold-fixed-deg`): the signed bank held at the Gate-2 crossing was found to *predict* Gate-3 lateral arrival (r = 0.87 across the tuning campaign); fixing it deterministically at −0.6° replaced a latched, run-dependent value.
- **Gate-2 exit level** and **leg-2 entry arrest**: transition-smoothing at the highest-energy crossing.

## The Gate-3 mechanisms (the frontier)

- **Derivative-aware commit** (`--gate-commit-rate-max 0.13 --gate-commit-stable-s 0.15`): the close-range latch that freezes lateral steering once the approach is *settled* — alignment (|u_f| ≤ 0.08) **and** low rate (|du_f| < 0.13) held for a real-time 0.15 s dwell, armed only after the previous gate's size residual clears. Replaced a frame-count rule that meant different things at different loop rates. Post-campaign forensics ([findings.md](findings.md)) showed the latch is a *symptom* of a good approach rather than its cause — but a valuable one: latched runs crossed 3× more laterally centered because the latch blinds the servo to a terminal parallax spike.
- **Terminal descent** (`--gate3-vert-descent-delta 0.015 --gate3-vert-descent-size 0.25`): a fixed thrust cut on the deep Gate-3 approach, converting the leg's systematic "arrives high" bias into a controlled sink.
- **Blind hold** (`--gate3-vert-descent-hold-s 0.30`): the gate leaves the frame ~2.4 m out while still descending; this maintains the same cut for a bounded time into the blind coast, releasing on reacquisition, timer, or leg advance.

## What 152 flags actually are

The flag count is the experiment log. Families that were built, tested, and **disproven** remain in the code as default-off flags rather than deleted branches — e.g. the upward flare (`--gate3-vert-flare*`), rail-tube lateral steering (`--tube-lateral`, hard-disarmed after the detector rebuild), pitch-hold (the qualifier always flew `--no-pitch-hold`), and the open-loop turn era. This was a deliberate trade: reproducibility of any historical run over code minimalism. See [experiments.md](experiments.md) for the family-by-family history and verdicts.

## Invariants the campaign enforced

- Real-time units everywhere a rule counts (seconds-based dwells, per-second slews) — frame counts proved rate-dependent and were retired.
- Every experimental behavior defaults **off**; absence of its flag is bit-for-bit the previous controller.
- One change per flight test; anything validated got frozen and guarded by tests (~530 in `tests/`).
