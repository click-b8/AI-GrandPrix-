# Controller — the coast-tube pipeline

`tools/schedule_flier.py --coast-tube` (3,925 lines) is the competition controller: one control loop, one file, every experiment a default-off flag. This page explains its mechanisms so nobody has to read the source to understand the system. Status labels: **[ACTIVE]** in the final VQ1 controller · **[EXPERIMENTAL]** functional, default-off · **[RULED OUT]** tested and disproven (kept for reproducibility — see [flags.md](flags.md)).

## The state machine: `active_gate`

The sim reports race progress as `active_gate` (`ag`) — the index of the current leg, equal to gates passed: `ag 0` = start→G1, `ag 2` = the Gate-3 approach, `ag > 5` or `race_fin > 0` = course complete. Nearly every mechanism is *leg-scoped*: gains, clamps, holds, and the Gate-3 terminal logic all key off `ag`. Legs 0–2 were tuned; legs 3–5 were never reached.

## Inputs (per tick)

From the vision pipeline ([vision.md](vision.md)): `u_err` (lateral, + = gate right of frame center), `v_err` (vertical, + = drone high), `size_frac` (blob area ≈ inverse distance). Low-passed to `u_f, v_f, sz_f` (time constant `gate-filter-tau`), with a dirty derivative on the filtered lateral:

```
du_f = (u_f − u_f_prev) / dt        # real loop dt, not frame count
```

From MAVLink: attitude, race state. The loop runs unthrottled (39–117 Hz observed); every rule below is written in real-time units because of that.

## Lateral law [ACTIVE]

```
des_roll = clamp( bank_seg(ag)                 # per-leg feed-forward schedule
                + hold_bank(t)                 # decaying post-gate hold
                + commit_k · k_gate_bank · u_f # vision servo, faded by commit
                , limits(ag) )                 # ±11°; Gate-3 leg: +0 / −9°
```

- **Feed-forward schedule** `bank_seg`: the coarse route. The Gate-3 leg's asymmetric clamp (`+0/−9°`) reflects the course geometry.
- **Post-gate hold** `hold_bank`: after each crossing, a scheduled bank (`+5°` post-G1, `−4.5°` post-G2) held 1.5 s then decayed (τ = 0.5 s) — vision is unreliable in the gate-to-gate gap.
- **The Gate-2 held-bank lever** [ACTIVE — the campaign's key tuning discovery]: the signed bank held at the moment of the G2 crossing *predicts* Gate-3 lateral arrival (r = 0.87 across the tuning campaign). Rather than latching whatever bank the servo happened to hold, it is **fixed**: `--gate2-hold-fixed-deg -0.6`, the regression's zero. One number replaced the largest run-to-run variance source.
- **Servo authority ramp**: gain scales in from `sz_f` 0.05 → full at 0.15 — a distant blob steers weakly.
- **Leg-2 entry arrest** [ACTIVE]: 0.12 thrust arrest for 1.6 s entering the Gate-3 leg — damps the high-energy G2 exit.
- **Rail-tube steering** [RULED OUT]: steering on the course's rail/tube structure. The detector rebuild changed its signal contract; the old law didn't transfer; hard-disarmed in code.

## The commit latch [ACTIVE] — close-range servo fade

Near a gate, centroid geometry amplifies small offsets (terminal parallax); an active servo *chases* those spikes. The latch freezes lateral steering once the approach is genuinely settled:

```
armed     : sz_f fell below commit_size since the last gate      # previous gate's
                                                                 # filtered residual must clear
eligible  : sz_f ≥ 0.16
settled   : |u_f| ≤ 0.08  AND  |du_f| < 0.13, held continuously ≥ 0.15 s (real time)
latch     : armed ∧ eligible ∧ settled   → commit_k lags to 0, servo fades out
```

Evolution (full story in [experiment-history.md](experiment-history.md)): size-only commit latched on the *previous* gate's residual → arm precondition; a frame-count streak latched on fast zero-crossings and meant different things at different loop rates → the **derivative-aware, real-time** rule. Post-campaign forensics ([findings.md](findings.md)) established its true role: it does **not** cause straight approaches (the trajectory fork precedes eligibility), but as a terminal instrument it *triples* lateral crossing precision (median |u_f| 0.088 latched vs 0.254 unlatched) by blinding the servo to the terminal spike. A strictly better rule (0.20 / 0.12 / 0.16) was replay-validated on 302 runs but never flown.

## Vertical law [ACTIVE]

No climb authority exists (thrust ceiling; −17.8° nose-down coast), so vertical control is *descent management*:

```
thrust = const_thrust                       # 0.275 base
       + descent_bias(ag)                   # per-leg "altitude ladder"
       + clamp(−(k_v·v_f + kd_v·dv_f), asymmetric authority)   # PD vision trim
       − gate3_descent_cut (if active)      # terminal stage, below
→ slew-limited (0.6/s both directions)
```

The **altitude ladder** (`descent-bias-leg0..2` = −0.005 / +0.039 / +0.024) is the backbone; the PD trim (k = 0.16, kd = 0.1) corrects within deliberately asymmetric authority (down 0.045, up 0.06 — reflecting the missing climb authority).

## Terminal Gate-3 stage [ACTIVE]

The gate exits the camera frame at `sz` ≈ 0.55, ~2.4 m before its plane; the corrected crossing metric showed approaches arriving systematically *high* and still sinking:

- **Terminal descent**: at `sz_f ≥ 0.25` on the Gate-3 leg, a fixed thrust cut (Δ = 0.015). Fixed, not v_f-scaled — stale vision must not modulate authority.
- **Blind descent hold**: on a valid→invalid detection transition with the descent active and the drone still high, the *same* cut is held up to 0.30 s into the blind coast — released by reacquisition, timer, or leg advance. Never increases the cut.
- Result across the overnight campaign: vertical-low misses 3 of 302 — the descent never overshot. The residual high bias (+0.15…+0.37 at loss) on fast transits is the documented open item.
- **Upward flare** [RULED OUT]: solved the wrong sign — the drone arrives high, not low.

## Clamps and authority limits (summary)

Bank: ±11° global; Gate-3 leg +0/−9° asymmetric; servo gain ramps with blob size. Thrust: slew 0.6/s; PD trim authority asymmetric; terminal cut fixed-size with a floor clamp. Commit: `commit_k` is a lagged (not stepped) fade, and once latched never unlatches within a leg. Every clamp is leg-scoped via `ag`.

## What is deliberately absent

Pitch control (uncontrollable in coast — `--no-pitch-hold` always) [RULED OUT]; trajectory planning or map (nothing to plan against — no pose); gain scheduling on loop rate (rules are written in real-time units instead); PnP pose estimation [EXPERIMENTAL — frozen at feasibility audit].
