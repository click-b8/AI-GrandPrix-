# Vision pipeline

`vq1_vision_servo.py` receives the simulator's FPV camera over UDP 5601 (MAVLink `ENCAPSULATED_DATA`) and reduces each frame to the three scalars the controller consumes.

## Detector

A deliberately minimal HSV color-mask blob detector: threshold the gate's color band, take the largest connected blob, output its **centroid** (cx, cy) and **area fraction**. No corner extraction, no contour fitting, no pose estimation — the sim's gates are high-contrast and the qualifier round rewards robustness over geometry.

```
frame ─ HSV mask ─ largest blob ─▶ u_err = (cx − W/2)/(W/2)     (+ = gate right of frame center)
                                  v_err = (cy − H/2)/(H/2)     (+ = gate low in frame = drone HIGH)
                                  size_frac = blob area / frame area   (≈ inverse distance)
```

Sign conventions were verified against source twice during the campaign (they were the root of an early tuning error) and are asserted in tests.

## Filtering and derived signals

The controller low-passes each scalar (`u_f`, `v_f`, `sz_f`, time-constant `--gate-filter-tau`) and computes a dirty derivative `du_f` on the *filtered* lateral signal — the input to the commit latch's settledness test. A far-gate rejector drops small distant blobs that would otherwise contaminate the early leg.

## Known failure modes (measured, not speculative)

- **Terminal parallax spike:** inside ~2.4 m the centroid geometry amplifies small offsets; post-plane spurious re-detections spike `v_err` to ±0.9. This produced a metric bug (max-size frame ≠ true crossing) that once inverted a whole tuning conclusion — the analysis layer now measures the **true crossing** (deepest stable frame before first sustained loss).
- **Gate loss before the plane:** the gate exits the frame at blob size ≈ 0.55; every approach ends open-loop. The terminal descent's blind hold exists because of this.
- **Off-rate flicker:** at degraded loop rates the last valid detection before a loss is often a shallow spurious blob, which interacted with arm-precondition logic (see [findings.md](findings.md) §arming).
- **Residual carry-over:** the *previous* gate's blob size decays through the filter for seconds into the next leg — the single largest blocker of the commit latch (43% of overnight runs never armed because of it).

## A road not taken: PnP

An 8-corner planar-gate PnP feasibility study (full pose from gate corners) was scoped, audited, and **frozen at the audit stage** — the corner detector rebuild it required was judged too risky against the qualifier deadline relative to improving the blob-servo path. The audit lives in the archived project notes.
