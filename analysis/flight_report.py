#!/usr/bin/env python3
"""One-line verdict for a schedule_flier tick-log CSV.
Gate-3 crossing is measured at the TRUE crossing = the deepest stable frame BEFORE the
gate is first lost, NOT the max-size frame over all of ag==2 (which catches post-plane
spurious re-detections and mislabels a high crossing as low). Flags DIRTY start, off-rate
by crossing dip, and rate. Exit code 3 if Gate 3 registered.

  flight_report.py LOG.csv                     human verdict line (unchanged)
  flight_report.py LOG.csv --json OUT.json     ...plus a machine-readable result file
  flight_report.py LOG.csv --json -            JSON on stdout INSTEAD of the verdict line

--json is purely additive: the verdict line's text and the exit code are identical with
and without it (except for `--json -`, which suppresses the line so stdout stays pure
JSON). The batch runner reads the file form; `-` is for interactive use.
"""
import csv, json, os, re, statistics, sys

DIRTY_G1 = 0.12
CROSS_WIN = 0.30
CROSS_MIN = 50.0
LOSS_RUN  = 3       # consecutive sz==0 frames = gate lost (post-plane spikes excluded after)
FLICKER_WIN = 0.75  # s before sustained Gate-3 loss that the flicker diagnostic inspects
DESCENT_SIZE = 0.25  # default --gate3-vert-descent-size; the flicker test's threshold

# COURSE SHAPE, from course_gates_cm.json + build_bank_schedule (NOT assumed):
#   6 gates, indices 0..5 -- 0=START, 1..4 intermediate, 5=FINISH.
#   active_gate is the index of the gate being flown TOWARD, so it doubles as the count
#   of gates already passed: ag==3 means gates 0,1,2 are behind (i.e. "Gate 3 passed").
#   The legs are seg0 SPAWN->START, seg1 START->g1, seg2 g1->g2, seg3 g2->g3,
#   seg4 g3->g4, seg5 g4->FINISH.
N_GATES = 6
GATE3_AG = 3        # active_gate value that means the third gate is behind us
# AUTHORITATIVE completion: the sim's race-finish field, shipped in the same
# ENCAPSULATED_DATA payload as active_gate. tools/schedule_flier.py ends the flight on
# `fin > 0 or ag > 5` -- fin>0 is the real signal, ag>5 is the fallback. A high
# active_gate alone is NOT completion.
FINISH_AG = N_GATES  # ag would have to exceed the last gate index to imply a finish


def _read_log(log_path):
    if not log_path or not os.path.exists(log_path):
        return ""
    try:
        with open(log_path, "r", errors="replace") as f:
            return f.read()
    except OSError:
        return ""


def _collisions(txt):
    """The flier prints collisions to its .log, not the CSV. Returns [(t, what), ...]."""
    return [(float(t), what) for what, t, _ag
            in re.findall(r"!!! COLLISION\s+(.+?)\s+at t=([\d.]+)s\s+\(ag=(\d+)\)", txt)]


def _collision_reason(log_path):
    txt = _read_log(log_path)
    if not txt:
        return ""
    col = _collisions(txt)
    if col:
        return f"COLLISION {col[0][1]} t={col[0][0]:.1f}s"
    for pat, name in ((r"max flight [\d.]+s -- stop", "TIMEOUT"),
                      (r"STALL", "GUARDIAN STALL"), (r"NO HEARTBEAT", "NO HEARTBEAT"),
                      (r"Traceback \(most recent", "PYTHON TRACEBACK")):
        if re.search(pat, txt):
            return name
    return ""


def analyse(fn, log_path=None, descent_size=DESCENT_SIZE):
    """Everything both outputs are derived from. Returns (dict, human_line, exit_code)."""
    with open(fn) as f:
        r = csv.reader(f); hdr = next(r); idx = {n: i for i, n in enumerate(hdr)}
        rows = [row for row in r if row and len(row) >= len(hdr)]
    G = lambda row, n: float(row[idx[n]])
    # Older logs predate the newer columns; never make their absence an error.
    H = lambda row, n, d=0.0: (float(row[idx[n]]) if n in idx else d)
    ag = [G(x, "active_gate") for x in rows]; t = [G(x, "t") for x in rows]; hz = [G(x, "loop_hz") for x in rows]
    maxag = int(max(ag)) if ag else 0

    fly = [hz[i] for i in range(len(rows)) if ag[i] >= 1 and 0 < hz[i] < 150]
    lhz = statistics.median(fly) if fly else 0.0
    pct_slow = (100.0 * sum(1 for h in fly if h < 55) / len(fly)) if fly else 100.0
    cross_bad = []
    for tgt in (1, 2, 3):
        ci = next((i for i in range(1, len(rows)) if ag[i] >= tgt and ag[i-1] < tgt), None)
        if ci is None: continue
        win = [hz[j] for j in range(len(rows)) if abs(t[j]-t[ci]) <= CROSS_WIN and 0 < hz[j] < 150]
        if win and min(win) < CROSS_MIN: cross_bad.append(f"g{tgt}:{min(win):.0f}Hz")
    badrate = (lhz < 55) or (lhz > 90) or bool(cross_bad)

    c1 = next((i for i in range(1, len(rows)) if ag[i] >= 1 and ag[i-1] < 1), None)
    g1uf = G(rows[c1], "u_f") if c1 else float("nan")
    dirty = (c1 is not None and abs(g1uf) > DIRTY_G1)

    # TRUE Gate-3 crossing: deepest stable frame before the gate is first lost.
    ag2 = [x for x in rows if G(x, "active_gate") == 2]
    g3 = ""
    g3_sz = g3_uf = g3_ve = float("nan")
    if ag2:
        gf = lambda x: (int(G(x, "gate_found")) if "gate_found" in idx else (G(x,"size_frac") > 0))
        # index of first sustained loss after a real approach (size>0.35 seen)
        seen_deep = False; cut = len(ag2)
        for i, x in enumerate(ag2):
            if G(x, "size_frac") > 0.35: seen_deep = True
            if seen_deep and all(G(ag2[j], "size_frac") == 0 for j in range(i, min(i+LOSS_RUN, len(ag2)))):
                cut = i; break
        pre = [x for x in ag2[:cut] if gf(x) and G(x, "size_frac") > 0.0]
        if pre:
            cr = max(pre, key=lambda x: G(x, "size_frac"))
            uf = G(cr, "u_f"); ve = G(cr, "v_err"); sz = G(cr, "size_frac")
            g3_sz, g3_uf, g3_ve = sz, uf, ve
            side = ("NEVER" if sz < 0.30 else "NEAR" if abs(uf) <= 0.25 else ("LEFT" if uf > 0 else "RIGHT"))
            vv = ("LOW" if ve < -0.15 else "HIGH" if ve > 0.15 else "vOK")
            g3 = f"G3* sz={sz:.2f} u_f={uf:+.3f}[{side}] v_err={ve:+.3f}[{vv}]"

    reg = maxag >= 3
    tag = "  *** GATE 3 REGISTERED ***" if reg else ""
    flags = ""
    if dirty:     flags += f" <<DIRTY g1={g1uf:+.2f}>>"
    if cross_bad: flags += f" <<CROSSING DIP {','.join(cross_bad)} — DISCARD>>"
    elif badrate: flags += f" <<RATE med={lhz:.0f} — DISCARD>>"
    ok = "" if (badrate or dirty) else "  OK"
    line = f"{fn:12} med={lhz:.0f} slow={pct_slow:.0f}% | g1={g1uf:+.3f} | gates={maxag} | {g3}{ok}{tag}{flags}"

    # ---- control-state summary over the ag==2 leg --------------------------------
    dt_med = statistics.median([t[i] - t[i-1] for i in range(1, len(t))]) if len(t) > 1 else 0.0
    desc_rows = [x for x in ag2 if H(x, "gate3_desc_on") > 0]
    hold_rows = [x for x in ag2 if H(x, "g3hold_on") > 0]
    hold_t = [G(x, "t") for x in hold_rows]
    res = dict(
        filename=os.path.basename(fn),
        median_rate=round(lhz, 1),
        slow_pct=round(pct_slow, 1),
        dirty_start=bool(dirty),
        crossing_dips=cross_bad,
        verdict=("DISCARD" if (badrate or dirty) else "OK"),
        gate1_u=(None if g1uf != g1uf else round(g1uf, 4)),
        max_active_gate=maxag,
        g3_size=(None if g3_sz != g3_sz else round(g3_sz, 4)),
        g3_u_f=(None if g3_uf != g3_uf else round(g3_uf, 4)),
        g3_v_err=(None if g3_ve != g3_ve else round(g3_ve, 4)),
        commit_latched=bool(any(H(x, "commit_latched") > 0 for x in ag2)),
        commit_stable_s=round(max([H(x, "commit_stable_s") for x in ag2] or [0.0]), 3),
        descent_frames=len(desc_rows),
        descent_impulse=round(sum(abs(H(x, "gate3_desc_raw")) for x in desc_rows) * dt_med, 5),
        hold_armed=bool(any(H(x, "g3hold_armed") > 0 for x in ag2)),
        hold_frames=len(hold_rows),
        hold_duration_s=(round(max(hold_t) - min(hold_t) + dt_med, 3) if hold_t else 0.0),
        registered=bool(reg),
        failure_reason=_collision_reason(log_path),
        rows=len(rows),
    )

    # ---- COURSE PROGRESSION: does the run actually finish? ------------------------
    # Registration at Gate 3 is a MILESTONE, never a success. The only authoritative
    # completion is the sim's race-finish field (race_fin>0 in the CSV, or the flier's
    # "*** COMPLETE *** ... finish=N" console line with N>0).
    txt = _read_log(log_path)
    fin_csv = max([H(x, "race_fin") for x in rows] or [0.0])
    fin_log = 0
    mfin = re.search(r"\*\*\* COMPLETE \*\*\*\s+active_gate=(-?\d+)\s+finish=(-?\d+)", txt)
    if mfin:
        fin_log = int(mfin.group(2))
    finish_val = int(max(fin_csv, fin_log))
    over_run = maxag > N_GATES - 1          # ag walked past the FINISH gate index
    complete = bool(finish_val > 0 or over_run)
    if finish_val > 0:
        sig = f"race_fin={finish_val}" + (" (log)" if (fin_log and not fin_csv) else " (csv)")
    elif over_run:
        sig = f"active_gate={maxag} > last gate index {N_GATES - 1}"
    else:
        sig = ""

    # time of each gate registration, from the ag step-ups
    gate_times = {}
    for i in range(1, len(rows)):
        if ag[i] > ag[i-1]:
            for a in range(int(ag[i-1]) + 1, int(ag[i]) + 1):
                gate_times.setdefault(a, round(t[i], 3))
    t_g3 = gate_times.get(GATE3_AG)
    later = {str(a): tt for a, tt in sorted(gate_times.items()) if a > GATE3_AG}

    timeout = bool(re.search(r"max flight [\d.]+s -- stop", txt))
    col_after = [c for c in _collisions(txt) if t_g3 is not None and c[0] >= t_g3]
    first_fail = ""
    if reg and not complete:
        if col_after:
            first_fail = f"COLLISION {col_after[0][1]} t={col_after[0][0]:.1f}s"
        elif timeout:
            first_fail = "TIMEOUT before finish"
        elif res["failure_reason"]:
            first_fail = res["failure_reason"]
        else:
            first_fail = f"ran out of log at active_gate={maxag} (no finish signal)"

    res.update(
        gate3_registered=bool(reg),
        highest_active_gate=maxag,
        gates_completed=min(maxag, N_GATES),
        full_course_complete=complete,
        finish_signal=sig,
        first_failure_after_gate3=first_fail,
        time_gate3_registered=t_g3,
        time_each_later_gate_registered=later,
        final_segment=(int(rows[-1][idx["seg"]]) if rows and "seg" in idx else None),
        collision_after_gate3=bool(col_after),
        timeout_after_gate3=bool(reg and timeout and not complete),
    )
    if complete:
        tag = "  *** FULL COURSE QUALIFIED ***"
    elif reg:
        tag = "  *** GATE 3 REGISTERED — CONTINUING COURSE ***"
    else:
        tag = ""
    line = (f"{fn:12} med={lhz:.0f} slow={pct_slow:.0f}% | g1={g1uf:+.3f} | "
            f"gates={maxag} | {g3}{ok}{tag}{flags}")

    # ---- detector-flicker diagnostic (REPORT ONLY) -------------------------------
    # The blind hold can only arm on a valid->invalid transition whose LAST valid frame
    # was still deep (sz >= descent size). If the detector fades out -- size decaying
    # below the threshold for a few frames before dropping to zero -- the descent has
    # already released and the hold never arms, even though the approach was deep a
    # moment earlier. That is a detector artefact masquerading as a shallow approach,
    # and it is invisible in the verdict line.
    fl = dict(applies=False, recent_max_size=None, last_valid_size=None,
              max_to_loss_s=None, descent_active=False, hold_armed=False,
              hold_activated=False, blocked=False)
    if ag2:
        seen_deep = False; cut = len(ag2)
        for i, x in enumerate(ag2):
            if G(x, "size_frac") > 0.35: seen_deep = True
            if seen_deep and all(G(ag2[j], "size_frac") == 0 for j in range(i, min(i+LOSS_RUN, len(ag2)))):
                cut = i; break
        if cut < len(ag2):                       # a sustained loss actually happened
            t_loss = G(ag2[cut], "t")
            win = [x for x in ag2[:cut]
                   if G(x, "size_frac") > 0.0 and (t_loss - G(x, "t")) <= FLICKER_WIN]
            if win:
                mx = max(win, key=lambda x: G(x, "size_frac"))
                last = win[-1]
                fl.update(
                    applies=True,
                    recent_max_size=round(G(mx, "size_frac"), 4),
                    last_valid_size=round(G(last, "size_frac"), 4),
                    max_to_loss_s=round(t_loss - G(mx, "t"), 3),
                    descent_active=bool(any(H(x, "gate3_desc_on") > 0 for x in win)),
                    hold_armed=bool(any(H(x, "g3hold_armed") > 0 for x in ag2)),
                    hold_activated=bool(len(hold_rows) > 0),
                )
                fl["blocked"] = bool(
                    fl["recent_max_size"] >= descent_size
                    and fl["last_valid_size"] < descent_size
                    and fl["descent_active"]
                    and not fl["hold_activated"])
    res["flicker"] = fl
    return res, line, (3 if reg else 0)


def main(argv):
    args = [a for a in argv[1:]]
    json_target = None
    if "--json" in args:
        i = args.index("--json")
        json_target = args[i + 1] if i + 1 < len(args) else "-"
        del args[i:i + 2]
    descent_size = DESCENT_SIZE
    if "--descent-size" in args:
        i = args.index("--descent-size")
        descent_size = float(args[i + 1]); del args[i:i + 2]
    log_path = None
    if "--log" in args:
        i = args.index("--log")
        log_path = args[i + 1]; del args[i:i + 2]
    if not args:
        print(__doc__); return 2
    fn = args[0]
    if log_path is None:
        cand = os.path.splitext(fn)[0] + ".log"
        log_path = cand if os.path.exists(cand) else None

    res, line, code = analyse(fn, log_path=log_path, descent_size=descent_size)

    if json_target == "-":
        print(json.dumps(res, indent=2))
        return code
    print(line)
    fl = res["flicker"]
    if fl["blocked"]:
        print(f"HOLD BLOCKED BY FLICKER  recent_max={fl['recent_max_size']:.3f} "
              f"last_valid={fl['last_valid_size']:.3f} "
              f"max->loss={fl['max_to_loss_s']:.3f}s "
              f"descent_active={int(fl['descent_active'])} "
              f"hold_armed={int(fl['hold_armed'])} hold_frames={res['hold_frames']}")
    if json_target:
        with open(json_target, "w") as f:
            json.dump(res, f, indent=2)
    return code


if __name__ == "__main__":
    sys.exit(main(sys.argv))
