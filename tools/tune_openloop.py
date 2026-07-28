#!/usr/bin/env python3
"""Semi-automatic tuning loop for the OpenLoopFlier (the anti-flip controller).

The flip is dead; what's left is a tunable vertical/attitude drift. This closes a
tight loop: fly one race -> parse the [openloop] trace -> report the failure
signature -> propose ONE knob change (with reasoning) -> apply it -> pause for you
to trigger the next race. Every iteration's config + result is journalled so you
see the TRAJECTORY of tuning, not just the last state.

SEMI-auto by design: the DCL sim only re-arms a race via its UI (Escape -> menu ->
R1 tab), which a script can't reliably drive, and each flight needs a live GO on
the sim's cadence. So this NEVER triggers the race itself -- it launches run_vq1,
waits for YOUR manual GO, captures the flight, then PAUSES and asks you to re-arm
before the next iteration. It will not fly into dead air.

One knob per flight (hard rule): two changes and you can't tell which helped.

Subcommands
-----------
    python tools/tune_openloop.py analyze openloop_run.log     # parse a log, propose (no fly)
    python tools/tune_openloop.py loop                          # the semi-auto loop
    python tools/tune_openloop.py show                          # print current tune.json + journal tail

The knob store is openloop_tune.json (repo root); OpenLoopFlier loads it at
construction and logs which overrides it flew, so the flight log is self-describing.
"""
import argparse
import json
import os
import re
import subprocess
import sys
import threading
import time
import traceback
from datetime import datetime

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from vq1_vision_servo import ServoConfig, OPENLOOP_TUNE_KEYS  # noqa: E402

TUNE_PATH = os.path.join(ROOT, "openloop_tune.json")
JOURNAL_PATH = os.path.join(ROOT, "openloop_tune_journal.jsonl")
LOG_DIR = os.path.join(ROOT, "tune_logs")

_OL = re.compile(
    r"seg=(?P<seg>\d+) ffthr=(?P<ff>[\d.]+).*?uLat=(?P<ulat>[-+\d.]+) uL=(?P<ul>[-+\d.]+) "
    r"crv=(?P<crv>[-+\d.]+) gsz=(?P<gsz>[\d.]+) desR=(?P<desr>[-+\d.]+) \| "
    r"EST roll=(?P<er>[-+\d.]+) pitch=(?P<ep>[-+\d.]+) (?P<fence>FENCE)? *\| "
    r"rate\(r=(?P<rr>[-+\d.]+) p=(?P<rp>[-+\d.]+) y=(?P<ry>[-+\d.]+)\)")


# ---------------------------------------------------------------- parsing
def _decode(path):
    """Decode a log to text. UTF-16 (PowerShell Tee) ONLY when the bytes actually say
    so -- a plain UTF-8 capture must NOT be force-decoded as UTF-16 (that "succeeds"
    into garbage and silently zeroes the parse)."""
    raw = open(path, "rb").read()
    if raw[:2] in (b"\xff\xfe", b"\xfe\xff"):          # explicit UTF-16 BOM
        return raw.decode("utf-16")
    if raw[:4000].count(0) > 50:                        # UTF-16 without BOM: many NULs
        return raw.decode("utf-16", "replace")
    for enc in ("utf-8-sig", "utf-8"):
        try:
            return raw.decode(enc)
        except UnicodeDecodeError:
            continue
    return raw.decode("latin-1")


def parse_log(path):
    """Return dict(rows=[...post-GO openloop frames...], meta={...}). Handles UTF-16
    PowerShell Tee output and console-wrapped [openloop] lines."""
    lines = _decode(path).splitlines()
    go_ts = next((l[:23] for l in lines if "[Race] GO" in l), None)
    ag = [int(m.group(1)) for l in lines for m in [re.search(r"active_gate=(\d+)", l)] if m]
    disarmed = any("DISARM" in l for l in lines)
    finished = any("race_finish_ns=" in l and not l.rstrip().endswith("-1") for l in lines)
    collisions = [l.strip() for l in lines if "COLLISION" in l and "[openloop]" not in l]

    rows = []
    for i, l in enumerate(lines):
        if "[openloop]" not in l:
            continue
        full = l
        if i + 1 < len(lines) and lines[i + 1].lstrip().startswith("desR="):
            full = l.rstrip() + " " + lines[i + 1].strip()
        m = _OL.search(full)
        if not m:
            continue
        d = {k: (v if k == "fence" else float(v)) for k, v in m.groupdict().items()}
        d["seg"] = int(d["seg"])
        d["ts"] = l[:23]
        d["post_go"] = (go_ts is None) or (l[:23] >= go_ts)
        rows.append(d)
    post = [r for r in rows if r["post_go"]]
    return {"rows_all": rows, "rows": post, "meta": {
        "go_ts": go_ts, "active_gate_max": max(ag) if ag else 0,
        "disarmed": disarmed, "finished": finished, "n_collision": len(collisions),
        "collisions": collisions[:5]}}


# ---------------------------------------------------------------- signature
def signature(parsed):
    rows, meta = parsed["rows"], parsed["meta"]
    if not rows:
        return {"ok": False, "reason": "no post-GO [openloop] frames parsed"}
    ep = [r["ep"] for r in rows]          # est pitch deg
    er = [abs(r["er"]) for r in rows]     # |est roll| deg
    ff = [r["ff"] for r in rows]
    gsz = [r["gsz"] for r in rows]
    cruise = ServoConfig().cruise_pitch_deg
    sig = {
        "ok": True,
        "n_frames": len(rows),
        "active_gate_max": meta["active_gate_max"],
        "pitch_first": ep[0], "pitch_peak_up": max(ep), "pitch_min": min(ep),
        "pitch_last": ep[-1], "pitch_target": cruise,
        "pitch_balloon": max(ep) - ep[0],           # + = nose ballooned UP off the glide
        "pitch_droop": ep[-1] - cruise,             # + = settled nose-up of target
        "roll_absmax": max(er),
        "ff_min": min(ff), "ff_max": max(ff), "ff_constant": (max(ff) - min(ff) < 1e-6),
        "gate_seen": max(gsz) > 0.02,
        "gsz_peak": max(gsz),
        "gate_lost_end": (max(gsz) > 0.02 and gsz[-1] < 0.01),
        "fence_frames": sum(1 for r in rows if r["fence"]),
        "n_collision": meta["n_collision"],
    }
    return sig


def _gate_phrase(sig):
    """Where it ended up relative to gate 1."""
    if sig["active_gate_max"] >= 1:
        return f"PASSED gate 1 (active_gate reached {sig['active_gate_max']})"
    if not sig["gate_seen"]:
        return "gate 1 never detected (never in frame)"
    if sig["gate_lost_end"]:
        return (f"reached gate 1 visually (gsz peak {sig['gsz_peak']:.3f}) then LOST it "
                f"-- ballooned above/past the gate, never passed")
    return f"gate 1 in view (gsz peak {sig['gsz_peak']:.3f}) but never passed (active_gate=0)"


def describe(sig):
    if not sig.get("ok"):
        return sig.get("reason", "unparseable")
    parts = [f"{sig['n_frames']} post-GO frames",
             f"pitch {sig['pitch_first']:+.1f}->peak {sig['pitch_peak_up']:+.1f}"
             f"->last {sig['pitch_last']:+.1f} (target {sig['pitch_target']:+.1f})",
             f"BALLOON {sig['pitch_balloon']:+.1f} deg, droop {sig['pitch_droop']:+.1f} deg",
             f"|roll|max {sig['roll_absmax']:.1f}",
             f"thrust {'CONSTANT' if sig['ff_constant'] else 'VARYING'} "
             f"{sig['ff_min']:.3f}-{sig['ff_max']:.3f}",
             _gate_phrase(sig)]
    if sig["fence_frames"]:
        parts.append(f"FENCE {sig['fence_frames']}f")
    if sig["n_collision"]:
        parts.append(f"{sig['n_collision']} COLLISION")
    return "\n           ".join(parts)


def print_report(sig, change, log_path=None):
    """Prominent on-screen block so you never have to hunt for a file."""
    print("\n" + "=" * 70)
    if log_path:
        print(f"  LOG: {log_path}")
    print("  SIGNATURE: " + describe(sig))
    if change is None:
        print("  PROPOSED:  (no change -- progress made or within tolerance; your call)")
    else:
        print(f"  PROPOSED ONE KNOB: {change['key']}  {change['old']} -> {change['new']}")
        print(f"    reason: {change['reason']}")
    print("=" * 70)


# ---------------------------------------------------------------- proposer (ONE knob)
def current_value(key):
    tune = read_tune()
    return float(tune.get(key, getattr(ServoConfig(), key)))


def propose(sig):
    """Return ONE {key, old, new, reason} change, or None if it looks good."""
    if not sig.get("ok"):
        return None
    if sig["active_gate_max"] >= 1:
        return None  # made progress -- let the human decide the next target

    # ARC-UP: nose ballooned up off the glide -> the attitude hold is too weak.
    if sig["pitch_balloon"] >= 6.0:
        kp = current_value("ol_kp_att")
        if kp < 3.0:
            new = round(min(kp * 2.0, 3.0), 3)
            return {"key": "ol_kp_att", "old": kp, "new": new,
                    "reason": (f"pitch ballooned {sig['pitch_balloon']:+.1f} deg off the "
                               f"-18 glide while thrust stayed CONSTANT -> pitch-hold too "
                               f"weak, not a thrust problem. Stiffen the hold {kp}->{new}. "
                               f"Safe: hard {current_value('ol_max_rate_rad_s')} rad/s rate "
                               f"clamp + no vertical feedback still make a flip impossible.")}
        # hold already stiff but it still balloons -> bias the glide target down instead
        cp = current_value("cruise_pitch_deg")
        return {"key": "cruise_pitch_deg", "old": cp, "new": round(cp - 2.0, 3),
                "reason": (f"hold already stiff (kp={kp}) yet it balloons -> bias the glide "
                           f"target {cp}->{cp-2.0} deg so the settle point drives forward-down.")}

    # SANK: settled well below target / nose pitched down hard -> too little lift.
    if sig["pitch_droop"] <= -6.0 or sig["pitch_min"] <= sig["pitch_target"] - 8.0:
        hi = current_value("ol_thrust_hi")
        return {"key": "ol_thrust_hi", "old": hi, "new": round(hi + 0.02, 3),
                "reason": (f"pitch droop {sig['pitch_droop']:+.1f} deg (sinking) with constant "
                           f"thrust -> lift a touch, ol_thrust_hi {hi}->{hi+0.02}.")}

    # LATERAL drift dominates.
    if sig["roll_absmax"] >= 15.0:
        kt = current_value("k_tube_lead")
        return {"key": "k_tube_lead", "old": kt, "new": round(kt * 0.6, 3),
                "reason": (f"|roll| hit {sig['roll_absmax']:.1f} deg (weaving) -> ease steering "
                           f"authority k_tube_lead {kt}->{round(kt*0.6,3)}.")}

    # residual droop, gentle target bias
    if abs(sig["pitch_droop"]) >= 3.0:
        cp = current_value("cruise_pitch_deg")
        d = -2.0 if sig["pitch_droop"] > 0 else +2.0
        return {"key": "cruise_pitch_deg", "old": cp, "new": round(cp + d, 3),
                "reason": (f"steady droop {sig['pitch_droop']:+.1f} deg -> bias glide target "
                           f"cruise_pitch {cp}->{cp+d} deg.")}
    return None


# ---------------------------------------------------------------- tune store + journal
def read_tune():
    if os.path.exists(TUNE_PATH):
        try:
            return json.load(open(TUNE_PATH, encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}
    return {}


def write_tune(tune):
    with open(TUNE_PATH, "w", encoding="utf-8") as fh:
        json.dump(tune, fh, indent=2, sort_keys=True)


def apply_change(change):
    assert change["key"] in OPENLOOP_TUNE_KEYS, change["key"]
    tune = read_tune()
    tune[change["key"]] = change["new"]
    write_tune(tune)
    return tune


def journal(entry):
    entry = dict(entry, wall=datetime.now().isoformat(timespec="seconds"))
    with open(JOURNAL_PATH, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(entry) + "\n")


# ---------------------------------------------------------------- subcommands
def cmd_analyze(args):
    parsed = parse_log(args.log)
    sig = signature(parsed)
    change = propose(sig)
    print_report(sig, change, log_path=args.log)
    if args.apply and change:
        tune = apply_change(change)
        journal({"event": "analyze+apply", "log": os.path.basename(args.log),
                 "signature": sig, "change": change, "tune_after": tune})
        print(f"[tune] APPLIED -> {TUNE_PATH}: {json.dumps(tune)}")
    return 0


def _fly_once(iter_n, go_timeout, flight_max_s):
    """Launch run_vq1 --controller open-loop, capture to a timestamped log, return
    (log_path, saw_go).

    A reader THREAD drains stdout to the file, so the main thread NEVER blocks on
    output -- the flight-timeout below always fires even if the drone arcs off-course
    and the race never 'ends' (no crash/finish signal exists). Ending conditions:
      - race-end line (DISARM / RACE COMPLETE / real race_finish_ns), or
      - flight_max_s wall-clock after GO (the reliable stop -- an off-course flight
        has NO natural end), or
      - run_vq1 exits on its own, or
      - no GO within go_timeout.
    An in-flight Ctrl-C ends THIS flight (parse what we have); it does NOT quit the loop."""
    os.makedirs(LOG_DIR, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(LOG_DIR, f"openloop_iter{iter_n:02d}_{stamp}.log")
    print(f"[tune] launching run_vq1 (iter {iter_n}); waiting for the manual race GO ...")
    print(f"[tune]   -> log: {log_path}")
    # CREATE_NEW_PROCESS_GROUP (Windows): _stop()'s CTRL_BREAK_EVENT targets ONLY the
    # child's group, not the tuner's -- so the graceful stop can't self-interrupt us.
    proc = subprocess.Popen([sys.executable, os.path.join(ROOT, "run_vq1.py"),
                             "--controller", "open-loop"],
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            cwd=ROOT, text=True, bufsize=1,
                            encoding="utf-8", errors="replace",
                            creationflags=(getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
                                           if os.name == "nt" else 0))
    st = {"saw_go": False, "go_wall": None, "race_end": False, "eof": False}

    def _reader():
        with open(log_path, "w", encoding="utf-8") as lf:
            for line in proc.stdout:
                lf.write(line)
                lf.flush()
                if "[Race] GO" in line and not st["saw_go"]:
                    st["saw_go"] = True
                    st["go_wall"] = time.time()
                    print("[tune] GO detected -- flying.")
                if st["saw_go"] and ("DISARM" in line or "RACE COMPLETE" in line
                        or ("race_finish_ns=" in line and "race_finish_ns=-1" not in line)):
                    st["race_end"] = True
        st["eof"] = True

    th = threading.Thread(target=_reader, daemon=True)
    th.start()
    t_launch = time.time()
    reason = "?"
    try:
        while True:
            time.sleep(0.2)
            if st["eof"]:
                reason = "run_vq1 exited"; break
            if not st["saw_go"]:
                if time.time() - t_launch > go_timeout:
                    reason = f"no GO within {go_timeout:.0f}s"; break
                continue
            if st["race_end"]:
                reason = "race-end signal"; break
            if time.time() - st["go_wall"] > flight_max_s:
                reason = f"flight timeout ({flight_max_s:.0f}s after GO)"; break
    except KeyboardInterrupt:
        reason = "Ctrl-C (this flight only -- loop continues)"
    finally:
        _stop(proc)
        th.join(timeout=3)
    print(f"[tune] FLIGHT ENDED ({reason}). PARSING {os.path.basename(log_path)} ...")
    return log_path, st["saw_go"]


def _stop(proc):
    """Graceful stop so run_vq1 sends throttle-0 + DISARM (needed for the sim to
    re-arm). Swallows everything -- stopping a flight must never raise into the loop."""
    if proc.poll() is not None:
        return
    try:
        if os.name == "nt":
            proc.send_signal(getattr(__import__("signal"), "CTRL_BREAK_EVENT", 2))
        else:
            proc.terminate()
        proc.wait(timeout=6)
    except KeyboardInterrupt:
        try:
            proc.kill()
        except Exception:
            pass
    except Exception:
        try:
            proc.kill()
        except Exception:
            pass


def _load_ui_auto():
    """Return (ui_reset module, cfg) if --ui-auto is usable, else raise with guidance."""
    sys.path.insert(0, os.path.join(ROOT, "tools"))
    import ui_reset  # noqa: E402
    cfg = ui_reset.load_config()
    steps = cfg.get("steps", [])
    bad = (not steps) or any(s.get("action") == "click" and s.get("x", 0) == 0
                             and s.get("y", 0) == 0 for s in steps)
    if bad:
        raise SystemExit("[tune] --ui-auto needs calibration first:\n"
                         "        python tools/ui_reset.py --calibrate")
    return ui_reset, cfg


def _next_race(it, ui):
    """Advance to the next armed race. Manual (wait for Enter) or UI-AUTO (replay clicks)."""
    if ui is None:
        input(f"\n[tune] ==== iteration {it} ====  Re-arm the race in the sim UI "
              f"(Escape -> menu -> R1 tab), then press Enter to launch (Ctrl-C to quit) ... ")
        return True
    ui_reset, cfg = ui
    print(f"\n[tune] ==== iteration {it} ==== UI-AUTO re-arming (focus the sim window) ...")
    if not ui_reset.run_sequence(cfg):
        print("[tune] UI re-arm reported failure -- falling back to manual for this one.")
        input("[tune] re-arm by hand, then press Enter ... ")
    return True


def cmd_loop(args):
    ui = None
    if args.ui_auto:
        ui = _load_ui_auto()   # raises SystemExit with instructions if not calibrated
        print("[tune] UI-AUTO enabled. FRAGILE: absolute click coords -- breaks if the sim "
              "window moves/resizes; re-run ui_reset.py --calibrate if so.")
    else:
        print("[tune] SEMI-AUTO. One knob per flight. I never trigger the race -- you do.")
    tune = read_tune()
    print(f"[tune] current overrides: {json.dumps(tune) if tune else '(none -> ServoConfig defaults)'}")
    it = args.start
    while True:
        try:
            _next_race(it, ui)
        except KeyboardInterrupt:   # Ctrl-C at the re-arm prompt = quit the loop
            print("\n[tune] stopped by user at the prompt.")
            return 0
        try:
            log_path, saw_go = _fly_once(it, args.go_timeout, args.flight_max_s)
            if not saw_go:
                print("[tune] no flight captured (no GO); retry this iteration.")
                if ui is not None:
                    time.sleep(2.0)
                continue
            parsed = parse_log(log_path)
            sig = signature(parsed)
            change = propose(sig)
            print(f"[tune] captured {sig.get('n_frames', 0)} post-GO frames.")
            print_report(sig, change, log_path=log_path)
        except Exception:  # a bad flight must NEVER kill the loop
            print("[tune] !!! iteration errored -- traceback below; loop CONTINUES.")
            traceback.print_exc()
            continue
        entry = {"event": "fly", "iter": it, "log": os.path.basename(log_path),
                 "tune_before": read_tune(), "signature": sig, "change": change}
        if change is None:
            journal(entry)
            print("[tune] no change -- progress or within tolerance. Your call on next step.")
        else:
            if args.auto_apply:
                tune = apply_change(change)
                journal(dict(entry, applied=True, tune_after=tune))
                print(f"[tune] AUTO-APPLIED -> {json.dumps(tune)}")
            else:
                ans = input("[tune] apply this? [Enter=yes / n=skip / q=quit] ").strip().lower()
                if ans == "q":
                    journal(dict(entry, applied=False))
                    break
                if ans == "n":
                    journal(dict(entry, applied=False))
                    print("[tune] skipped -- config unchanged.")
                else:
                    tune = apply_change(change)
                    journal(dict(entry, applied=True, tune_after=tune))
                    print(f"[tune] APPLIED -> {json.dumps(tune)}")
        it += 1
    return 0


def cmd_show(args):
    print(f"[tune] {TUNE_PATH}:")
    print(json.dumps(read_tune(), indent=2) or "  (none)")
    if os.path.exists(JOURNAL_PATH):
        entries = [json.loads(l) for l in open(JOURNAL_PATH, encoding="utf-8") if l.strip()]
        print(f"\n[tune] journal ({len(entries)} entries), last {min(8, len(entries))}:")
        for e in entries[-8:]:
            ch = e.get("change")
            chs = f"{ch['key']} {ch['old']}->{ch['new']}" if ch else "(no change)"
            print(f"  {e.get('wall','?')} it={e.get('iter','-')} "
                  f"ag={e.get('signature',{}).get('active_gate_max','?')} "
                  f"balloon={e.get('signature',{}).get('pitch_balloon','?')} -> {chs} "
                  f"{'APPLIED' if e.get('applied') else ''}")
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("analyze", help="parse a log, report signature + proposed knob")
    a.add_argument("log")
    a.add_argument("--apply", action="store_true", help="also write the proposed knob to tune.json")
    a.set_defaults(func=cmd_analyze)
    lp = sub.add_parser("loop", help="semi-auto (default) / ui-auto fly/parse/propose/apply loop")
    lp.add_argument("--start", type=int, default=1, help="starting iteration number")
    lp.add_argument("--go-timeout", type=float, default=300.0, help="s to wait for the manual GO")
    lp.add_argument("--flight-max-s", type=float, default=30.0,
                    help="HARD flight timeout: s after GO to end the attempt on its own "
                         "(an off-course flight has NO natural race-end). Then parse + advance.")
    lp.add_argument("--ui-auto", action="store_true",
                    help="STRETCH/FRAGILE: replay the calibrated ui_reset.py click sequence "
                         "between flights instead of waiting for Enter (hands-off grinding). "
                         "Requires: python tools/ui_reset.py --calibrate")
    lp.add_argument("--auto-apply", action="store_true",
                    help="apply each proposed knob without the Enter confirmation (pairs with --ui-auto)")
    lp.set_defaults(func=cmd_loop)
    sh = sub.add_parser("show", help="print current tune.json + journal tail")
    sh.set_defaults(func=cmd_show)
    args = ap.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
