#!/usr/bin/env python3
"""Reproduce the DCL sim's race re-arm UI sequence via input injection.

CONFIRMED (2026-07-25): the sim re-arms a race ONLY through UI navigation, not a
timer / reconnect / MAVLink command. The sequence is:
    Escape  ->  "Back to Main Menu"  ->  re-click the "AP Virtual Qualifier R1" tab
which arms a fresh countdown. This tool replays that between grinder attempts so
the grind runs unattended.

Coords are ABSOLUTE screen pixels, recorded by --calibrate (you click each button
once). If the sim window moves, re-calibrate (it takes ~15 s). The sequence and
per-step delays are data in the JSON config, so tweaks never touch code.

    pip install pyautogui           # one-time, on the sim box
    python tools/ui_reset.py --calibrate     # record button positions (do once)
    python tools/ui_reset.py --reset         # test the sequence live
    python tools/ui_reset.py --show          # print the saved config
The grinder calls run_sequence() between attempts when run with --ui-reset.
"""
import argparse
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DEFAULT_CONFIG = os.path.join(ROOT, "ui_reset.json")

# The sequence template. "click" steps get x/y filled by --calibrate; "key" steps
# need no coords. delay_after = seconds to wait for the UI to transition.
DEFAULT_STEPS = [
    {"action": "key", "key": "escape", "delay_after": 1.5,
     "note": "exit the race to the pause menu"},
    {"action": "click", "name": "back_to_main_menu", "x": 0, "y": 0, "delay_after": 2.5,
     "note": "the 'Back to Main Menu' button"},
    {"action": "click", "name": "r1_tab", "x": 0, "y": 0, "delay_after": 4.0,
     "note": "the 'AP Virtual Qualifier R1' tab -> arms a fresh countdown"},
]


def _pyautogui():
    try:
        import pyautogui
    except ImportError:
        sys.exit("pyautogui not installed. On the sim box run:  pip install pyautogui")
    pyautogui.PAUSE = 0.1
    return pyautogui


def load_config(path=DEFAULT_CONFIG):
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh)
    return {"description": "DCL VQ1 race re-arm UI sequence (absolute screen px).",
            "steps": [dict(s) for s in DEFAULT_STEPS]}


def save_config(cfg, path=DEFAULT_CONFIG):
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(cfg, fh, indent=2)
    print(f"[ui_reset] saved {path}")


def calibrate(path=DEFAULT_CONFIG):
    pg = _pyautogui()
    cfg = load_config(path)
    print("[ui_reset] CALIBRATION. For each button: move the mouse over it in the "
          "sim window, then press Enter here. (Ctrl-C to abort.)")
    for step in cfg["steps"]:
        if step["action"] != "click":
            continue
        input(f"  -> hover over {step['name']} ({step.get('note', '')}) and press Enter... ")
        x, y = pg.position()
        step["x"], step["y"] = int(x), int(y)
        print(f"     recorded {step['name']} at ({x}, {y})")
    save_config(cfg, path)
    print("[ui_reset] done. Test it with:  python tools/ui_reset.py --reset")


def run_sequence(cfg, countdown=3.0, verbose=True):
    """Execute the UI re-arm sequence. Returns True on completion."""
    pg = _pyautogui()
    steps = cfg.get("steps", [])
    if any(s["action"] == "click" and (s.get("x", 0) == 0 and s.get("y", 0) == 0)
           for s in steps):
        print("[ui_reset] WARNING: some click coords are (0,0) -- run --calibrate first.")
        return False
    if countdown:
        for i in range(int(countdown), 0, -1):
            print(f"[ui_reset] running UI re-arm in {i}s (focus the sim window)...")
            time.sleep(1.0)
    for step in steps:
        if step["action"] == "key":
            pg.press(step["key"])
            if verbose:
                print(f"[ui_reset] key {step['key']}")
        elif step["action"] == "click":
            pg.click(step["x"], step["y"])
            if verbose:
                print(f"[ui_reset] click {step['name']} ({step['x']},{step['y']})")
        time.sleep(float(step.get("delay_after", 1.0)))
    print("[ui_reset] sequence complete -- a fresh countdown should now be arming.")
    return True


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=DEFAULT_CONFIG)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--calibrate", action="store_true", help="record button positions")
    g.add_argument("--reset", action="store_true", help="run the sequence now (test)")
    g.add_argument("--show", action="store_true", help="print the saved config")
    args = ap.parse_args()

    if args.calibrate:
        calibrate(args.config)
    elif args.show:
        print(json.dumps(load_config(args.config), indent=2))
    elif args.reset:
        run_sequence(load_config(args.config))


if __name__ == "__main__":
    main()
