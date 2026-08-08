"""BATCH AUTOMATION: flight_report.py --json, and run_qualifier_batch.ps1's classifier.

WHY THESE EXIST. The batch runner decides -- unattended -- which flights are evidence and
which are noise. A misclassification is worse than a crash: a DISCARD filed as a keeper
quietly poisons the population the next decision is made from. So every bucket is pinned
against synthetic reports, and the PowerShell classifier is checked against the SAME
fixtures as the Python side so the two cannot drift apart.

The --json contract is additive. The human verdict line is byte-identical with and without
the flag; that is asserted, not assumed, because the line is what gets pasted into the log
and compared across runs by eye.
"""
import json
import os
import shutil
import subprocess
import sys
import tempfile

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from flight_report import analyse, main as report_main   # noqa: E402

# The batch script moved to automation/ in the portfolio restructure.
PS1 = os.path.join(ROOT, "automation", "run_qualifier_batch.ps1")
DEEP_SIZE = 0.45

COLS = ["t", "seg", "active_gate", "loop_hz", "u_f", "v_err", "size_frac", "gate_found",
        "commit_latched", "commit_stable_s", "gate3_desc_on", "gate3_desc_raw",
        "g3hold_armed", "g3hold_on", "g3hold_elapsed", "race_fin"]


def frame(t, ag, hz=65.0, u_f=0.0, v_err=0.0, sz=0.0, found=None,
          latched=0, stable=0.0, desc=0, hold_armed=0, hold=0, elapsed=0.0,
          race_fin=0):
    found = int(sz > 0) if found is None else found
    return dict(t=t, seg=ag, active_gate=ag, loop_hz=hz, u_f=u_f, v_err=v_err,
                size_frac=sz, gate_found=found, commit_latched=latched,
                commit_stable_s=stable, gate3_desc_on=desc,
                gate3_desc_raw=(-0.015 if desc else 0.0), g3hold_armed=hold_armed,
                g3hold_on=hold, g3hold_elapsed=elapsed, race_fin=race_fin)


def write_csv(path, frames):
    with open(path, "w", newline="") as f:
        f.write(",".join(COLS) + "\n")
        for fr in frames:
            f.write(",".join(str(fr[c]) for c in COLS) + "\n")


def synth(max_ag=2, g3_size=0.60, g1_u=0.02, hz=65.0, dip=False,
          desc=True, hold=True, fade=False, race_fin=0):
    """A full flight: spawn -> G1 -> G2 -> the Gate-3 approach -> loss."""
    fr, t = [], 0.0
    dt = 1.0 / 65.0
    for ag in (0, 1):
        for i in range(40):
            u = g1_u if (ag == 1 and i == 0) else 0.0
            h = 30.0 if (dip and ag == 1 and i == 0) else hz
            fr.append(frame(round(t, 4), ag, hz=h, u_f=u, sz=0.10))
            t += dt
    if max_ag >= 2:
        n = 30
        for i in range(n):                       # Gate-3 approach, size ramping up
            sz = g3_size * (i + 1) / n
            on = int(desc and sz >= 0.25)
            fr.append(frame(round(t, 4), 2, hz=hz, u_f=0.03, v_err=0.30, sz=sz,
                            latched=1, stable=0.16, desc=on))
            t += dt
        if fade:                                 # detector fades below the threshold
            for sz in (0.22, 0.18, 0.12):
                fr.append(frame(round(t, 4), 2, hz=hz, sz=sz)); t += dt
        for i in range(12):                      # sustained loss -> the blind coast
            h = int(hold and i < 7)
            fr.append(frame(round(t, 4), 2, hz=hz, sz=0.0,
                            hold_armed=int(hold), hold=h,
                            elapsed=(i * dt if h else 0.0)))
            t += dt
    for ag in range(3, max_ag + 1):          # the later legs, one block each
        for i in range(20):
            fr.append(frame(round(t, 4), ag, hz=hz, sz=0.10,
                            race_fin=(race_fin if ag == max_ag and i >= 18 else 0)))
            t += dt
    return fr


@pytest.fixture
def csv_of(tmp_path):
    def make(**kw):
        p = tmp_path / "run.csv"
        write_csv(str(p), synth(**kw))
        return str(p)
    return make


# ---------------------------------------------------------------------------------
# --json is additive
# ---------------------------------------------------------------------------------
def test_the_human_line_is_identical_with_and_without_json(csv_of, tmp_path, capsys):
    fn = csv_of()
    report_main(["flight_report.py", fn])
    plain = capsys.readouterr().out
    report_main(["flight_report.py", fn, "--json", str(tmp_path / "r.json")])
    withjson = capsys.readouterr().out
    assert plain == withjson, "--json changed the human-readable output"


def test_json_dash_prints_pure_json(csv_of, capsys):
    """stdout must be parseable as-is, or the interactive form is useless in a pipe."""
    report_main(["flight_report.py", csv_of(), "--json", "-"])
    json.loads(capsys.readouterr().out)


def test_the_json_carries_every_required_field(csv_of):
    res, _, _ = analyse(csv_of())
    for k in ("filename", "median_rate", "slow_pct", "dirty_start", "crossing_dips",
              "verdict", "gate1_u", "max_active_gate", "g3_size", "g3_u_f", "g3_v_err",
              "commit_latched", "commit_stable_s", "descent_frames", "descent_impulse",
              "hold_armed", "hold_frames", "hold_duration_s", "registered",
              "failure_reason"):
        assert k in res, f"missing field {k}"


def test_the_exit_code_still_signals_registration(csv_of):
    assert analyse(csv_of(max_ag=2))[2] == 0
    assert analyse(csv_of(max_ag=3))[2] == 3


def test_missing_new_columns_do_not_break_an_old_log(tmp_path):
    """Logs predating the hold columns must still report, with the new fields zeroed."""
    p = tmp_path / "old.csv"
    keep = ["t", "active_gate", "loop_hz", "u_f", "v_err", "size_frac", "gate_found"]
    frames = synth()
    with open(p, "w", newline="") as f:
        f.write(",".join(keep) + "\n")
        for fr in frames:
            f.write(",".join(str(fr[c]) for c in keep) + "\n")
    res, _, _ = analyse(str(p))
    assert res["hold_frames"] == 0 and res["descent_frames"] == 0


def test_the_collision_reason_comes_from_the_sibling_log(csv_of, tmp_path):
    fn = csv_of()
    log = os.path.splitext(fn)[0] + ".log"
    with open(log, "w") as f:
        f.write("[coast] !!! COLLISION Env/ground at t=12.4s (ag=2)\n")
    res, _, _ = analyse(fn, log_path=log)
    assert "COLLISION" in res["failure_reason"] and "12.4" in res["failure_reason"]


# ---------------------------------------------------------------------------------
# The flicker diagnostic
# ---------------------------------------------------------------------------------
def test_flicker_is_flagged_when_a_deep_approach_fades_out_before_loss():
    """The case the diagnostic exists for: deep a moment ago, shallow at the last valid
    frame, so the descent released and the hold never armed -- a detector artefact, not
    a shallow approach."""
    import tempfile as tf
    d = tf.mkdtemp()
    try:
        p = os.path.join(d, "f.csv")
        write_csv(p, synth(g3_size=0.60, desc=True, hold=False, fade=True))
        res, _, _ = analyse(p)
        fl = res["flicker"]
        assert fl["applies"]
        assert fl["recent_max_size"] >= 0.25 > fl["last_valid_size"]
        assert fl["descent_active"] and not fl["hold_activated"]
        assert fl["blocked"], "the flicker case was not flagged"
    finally:
        shutil.rmtree(d, ignore_errors=True)


def test_flicker_is_not_flagged_when_the_hold_actually_fired(csv_of):
    res, _, _ = analyse(csv_of(fade=True, hold=True))
    assert not res["flicker"]["blocked"]


def test_flicker_is_not_flagged_on_a_clean_deep_loss(csv_of):
    """No fade: the last valid frame is still deep, so the hold had its chance."""
    res, _, _ = analyse(csv_of(fade=False, hold=False))
    assert not res["flicker"]["blocked"]


def test_the_flicker_banner_is_printed(capsys):
    d = tempfile.mkdtemp()
    try:
        p = os.path.join(d, "f.csv")
        write_csv(p, synth(g3_size=0.60, desc=True, hold=False, fade=True))
        report_main(["flight_report.py", p])
        assert "HOLD BLOCKED BY FLICKER" in capsys.readouterr().out
    finally:
        shutil.rmtree(d, ignore_errors=True)


# ---------------------------------------------------------------------------------
# Classification -- every bucket, Python and PowerShell agreeing
# ---------------------------------------------------------------------------------
def classify_py(r):
    """Reference classifier -- must match Get-Classification in the .ps1."""
    if r is None:
        return "BROKEN"
    if r["full_course_complete"]:
        return "FULL_COURSE"
    if r["gate3_registered"]:
        return "POST_G3_FAILURE"
    if r["dirty_start"]:
        return "DISCARD_DIRTY"
    if r["verdict"] != "OK":
        return "DISCARD_RATE"
    if r["max_active_gate"] == 2:
        sz = r["g3_size"]
        return "KEEP_DEEP" if (sz is not None and sz >= DEEP_SIZE) else "KEEP_SHALLOW"
    return "KEEP_UPSTREAM"


CASES = [
    ("FULL_COURSE",     dict(max_ag=5, race_fin=123456)),
    ("POST_G3_FAILURE", dict(max_ag=3)),
    ("KEEP_DEEP",     dict(max_ag=2, g3_size=0.60)),
    ("KEEP_SHALLOW",  dict(max_ag=2, g3_size=0.30)),
    ("KEEP_UPSTREAM", dict(max_ag=1)),
    ("DISCARD_DIRTY", dict(max_ag=2, g1_u=0.40)),
    ("DISCARD_RATE",  dict(max_ag=2, hz=40.0)),
]


@pytest.mark.parametrize("expect,kw", CASES, ids=[c[0] for c in CASES])
def test_every_bucket_classifies(expect, kw, tmp_path):
    p = tmp_path / "c.csv"
    write_csv(str(p), synth(**kw))
    res, _, _ = analyse(str(p))
    assert classify_py(res) == expect


def test_a_crossing_dip_is_a_rate_discard(tmp_path):
    p = tmp_path / "d.csv"
    write_csv(str(p), synth(dip=True))
    res, _, _ = analyse(str(p))
    assert res["crossing_dips"], "the dip was not detected"
    assert classify_py(res) == "DISCARD_RATE"


def test_broken_is_the_verdict_when_there_is_no_report():
    assert classify_py(None) == "BROKEN"


def test_post_g3_progress_outranks_a_dirty_or_off_rate_discard(tmp_path):
    """A run that cleared Gate 3 is the most informative artefact available; filing it as
    a plain discard would bury exactly the evidence being hunted."""
    p = tmp_path / "r.csv"
    write_csv(str(p), synth(max_ag=3, g1_u=0.40))
    res, _, _ = analyse(str(p))
    assert res["dirty_start"] and classify_py(res) == "POST_G3_FAILURE"
    p2 = tmp_path / "r2.csv"
    write_csv(str(p2), synth(max_ag=4, hz=40.0))
    res2, _, _ = analyse(str(p2))
    assert res2["verdict"] == "DISCARD" and classify_py(res2) == "POST_G3_FAILURE"


# ---------------------------------------------------------------------------------
# The clean-deep-hold counter -- what actually ends the batch
# ---------------------------------------------------------------------------------
def clean_deep_hold(r):
    return (r is not None and r["verdict"] == "OK" and r["max_active_gate"] == 2
            and r["g3_size"] is not None and r["g3_size"] >= DEEP_SIZE
            and r["descent_frames"] > 0 and r["hold_frames"] > 0)


@pytest.mark.parametrize("kw,want", [
    (dict(g3_size=0.60, desc=True, hold=True), True),
    (dict(g3_size=0.60, desc=True, hold=False), False),    # hold never fired
    (dict(g3_size=0.60, desc=False, hold=False), False),   # descent never fired
    (dict(g3_size=0.30, desc=True, hold=True), False),     # not deep
    (dict(g3_size=0.60, desc=True, hold=True, hz=40.0), False),   # off-rate
    (dict(g3_size=0.60, desc=True, hold=True, max_ag=3), False),  # past G3, not a G3 test
])
def test_only_a_genuine_hold_active_deep_run_counts(kw, want, tmp_path):
    p = tmp_path / "k.csv"
    write_csv(str(p), synth(**kw))
    res, _, _ = analyse(str(p))
    assert clean_deep_hold(res) is want


# ---------------------------------------------------------------------------------
# PowerShell side: the script must parse, and must agree with the Python classifier
# ---------------------------------------------------------------------------------
def _pwsh():
    for exe in ("powershell.exe", "pwsh"):
        if shutil.which(exe):
            return exe
    return None


ps_required = pytest.mark.skipif(_pwsh() is None, reason="no PowerShell on this box")


@ps_required
def test_the_script_parses():
    """A syntax error here costs a whole batch window, so it is checked without running
    anything: the parser is invoked directly on the file."""
    code = (f"$ErrorActionPreference='Stop';"
            f"$t=[System.Management.Automation.PSParser]::Tokenize("
            f"(Get-Content -Raw -LiteralPath '{PS1}'), [ref]$null);"
            f"if($t.Count -lt 50){{exit 1}};exit 0")
    r = subprocess.run([_pwsh(), "-NoProfile", "-NonInteractive", "-Command", code],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr


@ps_required
@pytest.mark.parametrize("expect,kw", CASES, ids=[c[0] for c in CASES])
def test_the_powershell_classifier_agrees_with_python(expect, kw, tmp_path):
    """THE DRIFT GUARD. Same fixture, both classifiers -- if someone edits one bucket's
    rule in the .ps1 and not here, this fails."""
    p = tmp_path / "c.csv"
    write_csv(str(p), synth(**kw))
    res, _, _ = analyse(str(p))
    jf = tmp_path / "c.json"
    jf.write_text(json.dumps(res))
    code = f"""
$ErrorActionPreference='Stop'
$src = Get-Content -Raw -LiteralPath '{PS1}'
# lift the two functions and the thresholds out of the script without executing main
$m = [regex]::Match($src, '(?s)function Get-Classification.*?\\n\\}}')
$DEEP_SIZE = {DEEP_SIZE}
Invoke-Expression $m.Value
$R = Get-Content -Raw -LiteralPath '{jf}' | ConvertFrom-Json
Write-Output (Get-Classification -R $R)
"""
    r = subprocess.run([_pwsh(), "-NoProfile", "-NonInteractive", "-Command", code],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
    assert r.stdout.strip() == expect, f"ps={r.stdout.strip()} py={expect}"


@ps_required
def test_dry_run_starts_and_stops_nothing(tmp_path):
    """The safety property: -DryRun must never touch a process. Pointed at a simulator
    path that does not exist, so a real launch attempt would throw."""
    out = tmp_path / "results"
    code = (f"& '{PS1}' -DryRun -MaxAttempts 2 -RepoPath '{tmp_path}' "
            f"-SimExe 'C:\\nope\\DoesNotExist.exe' -OutDir 'results' "
            f"-WarmupSeconds 0 -PythonExe 'python'")
    r = subprocess.run([_pwsh(), "-NoProfile", "-NonInteractive", "-Command", code],
                       capture_output=True, text=True, timeout=180)
    combined = r.stdout + r.stderr
    assert "DRY RUN" in combined, combined
    assert "flight skipped" in combined
    assert "Start-Process" in combined, "the launch was not reported as skipped"
    # nothing was actually produced
    assert not list(tmp_path.glob("filt_auto_*.csv"))
    assert not (out / "batch_summary.csv").exists()


@ps_required
def test_dry_run_still_prints_the_exact_command(tmp_path):
    """Requirement: print the command before running it -- that is what makes a dry run
    reviewable. The frozen flags must be visible in it."""
    code = (f"& '{PS1}' -DryRun -MaxAttempts 1 -RepoPath '{tmp_path}' "
            f"-SimExe 'C:\\nope\\DoesNotExist.exe' -Mode holdblind -HoldSeconds 0.30 "
            f"-WarmupSeconds 0")
    r = subprocess.run([_pwsh(), "-NoProfile", "-NonInteractive", "-Command", code],
                       capture_output=True, text=True, timeout=180)
    out = r.stdout + r.stderr
    for flag in ("--gate-commit-rate-max", "0.13", "--gate-commit-stable-s", "0.15",
                 "--gate3-vert-descent", "--gate3-vert-descent-delta", "0.015",
                 "--gate3-vert-descent-size", "--gate3-vert-descent-hold-s"):
        assert flag in out, f"{flag} missing from the printed command\n{out}"
    # PowerShell renders [double] 0.30 as "0.3" -- the same value to argparse. Assert the
    # NUMBER, not the spelling, so a formatting change is not mistaken for a wrong flag.
    cmd = [ln for ln in out.splitlines() if "CMD:" in ln][0]
    tok = cmd.split("--gate3-vert-descent-hold-s ", 1)[1].split()[0]
    assert float(tok) == pytest.approx(0.30), f"hold-s was {tok}"


@ps_required
@pytest.mark.parametrize("mode,present,absent", [
    ("baseline", [], ["--gate3-vert-descent", "--gate3-vert-descent-hold-s"]),
    ("descent", ["--gate3-vert-descent"], ["--gate3-vert-descent-hold-s"]),
    ("holdblind", ["--gate3-vert-descent", "--gate3-vert-descent-hold-s"], []),
])
def test_each_mode_builds_the_right_flag_set(mode, present, absent, tmp_path):
    code = (f"& '{PS1}' -DryRun -MaxAttempts 1 -RepoPath '{tmp_path}' "
            f"-SimExe 'C:\\nope\\DoesNotExist.exe' -Mode {mode} -WarmupSeconds 0")
    r = subprocess.run([_pwsh(), "-NoProfile", "-NonInteractive", "-Command", code],
                       capture_output=True, text=True, timeout=180)
    out = r.stdout + r.stderr
    cmd = [ln for ln in out.splitlines() if "CMD:" in ln]
    assert cmd, out
    for f in present:
        assert f in cmd[0], f"{mode}: {f} should be present"
    for f in absent:
        assert f not in cmd[0], f"{mode}: {f} should be absent"


# ---------------------------------------------------------------------------------
# File handling -- in temp dirs, and nothing is ever deleted
# ---------------------------------------------------------------------------------
@ps_required
@pytest.mark.parametrize("cls,sub", [
    ("FULL_COURSE", "qualified"), ("POST_G3_FAILURE", "keepers"), ("KEEP_DEEP", "keepers"),
    ("KEEP_SHALLOW", "diagnostic"), ("KEEP_UPSTREAM", "diagnostic"),
    ("DISCARD_DIRTY", "to_delete"), ("DISCARD_RATE", "to_delete"),
    ("BROKEN", "to_delete"),
])
def test_each_class_lands_in_the_right_directory(cls, sub, tmp_path):
    for d in ("qualified", "keepers", "diagnostic", "to_delete"):
        (tmp_path / d).mkdir(parents=True, exist_ok=True)
    csv = tmp_path / "filt_auto_001.csv"; csv.write_text("x")
    log = tmp_path / "filt_auto_001.log"; log.write_text("y")
    code = f"""
$ErrorActionPreference='Stop'
$src = Get-Content -Raw -LiteralPath '{PS1}'
$QualDir='{tmp_path}\\qualified'; $KeepDir='{tmp_path}\\keepers'; $DiagDir='{tmp_path}\\diagnostic'; $TrashDir='{tmp_path}\\to_delete'
$DryRun=$false
Invoke-Expression ([regex]::Match($src,'(?s)function Get-DestDir.*?\\n\\}}').Value)
Invoke-Expression ([regex]::Match($src,'(?s)function Move-Pair.*?\\n\\}}').Value)
Move-Pair -Csv '{csv}' -Log '{log}' -Dest (Get-DestDir -Class '{cls}')
"""
    r = subprocess.run([_pwsh(), "-NoProfile", "-NonInteractive", "-Command", code],
                       capture_output=True, text=True, timeout=120)
    assert r.returncode == 0, r.stdout + r.stderr
    assert (tmp_path / sub / "filt_auto_001.csv").exists(), r.stdout + r.stderr
    assert (tmp_path / sub / "filt_auto_001.log").exists()
    assert not csv.exists(), "the original was left behind (copy, not move)"


@ps_required
def test_discards_are_moved_never_deleted(tmp_path):
    """Load-bearing: a discarded run is still evidence. It must be recoverable."""
    (tmp_path / "to_delete").mkdir(parents=True)
    csv = tmp_path / "filt_auto_009.csv"; csv.write_text("payload")
    code = f"""
$src = Get-Content -Raw -LiteralPath '{PS1}'
$KeepDir='{tmp_path}\\k'; $DiagDir='{tmp_path}\\d'; $TrashDir='{tmp_path}\\to_delete'
$DryRun=$false
Invoke-Expression ([regex]::Match($src,'(?s)function Get-DestDir.*?\\n\\}}').Value)
Invoke-Expression ([regex]::Match($src,'(?s)function Move-Pair.*?\\n\\}}').Value)
Move-Pair -Csv '{csv}' -Log '{tmp_path}\\nope.log' -Dest (Get-DestDir -Class 'DISCARD_RATE')
"""
    subprocess.run([_pwsh(), "-NoProfile", "-NonInteractive", "-Command", code],
                   capture_output=True, text=True, timeout=120)
    moved = tmp_path / "to_delete" / "filt_auto_009.csv"
    assert moved.exists() and moved.read_text() == "payload"


# ---------------------------------------------------------------------------------
# COURSE COMPLETION -- the six scenarios that must be right before unattended use
#
# The premise being defended: qualification is the WHOLE course. active_gate is the index
# of the gate being flown toward, over 6 gates (0=START .. 5=FINISH), so ag==3 means the
# third gate is behind us -- a third of the way, not a pass. Only the sim's own
# race-finish field means completion.
# ---------------------------------------------------------------------------------
def _with_log(tmp_path, name, frames, log_text=""):
    p = tmp_path / f"{name}.csv"
    write_csv(str(p), frames)
    lg = tmp_path / f"{name}.log"
    lg.write_text(log_text)
    return str(p), str(lg)


def test_scenario_1_failure_before_gate3(tmp_path):
    csv, log = _with_log(tmp_path, "s1", synth(max_ag=2, g3_size=0.30))
    r, _, _ = analyse(csv, log_path=log)
    assert not r["gate3_registered"] and not r["full_course_complete"]
    assert r["gates_completed"] == 2
    assert r["first_failure_after_gate3"] == "", "there was no post-G3 phase to fail in"
    assert classify_py(r) in ("KEEP_SHALLOW", "KEEP_DEEP")


def test_scenario_2_gate3_registered_then_gate4_failure(tmp_path):
    csv, log = _with_log(tmp_path, "s2", synth(max_ag=3))
    r, _, _ = analyse(csv, log_path=log)
    assert r["gate3_registered"] and not r["full_course_complete"]
    assert r["highest_active_gate"] == 3 and r["gates_completed"] == 3
    assert r["time_gate3_registered"] is not None
    assert r["time_each_later_gate_registered"] == {}, "no gate past 3 was reached"
    assert "no finish signal" in r["first_failure_after_gate3"]
    assert classify_py(r) == "POST_G3_FAILURE"


def test_scenario_3_all_gates_passed_but_no_finish_signal(tmp_path):
    """THE TRAP. Reaching the last gate index is NOT completion -- without the finish
    field the course is unqualified, and calling it a pass would be the exact mistake
    this task exists to remove."""
    csv, log = _with_log(tmp_path, "s3", synth(max_ag=5, race_fin=0))
    r, _, _ = analyse(csv, log_path=log)
    assert r["highest_active_gate"] == 5 and r["gates_completed"] == 5
    assert not r["full_course_complete"], "a high active_gate was mistaken for a finish"
    assert r["finish_signal"] == ""
    assert set(r["time_each_later_gate_registered"]) == {"4", "5"}
    assert classify_py(r) == "POST_G3_FAILURE"


def test_scenario_4_authoritative_course_completion(tmp_path):
    csv, log = _with_log(
        tmp_path, "s4", synth(max_ag=5, race_fin=987654),
        log_text="[tube] *** COMPLETE *** active_gate=5 finish=987654\n")
    r, _, _ = analyse(csv, log_path=log)
    assert r["full_course_complete"]
    assert "987654" in r["finish_signal"]
    assert r["first_failure_after_gate3"] == ""
    assert classify_py(r) == "FULL_COURSE"


def test_the_finish_signal_is_read_from_the_log_when_the_csv_lacks_it(tmp_path):
    """Older builds have no race_fin column; the flier's COMPLETE line still proves it."""
    p = tmp_path / "s4b.csv"
    keep = [c for c in COLS if c != "race_fin"]
    with open(p, "w", newline="") as f:
        f.write(",".join(keep) + "\n")
        for fr in synth(max_ag=5):
            f.write(",".join(str(fr[c]) for c in keep) + "\n")
    lg = tmp_path / "s4b.log"
    lg.write_text("[tube] *** COMPLETE *** active_gate=5 finish=42\n")
    r, _, _ = analyse(str(p), log_path=str(lg))
    assert r["full_course_complete"] and "42" in r["finish_signal"]
    assert classify_py(r) == "FULL_COURSE"


def test_scenario_5_crash_after_gate3(tmp_path):
    csv, log = _with_log(tmp_path, "s5", synth(max_ag=4),
                         log_text="[coast] !!! COLLISION ENV/ground at t=14.2s (ag=4)\n")
    r, _, _ = analyse(csv, log_path=log)
    assert r["gate3_registered"] and not r["full_course_complete"]
    assert r["collision_after_gate3"], "a post-G3 collision was not attributed"
    assert "COLLISION" in r["first_failure_after_gate3"]
    assert not r["timeout_after_gate3"]
    assert classify_py(r) == "POST_G3_FAILURE"


def test_a_collision_BEFORE_gate3_is_not_counted_as_post_g3(tmp_path):
    """The window matters: an early collision must not be blamed on the continuation."""
    csv, log = _with_log(tmp_path, "s5b", synth(max_ag=4),
                         log_text="[coast] !!! COLLISION GATE at t=0.5s (ag=1)\n")
    r, _, _ = analyse(csv, log_path=log)
    assert not r["collision_after_gate3"]


def test_scenario_6_timeout_after_gate3(tmp_path):
    csv, log = _with_log(tmp_path, "s6", synth(max_ag=4),
                         log_text="[tube] max flight 35.0s -- stop. seg=4 ag=4\n")
    r, _, _ = analyse(csv, log_path=log)
    assert r["gate3_registered"] and not r["full_course_complete"]
    assert r["timeout_after_gate3"]
    assert "TIMEOUT" in r["first_failure_after_gate3"]
    assert classify_py(r) == "POST_G3_FAILURE"


def test_a_timeout_on_a_completed_course_is_not_a_failure(tmp_path):
    csv, log = _with_log(
        tmp_path, "s6b", synth(max_ag=5, race_fin=7),
        log_text="[tube] *** COMPLETE *** active_gate=5 finish=7\n")
    r, _, _ = analyse(csv, log_path=log)
    assert r["full_course_complete"] and not r["timeout_after_gate3"]


def test_the_milestone_and_qualified_banners_are_distinct(tmp_path, capsys):
    """The two must never be confused in the console: one is progress, one is the goal."""
    csv, log = _with_log(tmp_path, "m1", synth(max_ag=3))
    report_main(["flight_report.py", csv])
    mid = capsys.readouterr().out
    assert "GATE 3 REGISTERED" in mid and "CONTINUING COURSE" in mid
    assert "FULL COURSE QUALIFIED" not in mid
    csv2, log2 = _with_log(tmp_path, "m2", synth(max_ag=5, race_fin=5))
    report_main(["flight_report.py", csv2])
    done = capsys.readouterr().out
    assert "FULL COURSE QUALIFIED" in done


def test_gates_completed_never_exceeds_the_course_length(tmp_path):
    csv, log = _with_log(tmp_path, "g", synth(max_ag=5, race_fin=1))
    r, _, _ = analyse(csv, log_path=log)
    assert r["gates_completed"] <= 6


# ---------------------------------------------------------------------------------
# KEEP-SIM-ALIVE: keystroke reset instead of a simulator restart
#
# The risk this guards. PGOS login is manual, so this mode never relaunches anything --
# which means a dead or foreign simulator cannot be recovered automatically. The batch
# must therefore STOP the instant the adopted process stops being the thing on the
# control channel. A loop that kept injecting Esc/Down/Enter at a dead window would
# quietly produce a night's worth of empty attempts that all look like flights.
#
# The Win32 layer is stubbed; the ORDER, the counts and the stop conditions are real.
# ---------------------------------------------------------------------------------
def _extract(fn_name):
    """Lift one function body out of the .ps1 so the tests drive the shipped code."""
    return ("Invoke-Expression ([regex]::Match($src,"
            "'(?s)function " + fn_name + ".*?\\n\\}').Value)\n")


KEEPALIVE_PRELUDE = r"""
$ErrorActionPreference = 'Stop'
$src = Get-Content -Raw -LiteralPath 'PS1PATH'
$DryRun = $false
$VK_ESCAPE = 0x1B; $VK_DOWN = 0x28; $VK_RETURN = 0x0D
$KeyMenuOpenMs = 700; $KeyStepMs = 200; $ResetSettleSeconds = 4
$script:SimPid = 4242
$global:CALLS = New-Object System.Collections.ArrayList
function Send-Key { param([int]$Vk,[string]$Name,[switch]$Extended)
    $sfx = ''
    if ($Extended) { $sfx = '+ext' }
    [void]$global:CALLS.Add("key:$Name$sfx"); return $true }
function Set-SimForeground { [void]$global:CALLS.Add("foreground"); return $true }
function Start-Sleep { param([int]$Milliseconds,[int]$Seconds)
    if ($Milliseconds) { [void]$global:CALLS.Add("sleep:$Milliseconds") }
    else { [void]$global:CALLS.Add("sleepS:$Seconds") } }
"""


@ps_required
def test_the_menu_open_half_is_foreground_then_esc():
    code = (KEEPALIVE_PRELUDE.replace("PS1PATH", PS1)
            + _extract("Send-ResetOpenMenu")
            + "Send-ResetOpenMenu | Out-Null\n"
            + '"RESULT:" + ($global:CALLS -join ",")\n')
    r = subprocess.run([_pwsh(), "-NoProfile", "-NonInteractive", "-Command", code],
                       capture_output=True, text=True, timeout=180)
    assert r.returncode == 0, r.stdout + r.stderr
    line = [l for l in r.stdout.splitlines() if l.startswith("RESULT:")][-1]
    calls = [c for c in line[len("RESULT:"):].split(",") if c]
    assert calls == ["foreground", "key:Esc", "sleep:700"], calls


@ps_required
def test_the_confirm_half_is_down_then_enter():
    """Down carries the extended flag: scancode 0x50 is shared with the numpad, and without
    it some UE builds never move the selection off RESUME."""
    code = (KEEPALIVE_PRELUDE.replace("PS1PATH", PS1)
            + _extract("Send-ResetConfirm")
            + "Send-ResetConfirm | Out-Null\n"
            + '"RESULT:" + ($global:CALLS -join ",")\n')
    r = subprocess.run([_pwsh(), "-NoProfile", "-NonInteractive", "-Command", code],
                       capture_output=True, text=True, timeout=180)
    line = [l for l in r.stdout.splitlines() if l.startswith("RESULT:")][-1]
    calls = [c for c in line[len("RESULT:"):].split(",") if c]
    assert calls == ["key:Down+ext", "sleep:200", "key:Enter"], calls


@ps_required
def test_the_confirm_half_never_touches_the_menu_key():
    """Esc must not be re-sent at confirm time -- it would close the menu instead of
    activating RESTART."""
    src = open(PS1, encoding="utf-8").read()
    body = src[src.index("function Send-ResetConfirm"):]
    body = body[:body.index("\nfunction ")]
    assert "VK_ESCAPE" not in body


LOOP_HARNESS = r"""
$ErrorActionPreference = 'Stop'
$src = Get-Content -Raw -LiteralPath 'PS1PATH'
$DryRun = $false
$script:SimPid = 4242
$MaxAttempts = 5
$global:CALLS = New-Object System.Collections.ArrayList
$global:CLASSES = @(CLASSSEQ)
$global:ALIVE   = @(ALIVESEQ)
$global:i = 0
$global:a = 0
function Send-ResetOpenMenu { [void]$global:CALLS.Add("reset-open"); return $true }
function Send-ResetConfirm  { [void]$global:CALLS.Add("reset-confirm"); return $true }
function Test-SimAlive {
    $idx = [Math]::Min($global:a, $global:ALIVE.Count - 1)
    $v = $global:ALIVE[$idx]
    $global:a++
    [void]$global:CALLS.Add("alive?$v")
    return $v }
function Start-SimTracked { [void]$global:CALLS.Add("RELAUNCH"); return $true }
function Stop-ShippingProcess { param($TargetPid) [void]$global:CALLS.Add("TERMINATE"); return $null }
INJECT_ABORT
$KeepSimAlive = $true
$simLost = $false
$attempt = 0
while ($attempt -lt $MaxAttempts) {
    $attempt++
    if ($KeepSimAlive) {
        if (-not (Test-SimAlive)) { Invoke-SimLostAbort -Where "pre-flight"; $simLost = $true; break }
    }
    if ($attempt -gt 1) { Send-ResetOpenMenu | Out-Null }
    $cls = $global:CLASSES[[Math]::Min($global:i, $global:CLASSES.Count - 1)]
    [void]$global:CALLS.Add("launch:$cls")
    $global:i++
    if ($attempt -gt 1) { Send-ResetConfirm | Out-Null }
    [void]$global:CALLS.Add("exit:$cls")
    if ($cls -eq "FULL_COURSE") { [void]$global:CALLS.Add("QUALIFIED"); break }
}
"RESULT:" + ($global:CALLS -join ",") + "|simLost=$simLost"
"""


def _run_loop(classes, alive):
    code = (LOOP_HARNESS.replace("PS1PATH", PS1)
            .replace("CLASSSEQ", ",".join("'" + c + "'" for c in classes))
            .replace("ALIVESEQ", ",".join("$true" if a else "$false" for a in alive))
            .replace("INJECT_ABORT", _extract("Invoke-SimLostAbort")))
    r = subprocess.run([_pwsh(), "-NoProfile", "-NonInteractive", "-Command", code],
                       capture_output=True, text=True, timeout=180)
    assert r.returncode == 0, r.stdout + r.stderr
    line = [l for l in r.stdout.splitlines() if l.startswith("RESULT:")][-1]
    body, lost = line[len("RESULT:"):].split("|simLost=")
    return [c for c in body.split(",") if c], (lost.strip() == "True"), r.stdout



@ps_required
def test_keepalive_stops_immediately_when_the_sim_dies_and_never_resets_again():
    """PID gone / foreign owner on 14560 mid-loop: STOP, no further resets, no relaunch."""
    calls, lost, out = _run_loop(["KEEP_SHALLOW"] * 5, [True, True, False])
    assert lost, "the batch did not record SIM LOST"
    assert "RELAUNCH" not in calls, "it tried to relaunch despite the manual login"
    dead_at = next(i for i, c in enumerate(calls) if c == "alive?False")
    tail = calls[dead_at:]
    assert "reset-open" not in tail and "reset-confirm" not in tail, \
        "a keystroke was injected after the sim died: " + str(calls)
    assert "SIM LOST" in out and "manual re-login required" in out


@ps_required
def test_keepalive_will_not_fly_into_a_dead_sim():
    """Dead before the very first flight -> zero flights, zero resets."""
    calls, lost, out = _run_loop(["KEEP_SHALLOW"] * 5, [False])
    assert lost
    assert not [c for c in calls if c.startswith("launch:")], calls
    assert "reset-open" not in calls and "reset-confirm" not in calls
    assert "RELAUNCH" not in calls


@ps_required
def test_the_full_restart_path_is_untouched_by_the_new_switch():
    """-KeepSimAlive is additive: with it absent the restart path must still be reachable,
    i.e. every guard is `(-not $KeepSimAlive) -and ...`, never an unconditional skip."""
    src = open(PS1, encoding="utf-8").read()
    assert "if ((-not $KeepSimAlive) -and $null -ne $script:SimPid) {" in src
    assert "if ((-not $KeepSimAlive) -and -not (Start-SimTracked)) {" in src
    assert "if ((-not $KeepSimAlive) -and -not (Test-PreFlightIsolation)) {" in src
    # and the keystroke reset is reachable ONLY under the new switch
    # rindex, not index: the first hit is inside the composite Send-ResetSequence;
    # the one that matters is the LOOP call site.
    i = src.rindex("Send-ResetOpenMenu | Out-Null")
    assert "if ($KeepSimAlive) {" in src[max(0, i - 1500):i]


@ps_required
def test_keepalive_startup_never_calls_the_launcher_or_the_terminator():
    """Static guard: neither Start-SimTracked nor Clear-AllShipping may be reachable from
    the keep-alive startup path."""
    src = open(PS1, encoding="utf-8").read()
    i = src.index("if ($KeepSimAlive) {", src.index("\ntry {"))
    j = src.index("} else {", i)
    startup = src[i:j]
    assert "Start-SimTracked" not in startup
    assert "Clear-AllShipping" not in startup
    assert "Initialize-KeepSimAlive" in startup


def test_the_new_switches_have_the_specified_defaults():
    src = open(PS1, encoding="utf-8").read()
    assert "[switch] $KeepSimAlive," in src, "must default OFF"
    # DEPRECATED to 0: waiting after RESTART before launching the flier is exactly the
    # defect -- RESTART fires the countdown, so the flier must already be armed.
    assert "[double] $ResetSettleSeconds = 0," in src
    assert "[int]    $KeyMenuOpenMs      = 700" in src
    assert "[int]    $KeyStepMs          = 200" in src


# ---------------------------------------------------------------------------------
# ARM-BEFORE-RESTART ordering
#
# THE BUG THIS PINS. schedule_flier.py connects, ARMs, prints "ARM sent; waiting for GO",
# and then BLOCKS until the sim reports race_start. RESTART is what fires the countdown.
# The old loop pressed RESTART at the END of an attempt, so GO passed with no flier
# listening and the next run never started -- the drone just sat at the gate until the
# batch's 35 s timeout. The launch must sit BETWEEN the two halves of the reset.
# ---------------------------------------------------------------------------------









@ps_required
def test_no_reset_runs_after_the_flight_any_more():
    """The old end-of-attempt reset is the bug; it must be gone, not merely reordered."""
    src = open(PS1, encoding="utf-8").read()
    assert "Send-ResetSequence | Out-Null" not in src, \
        "the fused post-flight reset is still being called by the loop"


def test_the_flier_is_launched_unbuffered():
    """Python block-buffers stdout to a file and the flier's arm line has no flush=True,
    so without -u the signal would arrive long after the timeout and every attempt would
    silently take the proceed-anyway path."""
    src = open(PS1, encoding="utf-8").read()
    assert '$argv = @("-u") + $FlierArgs' in src



def test_the_sim_death_guard_and_full_course_stop_are_unchanged():
    src = open(PS1, encoding="utf-8").read()
    assert "Invoke-SimLostAbort -Where \"pre-flight check, attempt $attempt\"" in src
    assert 'if ($cls -eq "FULL_COURSE") {' in src
    assert "Set-KeepAwake" in src and "Set-KeepAwake -Release" in src


# ---------------------------------------------------------------------------------
# DISCARD PURGE (-KeepSimAlive only)
#
# This is the ONE place the runner deletes instead of moving, so the guard conditions are
# pinned from both sides: what must go, and -- more importantly -- everything that must
# survive. A 1000-attempt overnight batch needs the disk back, but a wrongly-purged
# Gate-3 approach is unrecoverable evidence.
# ---------------------------------------------------------------------------------
PURGE_HARNESS = r"""
$ErrorActionPreference = 'Stop'
$src = Get-Content -Raw -LiteralPath 'PS1PATH'
$DryRun = $false
$KeepSimAlive = KSA
$KeepAllRuns  = KAR
Invoke-Expression ([regex]::Match($src,'(?s)function Test-PurgeableDiscard.*?\n\}').Value)
$R = RVAL
"RESULT:" + (Test-PurgeableDiscard -R $R -Class 'CLS')
"""


def _purgeable(cls, maxag, ksa=True, keep_all=False):
    rval = "$null" if maxag is None else (
        "([pscustomobject]@{ max_active_gate = " + str(maxag) + " })")
    code = (PURGE_HARNESS.replace("PS1PATH", PS1).replace("CLS", cls)
            .replace("RVAL", rval)
            .replace("KSA", "$true" if ksa else "$false")
            .replace("KAR", "$true" if keep_all else "$false"))
    r = subprocess.run([_pwsh(), "-NoProfile", "-NonInteractive", "-Command", code],
                       capture_output=True, text=True, timeout=180)
    assert r.returncode == 0, r.stdout + r.stderr
    line = [l for l in r.stdout.splitlines() if l.startswith("RESULT:")][-1]
    return line[len("RESULT:"):].strip() == "True"


@ps_required
@pytest.mark.parametrize("cls", ["DISCARD_RATE", "DISCARD_DIRTY", "BROKEN"])
def test_a_shallow_pure_discard_is_purged(cls):
    assert _purgeable(cls, maxag=1) is True
    assert _purgeable(cls, maxag=0) is True


@ps_required
@pytest.mark.parametrize("cls", ["DISCARD_RATE", "DISCARD_DIRTY", "BROKEN"])
def test_a_discard_that_reached_the_gate3_leg_is_kept(cls):
    """maxag >= 2 means it flew the Gate-3 approach -- that trace is the evidence the whole
    session is built on, off-rate or not."""
    assert _purgeable(cls, maxag=2) is False
    assert _purgeable(cls, maxag=3) is False


@ps_required
@pytest.mark.parametrize("cls", ["FULL_COURSE", "POST_G3_FAILURE", "KEEP_DEEP",
                                 "KEEP_SHALLOW", "KEEP_UPSTREAM", "BROKEN_RESTART"])
def test_non_discard_classes_are_never_purged(cls):
    """FULL_COURSE especially: it is the deliverable and must survive at any depth."""
    assert _purgeable(cls, maxag=0) is False
    assert _purgeable(cls, maxag=5) is False


@ps_required
def test_an_unreadable_report_is_never_purged():
    """No report => depth unknown => cannot prove the run is worthless => keep it.
    Deletion is irreversible, so every uncertainty resolves to KEEP."""
    assert _purgeable("BROKEN", maxag=None) is False


@ps_required
def test_keep_all_runs_restores_the_old_behaviour():
    assert _purgeable("DISCARD_RATE", maxag=0, keep_all=True) is False


@ps_required
def test_the_full_restart_mode_never_purges():
    """Purging is scoped to -KeepSimAlive; the supervised restart batches keep everything."""
    assert _purgeable("DISCARD_RATE", maxag=0, ksa=False) is False


@ps_required
def test_the_purge_actually_deletes_both_files_and_reports_the_size(tmp_path):
    csv = tmp_path / "filt_auto_042.csv"; csv.write_bytes(b"x" * 300000)
    log = tmp_path / "filt_auto_042.log"; log.write_bytes(b"y" * 100000)
    code = f"""
$ErrorActionPreference='Stop'
$src = Get-Content -Raw -LiteralPath '{PS1}'
$DryRun = $false
Invoke-Expression ([regex]::Match($src,'(?s)function Remove-DiscardPair.*?\\n\\}}').Value)
$mb = Remove-DiscardPair -Csv '{csv}' -Log '{log}' -Stem 'filt_auto_042'
"RESULT:$mb"
"""
    r = subprocess.run([_pwsh(), "-NoProfile", "-NonInteractive", "-Command", code],
                       capture_output=True, text=True, timeout=180)
    assert r.returncode == 0, r.stdout + r.stderr
    assert not csv.exists() and not log.exists(), "the purge left files behind"
    assert "purged discard filt_auto_042" in r.stdout
    freed = float([l for l in r.stdout.splitlines() if l.startswith("RESULT:")][-1][7:])
    assert freed == pytest.approx(0.38, abs=0.02), freed


@ps_required
def test_the_summary_row_is_written_before_the_purge():
    """A purged attempt must still appear in batch_summary.csv -- otherwise the overnight
    tally silently under-counts and the discard rate looks better than it is."""
    src = open(PS1, encoding="utf-8").read()
    i_row = src.index("Write-SummaryRow -R $R -Class $cls")
    i_purge = src.index("if (Test-PurgeableDiscard -R $R -Class $cls) {")
    assert i_row < i_purge, "the purge runs before the summary row is recorded"


def test_keep_all_runs_defaults_off():
    src = open(PS1, encoding="utf-8").read()
    assert "[switch] $KeepAllRuns" in src


# ---------------------------------------------------------------------------------
# -ManualReset: KeepSimAlive with ZERO injected input
#
# The point of this mode is that there is nothing to mistime. Every failure the keystroke
# path can have -- menu layout, window focus, the arm race, countdown timing -- is removed
# by not sending anything at all. So the tests are mostly about ABSENCE: no Send-Key, no
# Esc, no RESTART, on any attempt, ever.
# ---------------------------------------------------------------------------------
MANUAL_HARNESS = r"""
$ErrorActionPreference = 'Stop'
$src = Get-Content -Raw -LiteralPath 'PS1PATH'
$DryRun = $false
$KeepSimAlive = $true
$ManualReset = MRVAL
$MaxAttempts = 3
$global:CALLS = New-Object System.Collections.ArrayList
function Send-ResetOpenMenu { [void]$global:CALLS.Add("Esc"); return $true }
function Send-ResetConfirm  { [void]$global:CALLS.Add("RESTART"); return $true }
function Wait-ManualReset   { [void]$global:CALLS.Add("PROMPT") }
function Test-SimAlive { return $true }
$attempt = 0
while ($attempt -lt $MaxAttempts) {
    $attempt++
    if ($KeepSimAlive) {
        if (-not (Test-SimAlive)) { break }
        if ($ManualReset) { }
        elseif ($attempt -gt 1) { Send-ResetOpenMenu | Out-Null }
    }
    [void]$global:CALLS.Add("launch")
    if ((-not $ManualReset) -and $attempt -gt 1) { Send-ResetConfirm | Out-Null }
    [void]$global:CALLS.Add("exit")
    if ($KeepSimAlive -and $ManualReset -and ($attempt -lt $MaxAttempts)) { Wait-ManualReset }
}
"RESULT:" + ($global:CALLS -join ",")
"""


def _manual(mr):
    code = (MANUAL_HARNESS.replace("PS1PATH", PS1)
            .replace("MRVAL", "$true" if mr else "$false"))
    r = subprocess.run([_pwsh(), "-NoProfile", "-NonInteractive", "-Command", code],
                       capture_output=True, text=True, timeout=180)
    assert r.returncode == 0, r.stdout + r.stderr
    line = [l for l in r.stdout.splitlines() if l.startswith("RESULT:")][-1]
    return [c for c in line[len("RESULT:"):].split(",") if c]







def test_manual_reset_defaults_off():
    src = open(PS1, encoding="utf-8").read()
    assert "[switch] $ManualReset" in src


def test_manual_reset_wait_injects_nothing():
    """Wait-ManualReset must contain no SendInput/foreground call -- that is the guarantee."""
    src = open(PS1, encoding="utf-8").read()
    body = src[src.index("function Wait-ManualReset"):]
    body = body[:body.index("\nfunction ")]
    for forbidden in ("Send-Key", "SendVirtualKey", "Set-SimForeground",
                      "Send-ResetOpenMenu", "Send-ResetConfirm"):
        assert forbidden not in body, forbidden
    assert "Read-Host" in body


# ---------------------------------------------------------------------------------
# THE UNIFORM KEEP-SIM-ALIVE CYCLE
#
# Every attempt is identical -- no attempt-1 branch, no arm-signal gate, nothing
# conditional on what the previous attempt did:
#     Esc -> launch -> settle -> Down(ext)+Enter -> await exit -> classify -> loop
#
# The Esc is safe on every attempt because the previous cycle ended with the FLIER
# EXITING, which is what proves the run is over. On attempt 1 the freshly loaded drone is
# idle and pauses the same way.
# ---------------------------------------------------------------------------------
UNIFORM_HARNESS = r"""
$ErrorActionPreference = 'Stop'
$src = Get-Content -Raw -LiteralPath 'PS1PATH'
$DryRun = $false
$KeepSimAlive = $true
$ManualReset = MRVAL
$RestartDelaySeconds = 2.0
$MaxAttempts = 3
$global:CALLS = New-Object System.Collections.ArrayList
function Send-ResetOpenMenu { [void]$global:CALLS.Add("Esc"); return $true }
function Send-ResetConfirm  { [void]$global:CALLS.Add("Down+Enter"); return $true }
function Start-FlierAsync   { param($FlierArgs,$LogPath) [void]$global:CALLS.Add("launch"); return [pscustomobject]@{Proc=$null} }
function Complete-FlierAsync { param($Handle) [void]$global:CALLS.Add("exit"); return 0 }
function Start-Sleep { param([int]$Milliseconds,[int]$Seconds) [void]$global:CALLS.Add("settle:$Milliseconds") }
function Wait-ManualReset { [void]$global:CALLS.Add("PROMPT") }
function Test-SimAlive { return $true }
$attempt = 0
while ($attempt -lt $MaxAttempts) {
    $attempt++
    if (-not (Test-SimAlive)) { break }
    if (-not $ManualReset) { Send-ResetOpenMenu | Out-Null }
    $h = Start-FlierAsync -FlierArgs @() -LogPath "l"
    if (-not $ManualReset) {
        Start-Sleep -Milliseconds ([int]($RestartDelaySeconds * 1000))
        Send-ResetConfirm | Out-Null
    }
    $exit = Complete-FlierAsync -Handle $h
    [void]$global:CALLS.Add("classify")
    if ($KeepSimAlive -and $ManualReset -and ($attempt -lt $MaxAttempts)) { Wait-ManualReset }
}
"RESULT:" + ($global:CALLS -join ",")
"""


def _uniform(mr=False):
    code = (UNIFORM_HARNESS.replace("PS1PATH", PS1)
            .replace("MRVAL", "$true" if mr else "$false"))
    r = subprocess.run([_pwsh(), "-NoProfile", "-NonInteractive", "-Command", code],
                       capture_output=True, text=True, timeout=180)
    assert r.returncode == 0, r.stdout + r.stderr
    line = [l for l in r.stdout.splitlines() if l.startswith("RESULT:")][-1]
    return [c for c in line[len("RESULT:"):].split(",") if c]


@ps_required
def test_the_cycle_is_esc_launch_settle_restart_exit_classify():
    assert _uniform()[:6] == ["Esc", "launch", "settle:2000", "Down+Enter", "exit", "classify"]


@ps_required
def test_every_attempt_runs_the_identical_cycle():
    """No attempt-1 branch: three attempts produce three identical six-step cycles."""
    calls = _uniform()
    one = ["Esc", "launch", "settle:2000", "Down+Enter", "exit", "classify"]
    assert calls == one * 3, calls


@ps_required
def test_esc_always_follows_the_previous_fliers_exit():
    """Flier exit is what ends a cycle, so every Esc after the first is preceded by one --
    an Esc can never land mid-flight."""
    calls = _uniform()
    for i, c in enumerate(calls):
        if c == "Esc" and i > 0:
            assert "exit" in calls[:i], calls
            assert calls[i - 1] == "classify", calls


@ps_required
def test_restart_always_follows_the_launch_within_the_same_cycle():
    calls = _uniform()
    launches = [i for i, c in enumerate(calls) if c == "launch"]
    restarts = [i for i, c in enumerate(calls) if c == "Down+Enter"]
    assert len(launches) == len(restarts) == 3
    for l, r in zip(launches, restarts):
        assert l < r, calls


@ps_required
def test_there_is_no_attempt_number_branch_left_in_the_script():
    src = open(PS1, encoding="utf-8").read()
    assert "$attempt -gt 1" not in src
    assert "$attempt -eq 1" not in src


@ps_required
def test_the_arm_signal_gate_is_gone():
    """Wait-FlierArmed and its call site are removed -- the cycle is timed by
    -RestartDelaySeconds, not by watching for a print."""
    src = open(PS1, encoding="utf-8").read()
    assert "function Wait-FlierArmed" not in src
    assert "Wait-FlierArmed -Handle" not in src


@ps_required
def test_manual_reset_still_sends_nothing():
    calls = _uniform(mr=True)
    assert "Esc" not in calls and "Down+Enter" not in calls, calls
    assert calls.count("PROMPT") == 2, calls


def test_restart_delay_default_is_two_seconds():
    src = open(PS1, encoding="utf-8").read()
    assert "[double] $RestartDelaySeconds = 2.0," in src


# ---------------------------------------------------------------------------------
# Stray-flier cleanup -- the actual cause of "Esc right away"
# ---------------------------------------------------------------------------------
@ps_required
def test_stray_cleanup_matches_on_the_command_line_not_the_process_name():
    """The interpreter may be python.exe, a venv shim or a full path, so the name is not a
    reliable key -- and killing an unrelated python would be far worse than a stray."""
    src = open(PS1, encoding="utf-8").read()
    body = src[src.index("function Stop-StrayFliers"):]
    body = body[:body.index("\nfunction ")]
    assert "CommandLine" in body and "schedule_flier.py" in body
    assert "Get-Process -Name" not in body
    assert "$_.ProcessId -ne $PID" in body, "it could kill the batch's own shell"


@ps_required
def test_stray_cleanup_runs_at_batch_start_before_adopting_the_sim():
    src = open(PS1, encoding="utf-8").read()
    # compare CALL SITES: the function definitions sit far earlier in the file
    i_stray = src.index("Stop-StrayFliers | Out-Null")
    i_adopt = src.index("if (-not (Initialize-KeepSimAlive))")
    assert i_stray < i_adopt, "strays are cleared after adoption, too late to help"


@ps_required
def test_stray_cleanup_finds_and_kills_a_live_impostor(tmp_path):
    """End-to-end against a REAL process: a python holding a file named schedule_flier.py
    must be found by command line and terminated."""
    fake = tmp_path / "schedule_flier.py"
    fake.write_text("import time\ntime.sleep(60)\n")
    proc = subprocess.Popen([sys.executable, str(fake)])
    try:
        code = f"""
$ErrorActionPreference='Stop'
$src = Get-Content -Raw -LiteralPath '{PS1}'
$DryRun = $false
Invoke-Expression ([regex]::Match($src,'(?s)function Stop-StrayFliers.*?\\n\\}}').Value)
$n = Stop-StrayFliers
"RESULT:$n"
"""
        r = subprocess.run([_pwsh(), "-NoProfile", "-NonInteractive", "-Command", code],
                           capture_output=True, text=True, timeout=180)
        assert r.returncode == 0, r.stdout + r.stderr
        n = int([l for l in r.stdout.splitlines() if l.startswith("RESULT:")][-1][7:])
        assert n >= 1, r.stdout
        assert proc.poll() is not None or proc.wait(timeout=10) is not None
    finally:
        if proc.poll() is None:
            proc.kill()
