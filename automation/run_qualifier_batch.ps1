<#
.SYNOPSIS
  VQ1 qualification batch runner: full sim restart -> one flight -> report -> classify.

.DESCRIPTION
  Automates the restart/fly/report/file-handling loop so attempts are comparable. Every
  flight gets its own simulator process -- two flights in one process is the single
  biggest source of non-comparable runs, so it is structurally prevented here.

  QUALIFICATION IS THE WHOLE COURSE. The course has 6 gates (0=START .. 5=FINISH) and
  active_gate is the index of the gate being flown toward, so ag==3 means the THIRD gate
  is behind us -- a milestone, not a pass. The batch therefore stops only on the sim's
  own race-finish signal, or on MaxAttempts. A run that clears Gate 3 and then dies is
  classified POST_G3_FAILURE and kept, because it is the most informative artefact
  available; the batch restarts and carries on.

  This script does NOT change flight behaviour. The flag block in Get-FlightArgs is the
  frozen command; edit it only when the frozen command itself changes.

.EXAMPLE
  # two clean deep runs with the blind hold actually active
  .\run_qualifier_batch.ps1 -Mode holdblind -CleanDeepRequired 2 -MaxAttempts 20

.EXAMPLE
  .\run_qualifier_batch.ps1 -DryRun -MaxAttempts 3
#>
[CmdletBinding()]
param(
    # ---- CONFIGURATION -------------------------------------------------------------
    # Located on this box; the only install present is the official Development Kit build.
    [string] $SimExe            = "C:\Users\nohab\Desktop\AI GP\VQ1\AI-GP Simulator v1.0.3385\AIGP_3385\FlightSim.exe",
    [string] $RepoPath          = "C:\Users\nohab\Desktop\AI GP\AI-GrandPrix-",
    [string] $PythonExe         = "python",
    [double] $WarmupSeconds     = 25.0,
    [switch] $ManualReady,          # pause for "course loaded and ready" before each flight
    [int]    $StartRun          = 1,
    [int]    $CleanDeepRequired = 2,
    [int]    $MaxAttempts       = 1000,   # overnight variance harvest; stops early on FULL_COURSE
    [ValidateSet("baseline", "descent", "holdblind")]
    [string] $Mode              = "holdblind",
    [double] $HoldSeconds       = 0.30,
    [string] $OutDir            = "results",
    [switch] $DryRun,
    [switch] $TestRestartLifecycle,
    # RATE GOVERNOR, default ON: adds --log-tube as flight-inert per-frame load to hold the
    # loop near the ~68 Hz the frozen tune was validated at. See Get-FlightArgs.
    # Clear with -RateBallast:$false.
    [switch] $RateBallast = $true,
    # HANDS-FREE RESET: never restart the sim; reset the run through the pause menu with
    # injected keystrokes. Login is manual, so the sim is launched and logged in ONCE.
    [switch] $KeepSimAlive,
    # DEPRECATED / unused. It used to wait after RESTART before launching the flier --
    # which is precisely the bug: RESTART fires the countdown, so a flier launched after it
    # misses GO entirely. The flier is now armed and waiting BEFORE the keystroke, so no
    # post-RESTART wait is needed. Kept only so existing command lines still parse.
    [double] $ResetSettleSeconds = 0,
    [int]    $KeyMenuOpenMs      = 700, # after Esc: let the pause menu finish opening
    [int]    $KeyStepMs          = 200, # between Down and Enter
    # Fixed settle between launching the flier and pressing RESTART. The flier needs this
    # long to connect, arm and start waiting for GO.
    [double] $RestartDelaySeconds = 2.0,
    # DEPRECATED, unused. The arm-signal gate is gone: the cycle is uniform and timed by
    # -RestartDelaySeconds. Kept so existing command lines still parse.
    [double] $ArmReadyTimeoutSeconds = 8,
    [double] $ArmReadySeconds        = 0.5,
    # Restore keep-everything. Without it, -KeepSimAlive purges worthless attempts (see
    # Test-PurgeableDiscard) so a 1000-run overnight batch does not fill the disk.
    [switch] $KeepAllRuns,
    # HANDS-ON RESET: -KeepSimAlive without any injected input. The batch flies, reports,
    # then waits at a prompt while YOU press Esc/Restart. Nothing is timed against the
    # countdown, so there is nothing to mistime.
    [switch] $ManualReset
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

# Thresholds the classifier uses. Kept here so a reclassification never needs a code edit.
$DEEP_SIZE     = 0.45      # corrected Gate-3 size at/above which an approach is "deep"
$DESCENT_SIZE  = 0.25      # --gate3-vert-descent-size, for the flicker diagnostic

# ------------------------------------------------------------------------------------
# Paths
# ------------------------------------------------------------------------------------
$Root       = Join-Path $RepoPath $OutDir
$QualDir    = Join-Path $Root "qualified"      # FULL-COURSE successes -- the deliverable
$KeepDir    = Join-Path $Root "keepers"
$DiagDir    = Join-Path $Root "diagnostic"
$TrashDir   = Join-Path $Root "to_delete"
$JsonDir    = Join-Path $Root "json"
$SummaryCsv = Join-Path $Root "batch_summary.csv"

function Initialize-Dirs {
    foreach ($d in @($Root, $QualDir, $KeepDir, $DiagDir, $TrashDir, $JsonDir)) {
        if (-not (Test-Path -LiteralPath $d)) {
            if ($DryRun) { Write-Host "[dry] mkdir $d" -ForegroundColor DarkGray }
            else { New-Item -ItemType Directory -Path $d -Force | Out-Null }
        }
    }
}

# ------------------------------------------------------------------------------------
# Simulator lifecycle
#
# THE DEFECT THIS REPLACES. FlightSim.exe is a ~167 KB Unreal LAUNCHER STUB: it spawns
# DCGame-Win64-Shipping.exe (~92 MB) and exits. Everything that matters -- the render
# loop, and both UDP sockets (14560 MAVLink source, 5601 vision source) -- lives in the
# shipping process. The old code matched the name "FlightSim", so it could only ever find
# the transient stub, and Stop-Sim never terminated the actual simulator. Observed
# directly: two shipping processes alive against a single surviving stub.
#
# So the stub PID is used for NOTHING except launching. The shipping PID is discovered by
# differencing the process list around the launch, stored, and is the only kill target.
# ------------------------------------------------------------------------------------
$ShippingName = "DCGame-Win64-Shipping"
$SimPorts     = @(14560, 5601)      # the SHIPPING process's own UDP source ports
$script:SimPid      = $null         # stored shipping PID -- the only kill target
$script:SimStart    = $null
$script:LauncherPid = $null

function Get-ShippingProcesses {
    # Shipping processes belonging to THIS install. Path-confirmed where the OS allows,
    # so another copy of the sim elsewhere on the box is never a candidate.
    $root = Split-Path -Parent $SimExe
    $procs = @(Get-Process -Name $ShippingName -ErrorAction SilentlyContinue)
    if ($procs.Count -eq 0) { return @() }
    $matched = @()
    foreach ($p in $procs) {
        $path = $null
        try { $path = $p.Path } catch { $path = $null }
        if ($null -eq $path) { $matched += $p }
        elseif ($path -like "$root*") { $matched += $p }
    }
    return $matched
}

function Get-PortOwners {
    # Which PIDs currently own the simulator's UDP source ports, so the lifecycle log can
    # show ownership before AND after termination.
    $rows = @()
    foreach ($prt in $SimPorts) {
        $eps = @(Get-NetUDPEndpoint -LocalPort $prt -ErrorAction SilentlyContinue)
        foreach ($e in $eps) {
            $nm = "?"
            try { $nm = (Get-Process -Id $e.OwningProcess -ErrorAction Stop).ProcessName } catch { $nm = "?" }
            $rows += [pscustomobject]@{ Port = $prt; ProcId = $e.OwningProcess; Name = $nm }
        }
    }
    return $rows
}

function Format-PortOwners {
    param($Rows)
    # @($null).Count is 1, NOT 0 -- so a null result must be tested for directly or the
    # formatter iterates once over $null and StrictMode trips on the missing property.
    if ($null -eq $Rows) { return "none" }
    $r = @($Rows)
    if ($r.Count -eq 0) { return "none" }
    return (($r | ForEach-Object { "$($_.Port)->PID $($_.ProcId) ($($_.Name))" }) -join ", ")
}

function Wait-PortsReleased {
    param([int] $TimeoutSec = 20)
    $deadline = (Get-Date).AddSeconds($TimeoutSec)
    while ((Get-Date) -lt $deadline) {
        if (@(Get-PortOwners).Count -eq 0) { return $true }
        Start-Sleep -Milliseconds 250
    }
    return $false
}

function Stop-ShippingProcess {
    # Clean first (CloseMainWindow), forced only after the grace period. Returns a
    # lifecycle record. NEVER kills by name.
    param([int] $TargetPid, [int] $GraceSec = 10, [int] $HardSec = 15,
          [int[]] $ExpectedOthers = @())
    $rec = [ordered]@{
        shipping_pid = $TargetPid
        stop_requested = (Get-Date -Format "HH:mm:ss.fff")
        ports_before = (Format-PortOwners (Get-PortOwners))
        forced = $false; exited = $false; exit_time = ""; exit_code = ""
        orphan = ""; ports_after = ""; ports_released = $false
    }
    $p = $null
    try { $p = Get-Process -Id $TargetPid -ErrorAction Stop } catch { $p = $null }
    if ($null -eq $p) {
        $rec.exited = $true
        $rec.exit_time = (Get-Date -Format "HH:mm:ss.fff")
        $rec.ports_released = (Wait-PortsReleased -TimeoutSec 20)
        $rec.ports_after = (Format-PortOwners (Get-PortOwners))
        return [pscustomobject]$rec
    }
    Write-Host "  sim: stopping shipping PID $TargetPid (clean)"
    try { $p.CloseMainWindow() | Out-Null } catch { }
    $deadline = (Get-Date).AddSeconds($GraceSec)
    while ((Get-Date) -lt $deadline) {
        $p.Refresh()
        if ($p.HasExited) { break }
        Start-Sleep -Milliseconds 250
    }
    $p.Refresh()
    if (-not $p.HasExited) {
        Write-Host "  sim: clean stop timed out -- forcing" -ForegroundColor Yellow
        $rec.forced = $true
        try { Stop-Process -Id $TargetPid -Force -ErrorAction Stop } catch {
            Write-Warning "  force-kill failed: $($_.Exception.Message)"
        }
        $deadline = (Get-Date).AddSeconds($HardSec)
        while ((Get-Date) -lt $deadline) {
            $p.Refresh()
            if ($p.HasExited) { break }
            Start-Sleep -Milliseconds 250
        }
    }
    $p.Refresh()
    if ($p.HasExited) {
        $rec.exited = $true
        $rec.exit_time = (Get-Date -Format "HH:mm:ss.fff")
        try {
            $ec = "$($p.ExitCode)"
            if ([string]::IsNullOrEmpty($ec)) { $ec = "n/a" }
            $rec.exit_code = $ec
        } catch { $rec.exit_code = "n/a" }
    }
    # Orphan / replacement check. The target itself can linger in the process table for
    # a moment after exit, and startup cleanup kills several instances deliberately --
    # neither is an orphan. Only an UNEXPECTED survivor counts.
    $left = @(Get-ShippingProcesses | Where-Object {
        $_.Id -ne $TargetPid -and ($ExpectedOthers -notcontains $_.Id) })
    if ($left.Count -gt 0) { $rec.orphan = (($left | ForEach-Object { $_.Id }) -join "|") }
    $rec.ports_released = (Wait-PortsReleased -TimeoutSec 20)
    $rec.ports_after = (Format-PortOwners (Get-PortOwners))
    return [pscustomobject]$rec
}

function Write-Lifecycle {
    param($Rec, [string] $Tag)
    Write-Host "  [lifecycle $Tag] launcher=$($script:LauncherPid) shipping=$($Rec.shipping_pid) " -NoNewline
    Write-Host "stop_req=$($Rec.stop_requested) exit=$($Rec.exit_time) code=$($Rec.exit_code) " -NoNewline
    Write-Host "forced=$($Rec.forced) orphan=$(if ($Rec.orphan) { $Rec.orphan } else { 'none' })"
    Write-Host "  [lifecycle $Tag] ports_before=$($Rec.ports_before) -> ports_after=$($Rec.ports_after) released=$($Rec.ports_released)"
}

function Clear-AllShipping {
    # Batch startup only: the box may hold shipping processes from an earlier session (or
    # an aborted parallel experiment). Clear them before anything is launched.
    $procs = @(Get-ShippingProcesses)
    if ($procs.Count -eq 0) { Write-Host "  startup: no shipping process running"; return $true }
    Write-Host "  startup: $($procs.Count) pre-existing shipping process(es): $(($procs | ForEach-Object { $_.Id }) -join ', ')" -ForegroundColor Yellow
    $allIds = @($procs | ForEach-Object { $_.Id })
    foreach ($p in $procs) {
        if ($DryRun) { Write-Host "[dry] stop shipping PID $($p.Id)" -ForegroundColor DarkGray; continue }
        # the other pre-existing instances are being cleared too -- not orphans
        $others = @($allIds | Where-Object { $_ -ne $p.Id })
        $r = Stop-ShippingProcess -TargetPid $p.Id -ExpectedOthers $others
        Write-Lifecycle -Rec $r -Tag "startup"
    }
    if ($DryRun) { return $true }
    $deadline = (Get-Date).AddSeconds(20)
    while ((Get-Date) -lt $deadline) {
        if (@(Get-ShippingProcesses).Count -eq 0) { break }
        Start-Sleep -Milliseconds 250
    }
    if (@(Get-ShippingProcesses).Count -ne 0) {
        Write-Warning "  startup: shipping processes still present -- refusing to launch"
        return $false
    }
    if (-not (Wait-PortsReleased -TimeoutSec 20)) {
        Write-Warning "  startup: UDP $($SimPorts -join '/') still owned: $(Format-PortOwners (Get-PortOwners))"
        return $false
    }
    Write-Host "  startup: clean -- no shipping process, ports released"
    return $true
}

function Start-SimTracked {
    # Launch via the official stub, then DISCOVER the shipping process by differencing the
    # PID set. Requires exactly one new shipping process; anything else is a lifecycle
    # failure rather than a guess.
    if ($DryRun) {
        Write-Host "[dry] Start-Process '$SimExe' (workdir $(Split-Path -Parent $SimExe))" -ForegroundColor DarkGray
        Write-Host "[dry] discover new $ShippingName PID (launcher PID is NOT the sim)" -ForegroundColor DarkGray
        $script:SimPid = -1; $script:SimStart = (Get-Date); $script:LauncherPid = -1
        return $true
    }
    if (-not (Test-Path -LiteralPath $SimExe)) { throw "simulator not found: $SimExe" }
    $before = @(Get-ShippingProcesses | ForEach-Object { $_.Id })
    $wd = Split-Path -Parent $SimExe
    Write-Host "  sim: launching stub $SimExe (workdir $wd)"
    $launcher = Start-Process -FilePath $SimExe -WorkingDirectory $wd -PassThru
    $script:LauncherPid = $launcher.Id
    Write-Host "  sim: launcher PID $($launcher.Id) -- NOT the simulator, never terminated"
    $deadline = (Get-Date).AddSeconds([Math]::Max($WarmupSeconds, 60))
    $new = @()
    while ((Get-Date) -lt $deadline) {
        $new = @(Get-ShippingProcesses | Where-Object { $before -notcontains $_.Id })
        if ($new.Count -ge 1) { break }
        Start-Sleep -Milliseconds 500
    }
    if ($new.Count -eq 0) {
        Write-Warning "  sim: no new $ShippingName process appeared within the launch window"
        return $false
    }
    if ($new.Count -gt 1) {
        Write-Warning "  sim: $($new.Count) new shipping processes appeared ($(($new | ForEach-Object { $_.Id }) -join ', ')) -- ambiguous, refusing"
        return $false
    }
    $script:SimPid = $new[0].Id
    try { $script:SimStart = $new[0].StartTime } catch { $script:SimStart = (Get-Date) }
    Write-Host "  sim: SHIPPING PID $($script:SimPid) started $($script:SimStart.ToString('HH:mm:ss.fff'))" -ForegroundColor Green
    Write-Host "  sim: warmup $WarmupSeconds s"
    Start-Sleep -Seconds $WarmupSeconds
    if ($ManualReady) { Read-Host "Press Enter when the course is loaded and ready" | Out-Null }
    return $true
}

function Test-PreFlightIsolation {
    # Refuse to fly unless exactly one simulator exists and it is OURS. Measured, two
    # instances are indistinguishable at both the socket and the protocol layer (same
    # source tuple 127.0.0.1:14560, same sys/comp id), so a merged stream cannot be
    # detected after the fact -- it has to be caught BEFORE the flight.
    if ($DryRun) { Write-Host "[dry] pre-flight isolation assertion" -ForegroundColor DarkGray; return $true }
    $procs = @(Get-ShippingProcesses)
    if ($procs.Count -ne 1) {
        Write-Warning "  ISOLATION: expected exactly 1 shipping process, found $($procs.Count)"
        return $false
    }
    if ($procs[0].Id -ne $script:SimPid) {
        Write-Warning "  ISOLATION: running shipping PID $($procs[0].Id) != stored $($script:SimPid)"
        return $false
    }
    $owners = @(Get-PortOwners | Where-Object { $_.Port -eq 14560 })
    if ($owners.Count -lt 1) {
        Write-Warning "  ISOLATION: nothing owns UDP 14560 -- the simulator is not up"
        return $false
    }
    $foreign = @($owners | Where-Object { $_.ProcId -ne $script:SimPid })
    if ($foreign.Count -gt 0) {
        Write-Warning "  ISOLATION: UDP 14560 also owned by $(Format-PortOwners $foreign)"
        return $false
    }
    Write-Host "  isolation OK: single shipping PID $($script:SimPid), sole owner of UDP 14560"
    return $true
}

function Invoke-RestartLifecycleSelfTest {
    # Two launch/terminate cycles, no flying. PASS only if each cycle yields a unique
    # shipping PID, the previous process is gone, no orphan remains, and the ports are
    # released between cycles.
    Write-Host ""
    Write-Host "=== RESTART LIFECYCLE SELF-TEST (2 cycles, no flight) ===" -ForegroundColor Cyan
    if (-not (Clear-AllShipping)) { Write-Host "SELF-TEST FAIL: could not reach a clean start" -ForegroundColor Red; return $false }
    $seen = @()
    $ok = $true
    for ($i = 1; $i -le 2; $i++) {
        Write-Host "--- cycle $i/2 ---" -ForegroundColor Cyan
        if ((-not $KeepSimAlive) -and -not (Start-SimTracked)) { Write-Host "SELF-TEST FAIL: launch $i produced no unique shipping PID" -ForegroundColor Red; return $false }
        $thisPid = $script:SimPid
        if ($seen -contains $thisPid) {
            Write-Host "SELF-TEST FAIL: cycle $i reused shipping PID $thisPid" -ForegroundColor Red
            $ok = $false
        }
        $seen += $thisPid
        $live = @(Get-ShippingProcesses)
        if (-not $DryRun -and $live.Count -ne 1) {
            Write-Host "SELF-TEST FAIL: cycle $i has $($live.Count) shipping processes" -ForegroundColor Red
            $ok = $false
        }
        if ((-not $KeepSimAlive) -and -not (Test-PreFlightIsolation)) { Write-Host "SELF-TEST FAIL: isolation assertion failed on cycle $i" -ForegroundColor Red; $ok = $false }
        if ($DryRun) { Write-Host "[dry] terminate stored shipping PID" -ForegroundColor DarkGray; continue }
        $rec = Stop-ShippingProcess -TargetPid $thisPid
        Write-Lifecycle -Rec $rec -Tag "cycle$i"
        if (-not $rec.exited)         { Write-Host "SELF-TEST FAIL: shipping PID $thisPid did not exit" -ForegroundColor Red; $ok = $false }
        if ($rec.orphan)              { Write-Host "SELF-TEST FAIL: orphan shipping process $($rec.orphan)" -ForegroundColor Red; $ok = $false }
        if (-not $rec.ports_released) { Write-Host "SELF-TEST FAIL: UDP $($SimPorts -join '/') not released" -ForegroundColor Red; $ok = $false }
    }
    Write-Host ""
    if ($ok) {
        Write-Host "SELF-TEST PASS  unique shipping PIDs: $($seen -join ', ')" -ForegroundColor Green
    } else {
        Write-Host "SELF-TEST FAIL" -ForegroundColor Red
    }
    return $ok
}

# ------------------------------------------------------------------------------------
# KEEP-SIM-ALIVE: reset the run with keystrokes instead of restarting the simulator
#
# WHY. PGOS login is manual and online, so a sim restart costs a human. In this mode the
# sim is launched, logged in and loaded ONCE by hand; every subsequent attempt is reset
# through the in-game pause menu: Esc (menu opens, RESUME focused) -> Down (RESTART) ->
# Enter. The "drone is no longer flying" signal is schedule_flier.py EXITING, which is
# already how every attempt ends.
#
# SendInput, not [System.Windows.Forms.SendKeys]: SendKeys posts WM_KEYDOWN messages,
# which Unreal's raw-input path ignores outright. SendInput injects at the driver level,
# which is what a fullscreen game actually reads. Scancodes are preferred over virtual
# keys for the same reason -- some UE builds map scancodes only.
# ------------------------------------------------------------------------------------
if (-not ([System.Management.Automation.PSTypeName]'VQ1Input').Type) {
    Add-Type -Namespace '' -Name 'VQ1Input' -MemberDefinition @'
    [StructLayout(LayoutKind.Sequential)]
    public struct KEYBDINPUT { public ushort wVk; public ushort wScan; public uint dwFlags; public uint time; public IntPtr dwExtraInfo; }
    [StructLayout(LayoutKind.Sequential)]
    public struct MOUSEINPUT { public int dx; public int dy; public uint mouseData; public uint dwFlags; public uint time; public IntPtr dwExtraInfo; }
    [StructLayout(LayoutKind.Sequential)]
    public struct HARDWAREINPUT { public uint uMsg; public ushort wParamL; public ushort wParamH; }
    [StructLayout(LayoutKind.Explicit)]
    public struct InputUnion {
        [FieldOffset(0)] public MOUSEINPUT mi;
        [FieldOffset(0)] public KEYBDINPUT ki;
        [FieldOffset(0)] public HARDWAREINPUT hi;
    }
    // MOUSEINPUT is the largest member, so an explicit union at offset 0 gets the size
    // and padding right on both x86 and x64 without hard-coding an offset.
    [StructLayout(LayoutKind.Sequential)]
    public struct INPUT { public uint type; public InputUnion U; }

    [DllImport("user32.dll", SetLastError=true)]
    public static extern uint SendInput(uint nInputs, INPUT[] pInputs, int cbSize);
    [DllImport("user32.dll")] public static extern uint MapVirtualKey(uint uCode, uint uMapType);
    [DllImport("user32.dll")] public static extern bool SetForegroundWindow(IntPtr hWnd);
    [DllImport("user32.dll")] public static extern IntPtr GetForegroundWindow();
    [DllImport("user32.dll")] public static extern uint GetWindowThreadProcessId(IntPtr hWnd, IntPtr pid);
    [DllImport("user32.dll")] public static extern bool AttachThreadInput(uint idAttach, uint idAttachTo, bool fAttach);
    [DllImport("user32.dll")] public static extern bool ShowWindow(IntPtr hWnd, int nCmdShow);
    [DllImport("kernel32.dll")] public static extern uint GetCurrentThreadId();
    [DllImport("kernel32.dll")] public static extern uint SetThreadExecutionState(uint esFlags);

    public static uint SendVirtualKey(ushort vk, bool keyUp, bool extended) {
        uint KEYEVENTF_EXTENDEDKEY = 0x0001, KEYEVENTF_KEYUP = 0x0002, KEYEVENTF_SCANCODE = 0x0008;
        ushort scan = (ushort)MapVirtualKey(vk, 0);          // MAPVK_VK_TO_VSC
        uint flags = 0;
        if (scan != 0) { flags |= KEYEVENTF_SCANCODE; }      // scancode when we have one
        if (extended)  { flags |= KEYEVENTF_EXTENDEDKEY; }
        if (keyUp)     { flags |= KEYEVENTF_KEYUP; }
        INPUT[] inp = new INPUT[1];
        inp[0].type = 1;                                     // INPUT_KEYBOARD
        inp[0].U.ki.wVk   = (scan != 0) ? (ushort)0 : vk;    // VK fallback if no scancode
        inp[0].U.ki.wScan = scan;
        inp[0].U.ki.dwFlags = flags;
        inp[0].U.ki.time = 0;
        inp[0].U.ki.dwExtraInfo = IntPtr.Zero;
        return SendInput(1, inp, Marshal.SizeOf(typeof(INPUT)));
    }
'@ | Out-Null
    # NOTE: no -UsingNamespace here. Add-Type -MemberDefinition already emits
    # `using System.Runtime.InteropServices;`, and adding it again is a duplicate-using
    # warning, which this compiler treats as an error.
}

$VK_ESCAPE = 0x1B
$VK_DOWN   = 0x28
$VK_RETURN = 0x0D

function Send-Key {
    # The ONLY place raw input is injected. Kept trivial and side-effect-free apart from
    # the call itself so the tests can replace it wholesale and assert order.
    param([int] $Vk, [string] $Name, [switch] $Extended)
    if ($DryRun) { Write-Host "[dry] key $Name" -ForegroundColor DarkGray; return $true }
    $d = [VQ1Input]::SendVirtualKey([uint16]$Vk, $false, [bool]$Extended)
    Start-Sleep -Milliseconds 40
    $u = [VQ1Input]::SendVirtualKey([uint16]$Vk, $true,  [bool]$Extended)
    if ($d -lt 1 -or $u -lt 1) {
        Write-Warning "  key $Name : SendInput accepted $d down / $u up event(s)"
        return $false
    }
    return $true
}

function Set-SimForeground {
    # A fullscreen game usually already holds focus, so a failure here is logged and
    # tolerated rather than fatal -- the keys are very likely to land anyway.
    if ($DryRun) { Write-Host "[dry] foreground sim window" -ForegroundColor DarkGray; return $true }
    $p = $null
    try { $p = [System.Diagnostics.Process]::GetProcessById($script:SimPid) } catch { $p = $null }
    if ($null -eq $p) { Write-Warning "  foreground: PID $($script:SimPid) not found"; return $false }
    $h = $p.MainWindowHandle
    if ($h -eq [IntPtr]::Zero) {
        Write-Host "  foreground: no MainWindowHandle (fullscreen) -- assuming focus"
        return $true
    }
    if ([VQ1Input]::SetForegroundWindow($h)) { return $true }
    # Windows refuses foreground changes from a background thread; attaching to the
    # target's input queue is the documented way around it.
    $fg  = [VQ1Input]::GetForegroundWindow()
    $tid = [VQ1Input]::GetWindowThreadProcessId($fg, [IntPtr]::Zero)
    $me  = [VQ1Input]::GetCurrentThreadId()
    [VQ1Input]::AttachThreadInput($me, $tid, $true) | Out-Null
    $ok = [VQ1Input]::SetForegroundWindow($h)
    [VQ1Input]::AttachThreadInput($me, $tid, $false) | Out-Null
    if (-not $ok) { Write-Host "  foreground: SetForegroundWindow refused -- assuming focus" }
    return $true
}

# THE RESET IS SPLIT IN TWO, and the split is the whole fix.
#
# schedule_flier.py connects, ARMs, prints "[sched] ARM sent; waiting for GO ...", and then
# BLOCKS until the sim reports race_start. RESTART is what fires the countdown. So pressing
# RESTART before the flier exists means GO passes with nobody armed, and the run never
# starts -- the drone just sits there until the batch times out.
#
# Correct order: open the menu, launch the flier, wait until it is armed and waiting, and
# only THEN confirm RESTART. The countdown now happens with a listener attached.
function Send-ResetOpenMenu {
    # Step 1: Esc. Leaves the sim paused with RESUME focused, holding the countdown.
    Write-Host "  reset 1/2: Esc (open pause menu)" -ForegroundColor Cyan
    Set-SimForeground | Out-Null
    Send-Key -Vk $VK_ESCAPE -Name "Esc" | Out-Null
    Start-Sleep -Milliseconds $KeyMenuOpenMs
    return $true
}

function Send-ResetConfirm {
    # Step 2: Down -> RESTART, Enter -> activate. Only ever called with an armed flier
    # already waiting for GO.
    Write-Host "  reset 2/2: Down -> Enter (RESTART)" -ForegroundColor Cyan
    Send-Key -Vk $VK_DOWN -Name "Down" -Extended | Out-Null
    Start-Sleep -Milliseconds $KeyStepMs
    Send-Key -Vk $VK_RETURN -Name "Enter" | Out-Null
    return $true
}

function Send-ResetSequence {
    # Retained composite (menu + confirm, no trailing settle) for manual use. The batch
    # loop deliberately does NOT call this -- it needs the flier launched between the two
    # halves.
    Send-ResetOpenMenu | Out-Null
    Send-ResetConfirm | Out-Null
    return $true
}

function Start-FlierAsync {
    # Launch the flier WITHOUT blocking, so the batch can watch for its arm signal and
    # press RESTART while it waits. stdout/stderr go to files; stderr is folded into the
    # run log at the end so the artifact matches the synchronous path's.
    param([string[]] $FlierArgs, [string] $LogPath)
    $errPath = "$LogPath.err"
    # -u FIRST: the flier prints "waiting for GO" without flush=True, and Python block-
    # buffers stdout when it is redirected to a file. Without -u that line would sit in the
    # buffer well past the arm timeout and every attempt would silently take the
    # proceed-anyway path -- i.e. the bug would look fixed while still racing the countdown.
    # -u affects buffering only; no flight code or parameter changes.
    $argv = @("-u") + $FlierArgs
    # QUOTE args containing whitespace: Start-Process -ArgumentList joins the array with
    # spaces but does NOT quote elements (PowerShell 5.1), so the tick-log path
    # 'C:\...\AI GP\AI-GrandPrix-\filt_auto_NNN.csv' was split at the space in 'AI GP'
    # and the flier died instantly on 'unrecognized arguments' -- before connecting or
    # arming. Every run since Start-FlierAsync was introduced failed this way. (The
    # synchronous path was immune: & $PythonExe @fargs passes the array natively.)
    $argv = @($argv | ForEach-Object { if ($_ -match '\s') { '"' + $_ + '"' } else { $_ } })
    $proc = Start-Process -FilePath $PythonExe -ArgumentList $argv `
                          -WorkingDirectory $RepoPath -NoNewWindow -PassThru `
                          -RedirectStandardOutput $LogPath -RedirectStandardError $errPath
    return [pscustomobject]@{ Proc = $proc; Log = $LogPath; Err = $errPath }
}

function Complete-FlierAsync {
    # Block until the flight ends, then fold stderr into the run log.
    param($Handle)
    $Handle.Proc.WaitForExit()
    $code = $Handle.Proc.ExitCode
    if (Test-Path -LiteralPath $Handle.Err) {
        $e = ""
        try { $e = [System.IO.File]::ReadAllText($Handle.Err) } catch { $e = "" }
        if ($e) { Add-Content -LiteralPath $Handle.Log -Value $e }
        Remove-Item -LiteralPath $Handle.Err -Force -ErrorAction SilentlyContinue
    }
    return $code
}

function Wait-ManualReset {
    # The whole of -ManualReset. No foreground, no SendInput, no timing: the human resets
    # the sim and presses Enter, which is what makes this mode immune to every failure the
    # keystroke path can have (menu layout, focus, arm race, countdown timing).
    Write-Host ""
    if ($DryRun) { Write-Host "[dry] Read-Host manual reset prompt" -ForegroundColor DarkGray; return }
    Read-Host "Reset the sim (Esc, Restart), then press Enter for the next run" | Out-Null
}

function Stop-StrayFliers {
    <#
      Kill leftover schedule_flier.py processes from a previous or killed batch.

      THE SILENT KILLER. A stray flier still holds the UDP link, so a NEW flier dies almost
      on contact. An instant-exit flier makes the loop race ahead to the next attempt and
      press Esc into a live countdown -- which is what "Esc right away" actually was. Key
      timing was never the cause.

      Matched on the COMMAND LINE, not the process name: the interpreter may be python.exe,
      a venv shim, or a full path, and an unrelated python must never be killed.
    #>
    $stray = @()
    try {
        $stray = @(Get-CimInstance Win32_Process -ErrorAction Stop | Where-Object {
            $_.CommandLine -and $_.CommandLine -like '*schedule_flier.py*' -and
            $_.ProcessId -ne $PID })
    } catch { Write-Warning "  stray scan failed: $($_.Exception.Message)"; return 0 }
    if ($stray.Count -eq 0) { Write-Host "  stray fliers: none"; return 0 }
    foreach ($st in $stray) {
        Write-Host "  stray flier PID $($st.ProcessId) -- terminating (it holds the UDP link)" -ForegroundColor Yellow
        if ($DryRun) { continue }
        try { Stop-Process -Id $st.ProcessId -Force -ErrorAction Stop } catch {
            Write-Warning "  could not stop stray $($st.ProcessId): $($_.Exception.Message)"
        }
    }
    if (-not $DryRun) { Start-Sleep -Milliseconds 500 }
    return $stray.Count
}

function Test-SimAlive {
    # The stored shipping process must still exist AND still own 14560. A foreign owner
    # means something else answered on the control channel -- never fly into that.
    if ($DryRun) { return $true }
    $p = $null
    try { $p = Get-Process -Id $script:SimPid -ErrorAction Stop } catch { $p = $null }
    if ($null -eq $p) { Write-Warning "  SIM CHECK: shipping PID $($script:SimPid) is gone"; return $false }
    $owners = @(Get-PortOwners | Where-Object { $_.Port -eq 14560 })
    if ($owners.Count -lt 1) { Write-Warning "  SIM CHECK: nothing owns UDP 14560"; return $false }
    $foreign = @($owners | Where-Object { $_.ProcId -ne $script:SimPid })
    if ($foreign.Count -gt 0) { Write-Warning "  SIM CHECK: foreign owner on 14560: $(Format-PortOwners $foreign)"; return $false }
    return $true
}

function Invoke-SimLostAbort {
    param([string] $Where)
    Write-Host ""
    Write-Host "########################################################" -ForegroundColor Red
    Write-Host "###  SIM LOST -- manual re-login required            ###" -ForegroundColor Red
    Write-Host "###  ($Where)" -ForegroundColor Red
    Write-Host "###  NOT relaunching: PGOS login is manual and online ###" -ForegroundColor Red
    Write-Host "########################################################" -ForegroundColor Red
    Write-Host ""
    if ($env:VQ1_NOTIFY_CMD) {
        try { & cmd.exe /c $env:VQ1_NOTIFY_CMD "VQ1: SIM LOST - manual re-login required" | Out-Null }
        catch { Write-Warning "  notify hook failed: $($_.Exception.Message)" }
    }
}

function Set-KeepAwake {
    # Display sleep or a lock screen destroys injected input mid-batch, so the run holds
    # the system and display awake for its duration and releases in the finally block.
    param([switch] $Release)
    if ($DryRun) { Write-Host "[dry] SetThreadExecutionState $(if ($Release) { 'release' } else { 'hold' })" -ForegroundColor DarkGray; return }
    $ES_CONTINUOUS = [uint32]2147483648   # 0x80000000
    $ES_SYSTEM     = [uint32]1
    $ES_DISPLAY    = [uint32]2
    if ($Release) { [VQ1Input]::SetThreadExecutionState($ES_CONTINUOUS) | Out-Null }
    else { [VQ1Input]::SetThreadExecutionState($ES_CONTINUOUS -bor $ES_SYSTEM -bor $ES_DISPLAY) | Out-Null }
}

function Initialize-KeepSimAlive {
    # Adopt the sim the user already launched and logged in. Never launches anything.
    Write-Host "=== KEEP-SIM-ALIVE: adopting the running simulator ===" -ForegroundColor Cyan
    $procs = @(Get-ShippingProcesses)
    if ($procs.Count -ne 1) {
        Write-Host ""
        Write-Host "Found $($procs.Count) $ShippingName process(es); need exactly 1." -ForegroundColor Red
        Write-Host "Launch the sim, log in, load the qualifier, put the drone at the start," -ForegroundColor Yellow
        Write-Host "THEN run with -KeepSimAlive." -ForegroundColor Yellow
        return $false
    }
    $script:SimPid = $procs[0].Id
    try { $script:SimStart = $procs[0].StartTime } catch { $script:SimStart = (Get-Date) }
    Write-Host "  adopted SHIPPING PID $($script:SimPid)" -ForegroundColor Green
    if (-not (Test-PreFlightIsolation)) {
        Write-Host ""
        Write-Host "Isolation assertion failed." -ForegroundColor Red
        Write-Host "Launch the sim, log in, load the qualifier, put the drone at the start," -ForegroundColor Yellow
        Write-Host "THEN run with -KeepSimAlive." -ForegroundColor Yellow
        return $false
    }
    Read-Host "Logged in, qualifier loaded, drone at the start gate? Press Enter to begin" | Out-Null
    return $true
}

# ------------------------------------------------------------------------------------
# The frozen flight command
# ------------------------------------------------------------------------------------
function Get-FlightArgs {
    param([string] $CsvPath)
    # FROZEN. Derivative-aware commit + Gate-3 descent; the hold is the only variable.
    $a = @(
        "tools/schedule_flier.py", "--coast-tube", "--const-thrust", "0.275",
        "--descent-bias-leg0", "-0.005", "--descent-bias-leg1", "0.039",
        "--descent-bias-leg2", "0.024",
        "--gate-vert", "--gate-vert-size-min", "0.05",
        "--vert-auth-down", "0.045,0.015", "--vert-auth-up", "0.06,0.10",
        "--k-thrust-v", "0.16", "--kd-v", "0.1", "--k-gate-bank", "1.0",
        "--post-gate-hold-s", "1.5", "--post-gate-hold-decay", "0.5",
        "--thrust-slew", "0.6", "--thrust-slew-down", "0.6",
        "--gate-max-bank-deg", "11", "--gate-max-bank-ag2", "9",
        "--gate-max-bank-ag2-left", "0",
        "--post-gate1-bank", "5", "--post-gate2-bank", "-4.5",
        "--gate2-hold-fixed-deg", "-0.6",
        "--gate-bank-size-min-ag2", "0.05", "--gate-bank-full-size-ag2", "0.15",
        "--gate2-exit-level-size", "0.35", "--gate2-exit-level-tau", "0.10",
        "--gate-commit-size", "0.16", "--gate-commit-align", "0.08",
        "--gate-commit-stable-frames", "3",
        "--gate-commit-rate-max", "0.13", "--gate-commit-stable-s", "0.15",
        "--gate-vert-commit-size", "0.18", "--gate-vert-commit-to-ag", "0",
        "--leg2-entry-arrest", "0.12", "--leg2-entry-arrest-s", "1.6"
    )
    if ($Mode -eq "descent" -or $Mode -eq "holdblind") {
        $a += @("--gate3-vert-descent", "--gate3-vert-descent-delta", "0.015",
                "--gate3-vert-descent-size", "$DESCENT_SIZE")
    }
    if ($Mode -eq "holdblind") {
        $a += @("--gate3-vert-descent-hold-s", "$HoldSeconds")
    }
    # --log-tube AS RATE BALLAST (-RateBallast, default ON).
    #
    # It runs the rail detector every frame: ~3.4 ms/frame of deterministic, flight-inert
    # load. Inert is the load-bearing word -- its 22 columns are report-invisible, and
    # zeroing all 22 in a real log leaves flight_report.py's output and JSON
    # byte-identical, so this buys loop time without touching a single command term.
    #
    # WHY BALLAST IS WANTED: with the process-lifecycle fix removing merged-stream
    # variance, the loop now runs ~88 Hz on Balanced power. The frozen tune was validated
    # at ~68 Hz. The ballast pulls 88 Hz back to ~68 Hz, i.e. back to the rate the tune
    # was actually measured at.
    #
    # Clear it with -RateBallast:$false to fly at the higher rate.
    if ($RateBallast) { $a += @("--log-tube") }
    $a += @("--no-pitch-hold", "--tick-log", $CsvPath)
    return $a
}

# ------------------------------------------------------------------------------------
# Classification
# ------------------------------------------------------------------------------------
function Get-Classification {
    <#
      Order is load-bearing.

      FULL_COURSE first: the ONLY success. It is the sim's own race-finish signal, not a
      high active_gate -- Gate 3 is the third of six gates and registering it proves
      nothing about the rest of the course.

      POST_G3_FAILURE second, ABOVE the rate/dirty discards: a run that passed Gate 3 and
      then died is the most informative artefact we can produce right now, and filing it
      as a plain discard would bury exactly the evidence we are hunting for.
    #>
    param($R)      # parsed JSON object, or $null when the report could not be produced
    if ($null -eq $R) { return "BROKEN" }
    if ($R.full_course_complete) { return "FULL_COURSE" }
    if ($R.gate3_registered) { return "POST_G3_FAILURE" }
    if ($R.dirty_start) { return "DISCARD_DIRTY" }
    if ($R.verdict -ne "OK") { return "DISCARD_RATE" }
    if ($R.max_active_gate -eq 2) {
        $sz = $R.g3_size
        if ($null -ne $sz -and $sz -ge $DEEP_SIZE) { return "KEEP_DEEP" }
        return "KEEP_SHALLOW"
    }
    return "KEEP_UPSTREAM"
}

function Test-CleanDeepHold {
    param($R, [string] $Class)
    if ($null -eq $R) { return $false }
    return ($R.verdict -eq "OK" -and $R.max_active_gate -eq 2 -and
            $null -ne $R.g3_size -and $R.g3_size -ge $DEEP_SIZE -and
            $R.descent_frames -gt 0 -and $R.hold_frames -gt 0)
}

function Get-DestDir {
    param([string] $Class)
    switch ($Class) {
        "FULL_COURSE"     { return $QualDir }
        "POST_G3_FAILURE" { return $KeepDir }
        "KEEP_DEEP"     { return $KeepDir }
        "KEEP_SHALLOW"  { return $DiagDir }
        "KEEP_UPSTREAM" { return $DiagDir }
        "BROKEN_RESTART"  { return $TrashDir }
        default         { return $TrashDir }   # discards + BROKEN: moved, never deleted
    }
}

function Test-PurgeableDiscard {
    <#
      A run is purgeable ONLY when it is worthless: a pure discard that never even reached
      the Gate-3 leg. Everything with max_active_gate >= 2 is evidence about the terminal
      and is kept, as are POST_G3_FAILURE and FULL_COURSE regardless of depth.

      This is the one place the runner deletes rather than moves, and it exists because a
      1000-attempt overnight batch otherwise fills the disk with runs that died before the
      interesting part. Deletion is irreversible, so every uncertainty resolves to KEEP:
      wrong mode, KeepAllRuns, a missing report, or a report without the depth field all
      fall through to the normal move.
    #>
    param($R, [string] $Class)
    if ($KeepAllRuns)   { return $false }
    if (-not $KeepSimAlive) { return $false }
    if (@("DISCARD_RATE", "DISCARD_DIRTY", "BROKEN") -notcontains $Class) { return $false }
    # No parsed report => depth unknown => cannot prove it is worthless => keep it.
    if ($null -eq $R) { return $false }
    if (-not ($R.PSObject.Properties.Name -contains "max_active_gate")) { return $false }
    if ($null -eq $R.max_active_gate) { return $false }
    if ($R.max_active_gate -ge 2) { return $false }
    return $true
}

function Remove-DiscardPair {
    # Returns the MB freed so the batch can report a running total.
    param([string] $Csv, [string] $Log, [string] $Stem)
    $bytes = 0
    foreach ($f in @($Csv, $Log)) {
        if (Test-Path -LiteralPath $f) { $bytes += (Get-Item -LiteralPath $f).Length }
    }
    $mb = [math]::Round($bytes / 1MB, 2)
    if ($DryRun) { Write-Host "[dry] purged discard $Stem, freed $mb MB" -ForegroundColor DarkGray; return $mb }
    foreach ($f in @($Csv, $Log)) {
        if (Test-Path -LiteralPath $f) { Remove-Item -LiteralPath $f -Force -ErrorAction SilentlyContinue }
    }
    Write-Host "  purged discard $Stem, freed $mb MB" -ForegroundColor DarkGray
    return $mb
}

function Move-Pair {
    param([string] $Csv, [string] $Log, [string] $Dest)
    foreach ($f in @($Csv, $Log)) {
        if (-not (Test-Path -LiteralPath $f)) { continue }
        $target = Join-Path $Dest (Split-Path -Leaf $f)
        if ($DryRun) { Write-Host "[dry] move '$f' -> '$target'" -ForegroundColor DarkGray; continue }
        Move-Item -LiteralPath $f -Destination $target -Force
    }
}

# ------------------------------------------------------------------------------------
# Summary CSV
# ------------------------------------------------------------------------------------
$SUMMARY_COLS = @(
    "timestamp", "attempt", "run", "mode", "hold_s", "csv", "class", "verdict",
    "median_rate", "slow_pct", "dirty_start", "crossing_dips", "gate1_u",
    "max_active_gate", "g3_size", "g3_u_f", "g3_v_err", "commit_latched",
    "commit_stable_s", "descent_frames", "descent_impulse", "hold_armed",
    "hold_frames", "hold_duration_s", "registered", "flicker_blocked",
    "gate3_registered", "highest_active_gate", "gates_completed",
    "full_course_complete", "finish_signal", "first_failure_after_gate3",
    "time_gate3_registered", "final_segment", "collision_after_gate3",
    "timeout_after_gate3", "failure_reason", "exit_code"
)

function Write-SummaryRow {
    param($R, [string] $Class, [int] $Attempt, [int] $Run, [string] $Csv, [int] $ExitCode)
    if (-not (Test-Path -LiteralPath $SummaryCsv)) {
        if ($DryRun) { Write-Host "[dry] create $SummaryCsv" -ForegroundColor DarkGray }
        else { ($SUMMARY_COLS -join ",") | Out-File -FilePath $SummaryCsv -Encoding utf8 }
    }
    $g = {
        param($o, $n)
        if ($null -eq $o) { return "" }
        if (-not ($o.PSObject.Properties.Name -contains $n)) { return "" }
        $v = $o.$n
        if ($null -eq $v) { return "" }
        if ($v -is [bool]) { if ($v) { return "1" } else { return "0" } }
        if ($v -is [array]) { return ($v -join "|") }
        return "$v"
    }
    $fb = ""
    if ($null -ne $R -and $R.PSObject.Properties.Name -contains "flicker") {
        if ($R.flicker.blocked) { $fb = "1" } else { $fb = "0" }
    }
    $vals = @(
        (Get-Date -Format "yyyy-MM-ddTHH:mm:ss"), "$Attempt", "$Run", $Mode, "$HoldSeconds",
        (Split-Path -Leaf $Csv), $Class,
        (& $g $R "verdict"), (& $g $R "median_rate"), (& $g $R "slow_pct"),
        (& $g $R "dirty_start"), (& $g $R "crossing_dips"), (& $g $R "gate1_u"),
        (& $g $R "max_active_gate"), (& $g $R "g3_size"), (& $g $R "g3_u_f"),
        (& $g $R "g3_v_err"), (& $g $R "commit_latched"), (& $g $R "commit_stable_s"),
        (& $g $R "descent_frames"), (& $g $R "descent_impulse"), (& $g $R "hold_armed"),
        (& $g $R "hold_frames"), (& $g $R "hold_duration_s"), (& $g $R "registered"),
        $fb,
        (& $g $R "gate3_registered"), (& $g $R "highest_active_gate"),
        (& $g $R "gates_completed"), (& $g $R "full_course_complete"),
        (& $g $R "finish_signal"), (& $g $R "first_failure_after_gate3"),
        (& $g $R "time_gate3_registered"), (& $g $R "final_segment"),
        (& $g $R "collision_after_gate3"), (& $g $R "timeout_after_gate3"),
        (& $g $R "failure_reason"), "$ExitCode"
    )
    $quoted = $vals | ForEach-Object { '"' + ($_ -replace '"', '""') + '"' }
    $line = $quoted -join ","
    if ($DryRun) { Write-Host "[dry] summary += $line" -ForegroundColor DarkGray }
    else { $line | Out-File -FilePath $SummaryCsv -Append -Encoding utf8 }
}

# ------------------------------------------------------------------------------------
# Resume: never overwrite an existing run number, in any of the result dirs
# ------------------------------------------------------------------------------------
function Get-NextRun {
    param([int] $From)
    $n = $From
    while ($true) {
        $stem = "filt_auto_{0:d3}" -f $n
        $hit = $false
        foreach ($d in @($RepoPath, $KeepDir, $DiagDir, $TrashDir)) {
            if (Test-Path -LiteralPath (Join-Path $d "$stem.csv")) { $hit = $true; break }
            if (Test-Path -LiteralPath (Join-Path $d "$stem.log")) { $hit = $true; break }
        }
        if (-not $hit) { return $n }
        $n++
    }
}

# ====================================================================================
# MAIN
# ====================================================================================
$attempt = 0
$cleanDeep = 0
$qualified = $false
$run = $StartRun
$results = @()
# Course-progress tally. active_gate is the index of the gate being flown TOWARD, so it
# doubles as the count of gates already passed (6 gates, 0=START .. 5=FINISH).
$cnt = @{ clean = 0; reachedG3leg = 0; g3reg = 0; reachedG4 = 0; reachedFinal = 0;
          full = 0; postG3fail = 0 }
$purged = 0
$freedMb = 0.0
$failPoints = @{}

$brokenRestart = $false
if ($TestRestartLifecycle) {
    # Standalone: no batch state, no summary, no finally-tally -- just the lifecycle proof.
    $pass = Invoke-RestartLifecycleSelfTest
    if ($pass) { exit 0 } else { exit 1 }
}
$simLost = $false
try {
    Initialize-Dirs
    if ($KeepSimAlive) {
        # Adopt the already-running, already-logged-in simulator. Nothing is launched or
        # terminated in this mode.
        # BEFORE anything else: strays from a previously killed batch hold the UDP link and
        # make every new flier die on contact.
        Stop-StrayFliers | Out-Null
        if (-not (Initialize-KeepSimAlive)) { throw "keep-sim-alive preconditions not met" }
        Set-KeepAwake
    } else {
        if (-not (Clear-AllShipping)) { throw "startup cleanup failed -- refusing to start the batch" }
    }
    $run = Get-NextRun -From $StartRun
    Write-Host ""
    Write-Host "=== VQ1 QUALIFIER BATCH ===" -ForegroundColor Cyan
    Write-Host "  mode=$Mode  hold=$HoldSeconds s  need $CleanDeepRequired clean-deep-hold  max $MaxAttempts attempts"
    if ($KeepSimAlive -and $ManualReset) {
        Write-Host "  MANUAL RESET: no keystrokes are sent; you reset between runs" -ForegroundColor Yellow
    }
    Write-Host "  first run number: $run    output: $Root"
    if ($DryRun) { Write-Host "  *** DRY RUN -- no process is started or stopped ***" -ForegroundColor Yellow }
    Write-Host ""

    while ($attempt -lt $MaxAttempts) {
        $attempt++
        $stem = "filt_auto_{0:d3}" -f $run
        $csv = Join-Path $RepoPath "$stem.csv"
        $log = Join-Path $RepoPath "$stem.log"
        $jsn = Join-Path $JsonDir "$stem.json"

        Write-Host "--- attempt $attempt/$MaxAttempts   run $stem ---" -ForegroundColor Cyan

        # 1a. KEEP-SIM-ALIVE: no restart. Confirm the adopted sim is still the thing on the
        #     control channel, then OPEN the pause menu -- the RESTART confirm comes later,
        #     after the flier is armed.
        #
        #     ATTEMPT 1 IS DIFFERENT, and it has to be. The reset is a BETWEEN-RUNS setup
        #     for the next flight, not a pre-flight action: Esc/RESTART exist to recover a
        #     drone that has already crashed or idled after the previous flier exited. On
        #     attempt 1 there is no previous run -- the user has just loaded the qualifier
        #     and the drone is entering its own post-load GO countdown. Pressing Esc there
        #     interrupts that launch. So attempt 1 only arms and rides the initial GO.
        if ($KeepSimAlive) {
            if (-not (Test-SimAlive)) {
                Invoke-SimLostAbort -Where "pre-flight check, attempt $attempt"
                $simLost = $true
                break
            }
            if ($ManualReset) {
                Write-Host "  manual reset mode: no keystrokes will be sent" -ForegroundColor Cyan
            }
        }
        # 1b. full restart path -- UNCHANGED. Terminates ONLY the stored shipping PID.
        if ((-not $KeepSimAlive) -and $null -ne $script:SimPid) {
            $rec = Stop-ShippingProcess -TargetPid $script:SimPid
            Write-Lifecycle -Rec $rec -Tag "restart"
            if ((-not $rec.exited) -or $rec.orphan -or (-not $rec.ports_released)) {
                Write-Host "  BROKEN_RESTART: cleanup failed -- refusing to launch an overlapping instance" -ForegroundColor Red
                Write-SummaryRow -R $null -Class "BROKEN_RESTART" -Attempt $attempt -Run $run -Csv $csv -ExitCode -1
                $results += [pscustomobject]@{ run = $stem; class = "BROKEN_RESTART" }
                $brokenRestart = $true
                break
            }
        }
        if ((-not $KeepSimAlive) -and -not (Start-SimTracked)) {
            Write-Host "  BROKEN_RESTART: could not establish exactly one shipping process" -ForegroundColor Red
            Write-SummaryRow -R $null -Class "BROKEN_RESTART" -Attempt $attempt -Run $run -Csv $csv -ExitCode -1
            $results += [pscustomobject]@{ run = $stem; class = "BROKEN_RESTART" }
            $brokenRestart = $true
            break
        }
        if ((-not $KeepSimAlive) -and -not (Test-PreFlightIsolation)) {
            Write-Host "  BROKEN_RESTART: pre-flight isolation assertion failed -- NOT flying" -ForegroundColor Red
            Write-SummaryRow -R $null -Class "BROKEN_RESTART" -Attempt $attempt -Run $run -Csv $csv -ExitCode -1
            $results += [pscustomobject]@{ run = $stem; class = "BROKEN_RESTART" }
            $brokenRestart = $true
            break
        }

        # 2. exactly one flight
        $fargs = Get-FlightArgs -CsvPath $csv
        $shown = ($fargs | ForEach-Object { if ($_ -match '[\s"]') { '"' + $_ + '"' } else { $_ } }) -join " "
        Write-Host "  CMD: $PythonExe $shown"
        $exit = 0
        if ($DryRun) {
            Write-Host "[dry] (flight skipped)" -ForegroundColor DarkGray
            if ($KeepSimAlive -and -not $ManualReset) {
                Send-ResetOpenMenu | Out-Null
                Write-Host "[dry] launch flier; settle $RestartDelaySeconds s" -ForegroundColor DarkGray
                Send-ResetConfirm | Out-Null
            }
        } elseif ($KeepSimAlive) {
            # THE UNIFORM CYCLE. Every attempt is identical -- no attempt-1 branch, no
            # arm-signal gate, nothing conditional:
            #   1. Esc            previous run is over (its flier exited), so the drone is
            #                     idle/crashed; on attempt 1 the loaded drone pauses the same
            #   2. launch flier   connects, arms, waits for GO
            #   3. settle         -RestartDelaySeconds
            #   4. Down + Enter   RESTART -> fresh countdown -> the armed flier catches GO
            #   5. await exit     the run is clearly done
            # Steps 6-7 (classify / purge / summary / loop) are below and unchanged.
            if (-not $ManualReset) { Send-ResetOpenMenu | Out-Null }
            $h = Start-FlierAsync -FlierArgs $fargs -LogPath $log
            if (-not $ManualReset) {
                Write-Host "  settle $RestartDelaySeconds s (flier connecting + arming)"
                Start-Sleep -Milliseconds ([int]($RestartDelaySeconds * 1000))
                Send-ResetConfirm | Out-Null
            }
            $exit = Complete-FlierAsync -Handle $h
            if ($exit -ne 0) { Write-Warning "  flier exited $exit" }
        } else {
            # FULL-RESTART PATH -- unchanged. The sim was just launched, so the countdown
            # has not run yet and a synchronous flier catches GO normally.
            Push-Location $RepoPath
            try {
                & $PythonExe @fargs 2>&1 | Tee-Object -FilePath $log
                # $? is unreliable here: in PS 5.1 a native command's stderr becomes
                # ErrorRecords in the pipeline and clears it even on success. The exe's
                # own code is the only trustworthy signal.
                $exit = $LASTEXITCODE
            } finally { Pop-Location }
            if ($exit -ne 0) { Write-Warning "  flier exited $exit" }
        }

        # 3. report
        $R = $null
        $cls = "BROKEN"
        if ($DryRun) {
            Write-Host "[dry] $PythonExe flight_report.py `"$csv`" --json `"$jsn`"" -ForegroundColor DarkGray
            $cls = "DRYRUN"
        } elseif (-not (Test-Path -LiteralPath $csv)) {
            Write-Warning "  no CSV produced -> BROKEN"
        } elseif ((Get-Item -LiteralPath $csv).Length -eq 0) {
            Write-Warning "  empty CSV -> BROKEN"
        } else {
            Push-Location $RepoPath
            try {
                & $PythonExe "analysis/flight_report.py" $csv "--json" $jsn "--descent-size" "$DESCENT_SIZE"
                $rexit = $LASTEXITCODE
            } finally { Pop-Location }
            if (Test-Path -LiteralPath $jsn) {
                try { $R = Get-Content -LiteralPath $jsn -Raw | ConvertFrom-Json } catch {
                    Write-Warning "  unparseable report: $($_.Exception.Message)"; $R = $null
                }
            } else {
                Write-Warning "  flight_report.py produced no JSON (exit $rexit)"
            }
            $cls = Get-Classification -R $R
        }

        Write-Host "  CLASS: $cls" -ForegroundColor Yellow
        if ($null -ne $R -and $R.PSObject.Properties.Name -contains "flicker") {
            if ($R.flicker.blocked) {
                Write-Host "  HOLD BLOCKED BY FLICKER (see report line)" -ForegroundColor Magenta
            }
        }

        Write-SummaryRow -R $R -Class $cls -Attempt $attempt -Run $run -Csv $csv -ExitCode $exit
        if (Test-PurgeableDiscard -R $R -Class $cls) {
            # Worthless attempt: never reached the Gate-3 leg AND failed on rate/dirty.
            # The summary row is already written above, so the attempt stays on the record
            # even though its artifacts are gone.
            $freedMb += Remove-DiscardPair -Csv $csv -Log $log -Stem $stem
            $purged++
            $run++
            # the purge path `continue`s past the end of the loop, so the prompt has to be
            # repeated here or a purged attempt would roll straight into the next flight
            if ($KeepSimAlive -and $ManualReset -and ($attempt -lt $MaxAttempts)) {
                Wait-ManualReset
            }
            continue
        }
        $dest = Get-DestDir -Class $cls
        Move-Pair -Csv $csv -Log $log -Dest $dest
        if ($cls -eq "FULL_COURSE") {
            # The deliverable: keep the report JSON and any dumped frames WITH the run.
            foreach ($extra in @($jsn)) {
                if (Test-Path -LiteralPath $extra) {
                    if ($DryRun) { Write-Host "[dry] copy '$extra' -> '$dest'" -ForegroundColor DarkGray }
                    else { Copy-Item -LiteralPath $extra -Destination $dest -Force }
                }
            }
            $fr = Join-Path $RepoPath "leg1_frames"
            if (Test-Path -LiteralPath $fr) {
                $frDest = Join-Path $dest "$stem`_frames"
                if ($DryRun) { Write-Host "[dry] copy frames -> '$frDest'" -ForegroundColor DarkGray }
                else { Copy-Item -LiteralPath $fr -Destination $frDest -Recurse -Force }
            }
        }
        $results += [pscustomobject]@{ run = $stem; class = $cls }

        # ---- counters ------------------------------------------------------------
        if ($null -ne $R) {
            if ($R.verdict -eq "OK") { $cnt.clean++ }
            if ($R.max_active_gate -ge 2) { $cnt.reachedG3leg++ }
            if ($R.gate3_registered)      { $cnt.g3reg++ }
            if ($R.highest_active_gate -ge 4) { $cnt.reachedG4++ }
            if ($R.highest_active_gate -ge 5) { $cnt.reachedFinal++ }
            if ($R.full_course_complete)  { $cnt.full++ }
            if ($cls -eq "POST_G3_FAILURE") {
                $cnt.postG3fail++
                $fp = "$($R.first_failure_after_gate3)"
                if ([string]::IsNullOrWhiteSpace($fp)) { $fp = "unknown" }
                # bucket by WHERE it died, so repeated failures aggregate usefully
                $key = "ag$($R.highest_active_gate): $fp"
                if ($failPoints.ContainsKey($key)) { $failPoints[$key]++ } else { $failPoints[$key] = 1 }
            }
        }

        # ---- milestones and the ONLY stop condition ------------------------------
        # Gate 3 is the third of six gates. Registering it is a milestone, and the flight
        # is already written to continue past it (the flier breaks on `fin > 0 or ag > 5`,
        # not on ag>=3), so the batch must not cut the run or the series short here.
        if ($cls -eq "POST_G3_FAILURE") {
            Write-Host "*** GATE 3 REGISTERED — CONTINUING COURSE ***" -ForegroundColor Green
            Write-Host "  ...but the course did not complete: $($R.first_failure_after_gate3)" -ForegroundColor Yellow
        }
        if ($cls -eq "FULL_COURSE") {
            $qualified = $true
            Write-Host "*** GATE 3 REGISTERED — CONTINUING COURSE ***" -ForegroundColor Green
            Write-Host "*** FULL COURSE QUALIFIED ***" -ForegroundColor Green
            Write-Host "  finish signal: $($R.finish_signal)" -ForegroundColor Green
            break
        }
        if (Test-CleanDeepHold -R $R -Class $cls) {
            $cleanDeep++
            Write-Host "  clean deep + hold active: $cleanDeep/$CleanDeepRequired" -ForegroundColor Green
        }

        # NOTE: in the automatic path there is no reset here -- it WRAPS the next attempt's
        # flier launch (Esc -> launch -> wait armed -> RESTART), because RESTART fires the
        # countdown and the flier has to be armed and listening before it does.
        # Under -ManualReset the reset is entirely yours, and the batch simply waits.
        if ($KeepSimAlive -and $ManualReset -and ($attempt -lt $MaxAttempts)) {
            Wait-ManualReset
        }
        $run++
    }
} finally {
    # Runs on Ctrl+C too, so a batch is never left without its tally.
    Write-Host ""
    Write-Host "=== BATCH END ===" -ForegroundColor Cyan
    Write-Host "  total attempts     : $attempt"
    Write-Host "  clean attempts     : $($cnt.clean)"
    Write-Host "  reached Gate-3 leg : $($cnt.reachedG3leg)"
    Write-Host "  Gate 3 registered  : $($cnt.g3reg)"
    Write-Host "  reached Gate 4     : $($cnt.reachedG4)"
    Write-Host "  reached final gate : $($cnt.reachedFinal)"
    Write-Host "  FULL COURSE        : $($cnt.full)"
    Write-Host "  post-G3 failures   : $($cnt.postG3fail)"
    if ($failPoints.Count -gt 0) {
        Write-Host "  post-G3 first failure point:"
        foreach ($k in ($failPoints.Keys | Sort-Object)) { Write-Host ("    {0,-3} {1}" -f $failPoints[$k], $k) }
    }
    Write-Host "  purged discards    : $purged  ($([math]::Round($freedMb,1)) MB freed)"
    Write-Host "  clean deep+hold    : $cleanDeep"
    Write-Host "  qualified          : $qualified"
    if ($brokenRestart) { Write-Host "  *** BATCH STOPPED: BROKEN_RESTART ***" -ForegroundColor Red }
    if ($simLost) { Write-Host "  *** BATCH STOPPED: SIM LOST -- manual re-login required ***" -ForegroundColor Red }
    if ($KeepSimAlive) { Set-KeepAwake -Release }
    if (@($results).Count -gt 0) { @($results) | Group-Object class | ForEach-Object { Write-Host ("  {0,-15} {1}" -f $_.Name, $_.Count) } }
    Write-Host "  summary          : $SummaryCsv"
    Write-Host "  next run number  : $run"
}
