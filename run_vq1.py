#!/usr/bin/env python3
"""
SCUBA LAB - VQ1 Competition Entry Point
========================================
Usage:
    python3 run_vq1.py [--host 127.0.0.1] [--port 14540] [--hz 50] [--allow-stub-vision]

This is the single canonical entry point for VQ1 submission.
It loads the distilled vision model, connects to the DCL simulator
over UDP MAVLink v2, and runs the control loop.

Model path: ./models_release/aigp_distill_final.zip
  (canonical location per obsidian/fragilities.md and obsidian/submission-readiness.md)

Control interface:
  SET_ATTITUDE_TARGET with type_mask=128 (IGNORE_ATTITUDE)
  body_roll/pitch/yaw_rate = policy_output[1:4] * MAX_BODY_RATE (12.0 rad/s)
  thrust = policy_output[0] in [0, 1]

Vision:
  BLOCKED until DCL simulator ships (May 2026). _get_vision_frame() raises
  NotImplementedError by default so this script cannot accidentally run
  against zero-filled frames and emit plausible-looking but garbage MAVLink
  control commands.

  For local testing/development only, pass --allow-stub-vision to permit
  the black-frame stub path. Submission must NOT use this flag.

  When the DCL simulator ships, replace _get_vision_frame() with the real
  image-API call; the guard then falls out automatically.

What to ask DCL when the simulator ships (from obsidian/submission-readiness.md):
  1. Exact MAVLink message accepted for control (we use SET_ATTITUDE_TARGET).
  2. Image topic, frame rate, and encoding for the FPV stream.
  3. Whether TIMESYNC handshake is required before control is accepted.
  4. Whether MAV_TYPE in heartbeat affects anything (we send QUADROTOR=2).
"""

import argparse
import asyncio
import logging
import os
import struct
import sys
import threading
import time
from collections import deque
from typing import Optional

import numpy as np
from PIL import Image
from pymavlink import mavutil

from attitude_filter import GravityEstimator, GRAVITY
from dcl_vision_receiver import DCLVisionReceiver
from trajectory_logger import TrajectoryLogger

# Image.BILINEAR was removed in Pillow 10; Resampling.BILINEAR is canonical in ≥9.1.
try:
    _BILINEAR = Image.Resampling.BILINEAR
except AttributeError:
    _BILINEAR = Image.BILINEAR  # type: ignore[attr-defined]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("run_vq1")

# Module-level instances — populated by run() before the control loop starts.
_vision_receiver: Optional[DCLVisionReceiver] = None
_mavlink_rx: Optional["_MAVLinkReceiver"] = None
_timesync = None  # DCLTimesync instance; imported lazily from dcl_mavlink_adapter
_trajectory_logger: Optional[TrajectoryLogger] = None
_telem_lock = threading.Lock()

# Use the fine-tuned model (FPV_TILT=+20, EVENT_CAMERA=False) if available,
# otherwise fall back to original distill model.
_FINETUNE_PATH = os.path.join(os.path.dirname(__file__), "models_release", "aigp_finetune_tilt_final.zip")
_DISTILL_PATH  = os.path.join(os.path.dirname(__file__), "models_release", "aigp_distill_final.zip")
CANONICAL_MODEL_PATH = _DISTILL_PATH   # force the 14-ch event-cam distill; finetune (==aigp_distill_11600000_steps) diverged ~14M and tumbles


def _check_model_path(path: str):
    """Fail loudly if the model file is missing. No silent fallbacks."""
    if not os.path.exists(path):
        logger.error(
            "Model not found at: %s\n"
            "  The canonical model path is ./models_release/aigp_distill_final.zip\n"
            "  Copy the model from drone-race-sim/models_release/ if needed.\n"
            "  Do NOT change this path to a Desktop location — see obsidian/fragilities.md.",
            path,
        )
        sys.exit(1)


# ---------------------------------------------------------------------------
# Vision stream stub — replace when DCL simulator ships
# ---------------------------------------------------------------------------

# Gate for the stub path. main() sets this to True when --allow-stub-vision
# is passed. Left False by default so any call site (direct import, test
# harness, or the production control loop) raises loudly instead of silently
# feeding zeros into the policy and emitting plausible-looking MAVLink.
# Submission must leave this False; replace _get_vision_frame() with a real
# DCL image-API call when the sim ships and the guard becomes unreachable.
_allow_stub_vision: bool = False


def _get_vision_frame() -> np.ndarray:
    """
    Returns the latest 48x48 RGB frame from the DCL UDP vision stream.

    Pulls the most-recently-decoded JPEG frame from the background
    DCLVisionReceiver thread (port 5600, chunked per VADR-TS-002 s4.6),
    resizes to 48x48 RGB, and returns as uint8 numpy array.

    The --allow-stub-vision guard is preserved: when that flag is active
    (local dev only) a black frame is returned instead. Submission runs
    must NOT pass --allow-stub-vision — the real receiver path is taken
    by default.
    """
    if _allow_stub_vision:
        return np.zeros((48, 48, 3), dtype=np.uint8)

    if _vision_receiver is None:
        raise NotImplementedError(
            "Vision stream is a stub: DCLVisionReceiver not started. "
            "Pass --allow-stub-vision for local dev/testing only. "
            "See obsidian/fragilities.md §Silent failure chain."
        )

    frame = _vision_receiver.get_latest_frame()  # (H, W, 3) uint8 RGB
    img = Image.fromarray(frame).resize((48, 48), _BILINEAR)
    return np.array(img, dtype=np.uint8)


# ---------------------------------------------------------------------------
# Telemetry — live MAVLink parsing from DCL simulator UDP stream
# ---------------------------------------------------------------------------

_latest_telemetry = {
    "attitude":          (0.0, 0.0, 0.0),   # (roll, pitch, yaw) rad  — from ATTITUDE
    "velocity":          (0.0, 0.0, 0.0),   # body angular rates rad/s — from ATTITUDE/HIGHRES_IMU
    "position":          (0.0, 0.0, 0.0),   # (x, y, z) m NED         — from LOCAL_POSITION_NED
    "linear_velocity":   (0.0, 0.0, 0.0),   # (vx, vy, vz) m/s NED   — from LOCAL_POSITION_NED
    "acceleration":      (0.0, 0.0, 0.0),   # (ax, ay, az) m/s^2 FRD body — from HIGHRES_IMU (specific force)
    "gravity_frd":       (0.0, 0.0, GRAVITY),  # gravity-down FRD (norm~=g) from A2 GravityEstimator
}
_race_started: bool = False  # set True by _MAVLinkReceiver once GO fires this session
_countdown_armed: bool = False  # True once a GENUINE future countdown is observed
_armed_race_start_ms: int = -1  # the fresh race_start latched when the countdown armed
_active_gate: int = -1  # latest active_gate index from ENCAPSULATED_DATA race status;
                        # the hardcoded vision-servo uses this to sequence gates.
START_MARGIN_MS = 100  # delay GO this far past race_start to absorb detect lag;
                       # starting late is safe, starting early is an instant DQ.
STALE_DELTA_MS = -10000  # race_start whose delta is below this is a stale echo from
                         # a prior race (sim_boot already far past it) — ignore it.


class _MAVLinkReceiver:
    """Background receive loop on the sim_conn pymavlink connection.

    Calls sim_conn.recv_match() continuously and routes ATTITUDE, HIGHRES_IMU,
    and LOCAL_POSITION_NED into _latest_telemetry. Replaces the old raw-socket
    _MAVLinkTelemetryReceiver now that all traffic flows through the single
    udpin: connection established at startup.
    """

    def __init__(self, sim_conn):
        self._conn = sim_conn
        self._thread: Optional[threading.Thread] = None
        self._running = False
        self._last_race_log = 0.0
        self._last_arm_log = 0.0
        self._last_imu_log = 0.0
        self._imu_count = 0
        self._grav_est = GravityEstimator()          # A2 filter, updated per IMU sample
        self._last_imu_usec = None                   # for dt from message time, not wall-clock
        self._accel_window = deque(maxlen=10)        # rolling accel for a denoised GO seed
        self._dt_fallback_count = 0                  # times dt fell back to nominal
        self._dt_sample_count = 0                    # total _imu_dt calls (deduped)
        self._dt_last_warn = 0                        # fallback count at last warning
        self._dup_dropped = 0                        # duplicate-time_usec samples dropped
        self._dup_last_log = 0                        # dup count at last log

    def _imu_dt(self, time_usec):
        """dt from HIGHRES_IMU.time_usec deltas (sim time), NOT wall-clock arrival.
        UDP jitter must not corrupt gyro integration. Nominal 1/115 fallback + warn."""
        nominal = 1.0 / 115.0
        self._dt_sample_count += 1
        if time_usec is None:
            self._warn_dt("HIGHRES_IMU.time_usec missing"); return nominal
        if self._last_imu_usec is None:
            self._last_imu_usec = time_usec; return nominal          # first sample
        delta = time_usec - self._last_imu_usec
        self._last_imu_usec = time_usec
        if delta <= 0 or delta > 1_000_000:                          # non-monotonic / >1s gap
            self._warn_dt(f"bad time_usec delta={delta}us"); return nominal
        return delta / 1e6

    def _note_dup(self):
        """Counted log for dropped duplicate-time_usec samples. Measured on the live
        v3385 sim: HIGHRES_IMU repeats the previous time_usec ~12-28% of samples while
        armed. We DROP those (see _loop) rather than feed the filter a repeated instant
        with a fallback dt — dedup is strictly better than integrate-twice."""
        self._dup_dropped += 1
        if self._dup_dropped == 1 or self._dup_dropped - self._dup_last_log >= 1000:
            self._dup_last_log = self._dup_dropped
            total = self._dup_dropped + self._dt_sample_count
            logger.info("[GravityFilter] dropped %d duplicate-time_usec IMU samples "
                        "(%.1f%% of %d received) — dedup, not double-integrate.",
                        self._dup_dropped, 100.0 * self._dup_dropped / max(1, total), total)

    def _warn_dt(self, why):
        # COUNTED warning (was warn-once). With duplicate stamps now DROPPED upstream
        # (see _note_dup / _loop), this fires only for genuine gaps: missing time_usec,
        # true reordering (delta<0, measured 0% live), or a >1s hole. Rare, but kept as
        # a guard. Warn on the first hit, then periodically with the running rate.
        self._dt_fallback_count += 1
        if (self._dt_fallback_count == 1
                or self._dt_fallback_count - self._dt_last_warn >= 500):
            self._dt_last_warn = self._dt_fallback_count
            frac = 100.0 * self._dt_fallback_count / max(1, self._dt_sample_count)
            logger.warning("[GravityFilter] dt fallback to 1/115 nominal (%s); "
                           "count=%d (%.1f%% of %d non-dup IMU samples) — genuine "
                           "gap/reorder; gyro integration runs on nominal dt for those "
                           "steps.", why, self._dt_fallback_count,
                           frac, self._dt_sample_count)

    def start(self) -> None:
        self._running = True
        self._thread = threading.Thread(
            target=self._loop, daemon=True, name="DCLMAVLinkRX"
        )
        self._thread.start()
        logger.info("[DCL MAVLink RX] Started")

    def stop(self) -> None:
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)
        logger.info("[DCL MAVLink RX] Stopped")

    def _loop(self) -> None:
        while self._running:
            try:
                msg = self._conn.recv_match(blocking=False)
            except ConnectionResetError:
                logger.warning("[DCL MAVLink RX] ConnectionResetError — stopping receiver")
                return
            except OSError:
                break

            if msg is None:
                time.sleep(0.001)
                continue

            msg_type = msg.get_type()
            if msg_type == "BAD_DATA":
                continue

            if msg_type == "HEARTBEAT":
                armed = bool(msg.base_mode & mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED)
                now = time.time()
                if now - self._last_arm_log >= 1.0:
                    self._last_arm_log = now
                    logger.info(
                        "[Sim HB] armed=%s  base_mode=0x%02x  system_status=%d",
                        armed, msg.base_mode, msg.system_status,
                    )
            elif msg_type == "ATTITUDE":
                with _telem_lock:
                    _latest_telemetry["attitude"] = (
                        float(msg.roll), float(msg.pitch), float(msg.yaw)
                    )
                    _latest_telemetry["velocity"] = (
                        float(msg.rollspeed), float(msg.pitchspeed), float(msg.yawspeed)
                    )
            elif msg_type == "HIGHRES_IMU":
                imu_usec = getattr(msg, "time_usec", None)
                # Drop duplicate-timestamp samples (~12-28% live). The sim repeats the
                # previous time_usec; feeding that repeated instant to the filter with a
                # fallback dt double-integrates it. Dedup is strictly better. Genuine
                # reordering (delta<0, 0% live) still hits the <=0 guard in _imu_dt.
                if imu_usec is not None and imu_usec == self._last_imu_usec:
                    self._note_dup()
                    continue
                accel = np.array([msg.xacc, msg.yacc, msg.zacc], dtype=float)    # FRD
                gyro = np.array([msg.xgyro, msg.ygyro, msg.zgyro], dtype=float)  # FRD
                dt = self._imu_dt(imu_usec)
                g_frd = self._grav_est.update(accel, gyro, dt)                   # gravity-down FRD
                self._accel_window.append(accel)
                with _telem_lock:
                    _latest_telemetry["velocity"] = (
                        float(msg.xgyro), float(msg.ygyro), float(msg.zgyro)
                    )
                    _latest_telemetry["acceleration"] = (
                        float(msg.xacc), float(msg.yacc), float(msg.zacc)
                    )
                    _latest_telemetry["gravity_frd"] = (
                        float(g_frd[0]), float(g_frd[1]), float(g_frd[2])
                    )
                # Throttled IMU diagnostic (~1 Hz): confirms accel capture, the
                # FRD sign convention (zacc ~ -9.81 at rest, level), and stream rate.
                self._imu_count += 1
                now = time.time()
                if now - self._last_imu_log >= 1.0:
                    hz = (self._imu_count / (now - self._last_imu_log)
                          if self._last_imu_log else float(self._imu_count))
                    self._last_imu_log = now
                    self._imu_count = 0
                    logger.info(
                        "[IMU] ~%.0f Hz  accel(FRD m/s2)=[%+.3f %+.3f %+.3f]  "
                        "gyro(rad/s)=[%+.3f %+.3f %+.3f]",
                        hz, float(msg.xacc), float(msg.yacc), float(msg.zacc),
                        float(msg.xgyro), float(msg.ygyro), float(msg.zgyro),
                    )
            elif msg_type == "LOCAL_POSITION_NED":
                with _telem_lock:
                    _latest_telemetry["position"] = (
                        float(msg.x), float(msg.y), float(msg.z)
                    )
                    _latest_telemetry["linear_velocity"] = (
                        float(msg.vx), float(msg.vy), float(msg.vz)
                    )
            elif msg_type == "ENCAPSULATED_DATA":
                raw = bytes(msg.data)
                if raw and raw[0] == 1:
                    try:
                        _, sim_boot_ms, race_start_ms, race_finish_ns, active_gate, _ = \
                            struct.unpack_from("<BQqqIq", raw)
                    except struct.error as exc:
                        logger.debug("[Race Status] unpack error: %s", exc)
                    else:
                        global _race_started, _countdown_armed, _armed_race_start_ms
                        global _active_gate
                        _active_gate = int(active_gate)
                        # race_start_ms is the server clock value (ms since sim
                        # boot) AT WHICH the race goes live. Two failure modes:
                        #   (1) Gating on race_start_ms >= 0 fired at the top of
                        #       the countdown -> DQ'd for an early start.
                        #   (2) At startup the sim echoes a STALE race_start from
                        #       a prior race against a huge current sim_boot, so
                        #       sim_boot >= race_start + margin is instantly true.
                        # Fix: only arm GO-detection after we observe a GENUINE
                        # countdown — race_start_ms >= 0 AND delta > 0 (the start
                        # is really ahead of us). Latch that fresh race_start and
                        # fire GO when sim_boot crosses the LATCHED value, never a
                        # later stale echo. A wildly-negative delta is a stale echo
                        # and is ignored for both arming and firing.
                        delta = race_start_ms - sim_boot_ms
                        is_stale = race_start_ms >= 0 and delta < STALE_DELTA_MS

                        if not is_stale:
                            if (not _countdown_armed) and race_start_ms >= 0 and delta > 0:
                                _countdown_armed = True
                                _armed_race_start_ms = race_start_ms
                                logger.info(
                                    "[Race] countdown armed — race_start=%d is %d ms ahead "
                                    "(sim_boot=%d)", race_start_ms, delta, sim_boot_ms,
                                )

                            if (_countdown_armed and not _race_started
                                    and sim_boot_ms >= _armed_race_start_ms + START_MARGIN_MS):
                                _race_started = True
                                # Seed the gravity filter at GO from a short rolling
                                # mean of accel (denoised vs a single sample) when it is
                                # within the 1g gate; otherwise KEEP the filter's current
                                # estimate (converged since receiver start) — a better
                                # fallback than a level reset. Always reset dt tracking.
                                if self._accel_window:
                                    a_mean = np.mean(np.stack(self._accel_window), axis=0)
                                    if 0.85 * GRAVITY <= float(np.linalg.norm(a_mean)) <= 1.15 * GRAVITY:
                                        self._grav_est.reset(gravity_frd=-a_mean)
                                self._last_imu_usec = None
                                logger.info(
                                    "[Race] GO — sim_boot=%d >= armed race_start=%d + margin=%d "
                                    "(actual margin used = %d ms), model output unblocked",
                                    sim_boot_ms, _armed_race_start_ms, START_MARGIN_MS,
                                    sim_boot_ms - _armed_race_start_ms,
                                )
                        now = time.time()
                        if now - self._last_race_log >= 1.0:
                            self._last_race_log = now
                            # delta > 0 = countdown running (GO is in the future);
                            # delta <= 0 = race live. Watch delta cross 0 at "Go!".
                            # armed=False until a genuine future countdown is seen;
                            # stale=True flags a prior-race echo we ignore.
                            logger.info(
                                "[Race Status] sim_boot=%d  race_start=%d  delta=%d  "
                                "armed=%s  stale=%s  active_gate=%d  race_finish_ns=%d",
                                sim_boot_ms, race_start_ms, delta,
                                _countdown_armed, is_stale, active_gate, race_finish_ns,
                            )


async def _get_telemetry() -> dict:
    """Return a snapshot of the latest telemetry parsed from the DCL MAVLink stream."""
    with _telem_lock:
        return dict(_latest_telemetry)


def _get_telemetry_sync() -> dict:
    """Sync snapshot for polling threads (e.g. TrajectoryLogger)."""
    with _telem_lock:
        return dict(_latest_telemetry)


# ---------------------------------------------------------------------------
# Vision stream as async generator
# ---------------------------------------------------------------------------

async def _vision_stream(target_hz: float = 50.0):
    """Yield vision frames at target_hz until cancelled."""
    interval = 1.0 / target_hz
    while True:
        yield _get_vision_frame()
        await asyncio.sleep(interval)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def run(
    host: str,
    port: int,
    hz: float,
    vision_port: int = 5600,
    log_trajectory: bool = False,
    control_mode: str = "attitude",
    device: str = None,
    hover_probe: float = None,
    controller: str = "model",
):
    global _vision_receiver, _mavlink_rx, _timesync, _trajectory_logger

    from dcl_mavlink_adapter import SCUBALabMAVLinkAdapter, DCLTimesync

    use_vision_servo = (controller == "vision-servo")
    if use_vision_servo:
        logger.warning(
            "[CONTROLLER] HARDCODED vision-servo (NO RL model). Gate detection + "
            "guidance gains are UNTUNED against the real sim — calibrate first "
            "(tools/vision_servo_dryrun.py on a real frame; --hover-probe for the "
            "hover fraction). See vq1_vision_servo.py CALIBRATE banner."
        )
    else:
        _check_model_path(CANONICAL_MODEL_PATH)
        logger.info("Model path verified: %s", CANONICAL_MODEL_PATH)

    if hover_probe is not None:
        logger.warning(
            "[HOVER PROBE] DIAGNOSTIC MODE — model loaded but IGNORED. After GO, "
            "commanding fixed thrust=%.3f with zero body rates for ~%.1fs, logging vz. "
            "Read the per-frame [HOVER PROBE] vz and the DONE verdict. Run separate "
            "races at several thrust values to bracket the hover point. NOT for submission.",
            hover_probe, 2.0,
        )

    sim_conn = None
    if not _allow_stub_vision:
        # Start vision receiver in background thread (VADR-TS-002 s4.6 chunked JPEG on UDP).
        _vision_receiver = DCLVisionReceiver(port=vision_port)
        _vision_receiver.start()

        # Establish pymavlink connection in listen mode — the sim connects to us.
        # wait_heartbeat() blocks until the sim's first heartbeat arrives, then
        # sets sim_conn.target_system / target_component automatically.
        logger.info("Waiting for heartbeat from DCL simulator on UDP port %d...", port)
        sim_conn = mavutil.mavlink_connection(f'udpin:0.0.0.0:{port}')
        sim_conn.wait_heartbeat()
        logger.info(
            "Heartbeat received — system: %d, component: %d",
            sim_conn.target_system, sim_conn.target_component,
        )

        # ARM the drone (matches controller.arm() from PyAIPilotExample exactly).
        sim_conn.mav.command_long_send(
            sim_conn.target_system,
            sim_conn.target_component,
            mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
            0,   # confirmation
            1,   # param1 = 1 → arm
            0, 0, 0, 0, 0, 0,
        )
        logger.info("ARM command sent to system %d", sim_conn.target_system)

        # Start TIMESYNC at 10 Hz (matches timesync.py from PyAIPilotExample).
        _timesync = DCLTimesync(sim_conn)
        _timesync.start()

        # Reset race-start state for a fresh session so a stale race_start cached
        # in module globals (or a prior race echoed on the wire) can't fire GO.
        global _race_started, _countdown_armed, _armed_race_start_ms
        _race_started = False
        _countdown_armed = False
        _armed_race_start_ms = -1

        # Start MAVLink receive loop to keep _latest_telemetry current.
        _mavlink_rx = _MAVLinkReceiver(sim_conn)
        _mavlink_rx.start()

    # Hardcoded vision-servo command source (no RL). It owns its own inputs:
    # pulls the FULL-RES frame from the vision receiver, the IMU-derived
    # gravity+gyro from _latest_telemetry, and the sequencing gate from
    # _active_gate. Resets on the GO edge so a prior race can't bleed in.
    command_source = None
    if use_vision_servo:
        from vq1_vision_servo import VisionServoController
        _servo = VisionServoController()
        _servo_prev_race = {"flag": False}

        def command_source():
            r = _race_started
            if r and not _servo_prev_race["flag"]:
                _servo.reset()
            _servo_prev_race["flag"] = r
            frame = (_vision_receiver.get_latest_frame()
                     if _vision_receiver is not None
                     else __import__("numpy").zeros((360, 640, 3), dtype="uint8"))
            with _telem_lock:
                telem = dict(_latest_telemetry)
            return _servo.command(frame, telem, active_gate=_active_gate)

    adapter = SCUBALabMAVLinkAdapter(
        model_path=CANONICAL_MODEL_PATH,
        udp_host=host,
        udp_port=port,
        target_hz=hz,
        sim_conn=sim_conn,
        control_mode=control_mode,
        race_started_source=lambda: _race_started,
        device=device,
        hover_probe=hover_probe,
        command_source=command_source,
    )

    if log_trajectory:
        _trajectory_logger = TrajectoryLogger(
            telemetry_source=_get_telemetry_sync,
            hz=10.0,
            output_path="trajectory_log.csv",
        )
        _trajectory_logger.start()

    logger.info("Starting control loop at %.0f Hz — mode: %s", hz, control_mode)
    if _allow_stub_vision:
        logger.warning(
            "Vision stream: STUB (black frames) — --allow-stub-vision is active. "
            "DO NOT use for VQ1 submission."
        )
    else:
        logger.info(
            "Vision stream: DCLVisionReceiver on UDP port %d. "
            "MAVLink connection on UDP port %d.",
            vision_port,
            port,
        )

    try:
        if not _allow_stub_vision and _vision_receiver is not None:
            logger.info("Waiting for race to start (no vision frames yet)...")
            while _vision_receiver.frames_received == 0:
                await asyncio.sleep(0.1)
            logger.info("Vision stream active — starting control loop")
        await adapter.run_loop(
            vision_stream_generator=_vision_stream(hz),
            telemetry_source=_get_telemetry,
        )
    except KeyboardInterrupt:
        logger.info("Interrupted by user.")
    finally:
        logger.info(
            "Shutting down. Frames sent: %d, Send errors: %d",
            adapter.frame_count,
            adapter.send_errors,
        )
        if adapter.send_errors > 0:
            logger.warning(
                "%d send errors occurred. Check UDP host/port and network connectivity.",
                adapter.send_errors,
            )
        if _timesync is not None:
            _timesync.stop()
        if _mavlink_rx is not None:
            _mavlink_rx.stop()
        if _vision_receiver is not None:
            _vision_receiver.stop()
        if _trajectory_logger is not None:
            _trajectory_logger.stop()
        adapter.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SCUBA LAB VQ1 Entry Point")
    parser.add_argument("--host", default="127.0.0.1", help="DCL simulator UDP host (stub mode fallback)")
    parser.add_argument("--port", type=int, default=14550,
                        help="Local UDP port to listen on for DCL MAVLink connection (default 14550)")
    parser.add_argument("--hz", type=float, default=250.0,
                        help="Control command rate in Hz (default 250, matching PyAIPilotExample CONTROL_HZ)")
    parser.add_argument("--control-mode", default="attitude",
                        choices=["attitude", "rates", "actuator"],
                        help="attitude/rates: SET_ATTITUDE_TARGET type_mask=128 (CTBR); "
                             "actuator: SET_ACTUATOR_CONTROL_TARGET group 0 at --hz")
    parser.add_argument("--controller", default="model",
                        choices=["model", "vision-servo"],
                        help="model: distilled RL policy (default, needs the model + "
                             "vision frames). vision-servo: HARDCODED gate-centering "
                             "controller, no RL (vq1_vision_servo.py). Reuses the same "
                             "arm/TIMESYNC/GO/SET_ATTITUDE_TARGET path.")
    parser.add_argument("--vision-port", type=int, default=5600,
                        help="UDP port for DCL FPV vision stream (VADR-TS-002 s4.6, default 5600)")
    parser.add_argument(
        "--allow-stub-vision",
        action="store_true",
        help="Permit running against zero-filled vision frames. Local dev/testing ONLY — "
             "submission must NOT use this flag. See obsidian/fragilities.md.",
    )
    parser.add_argument(
        "--log-trajectory",
        action="store_true",
        help="Start TrajectoryLogger — samples telemetry at 10 Hz and writes "
             "trajectory_log.csv on shutdown. Debug tool only.",
    )
    parser.add_argument(
        "--device",
        default=None,
        choices=["cuda", "mps", "cpu"],
        help="Inference device (default: auto-select cuda > mps > cpu).",
    )
    parser.add_argument(
        "--hover-probe",
        type=float,
        default=None,
        metavar="THRUST",
        help="DIAGNOSTIC — ignore the model and command a FIXED thrust THRUST in "
             "[0,1] with zero body rates for ~2s after GO, logging vz from "
             "LOCAL_POSITION_NED each frame. Run separate races at e.g. 0.15/0.20/"
             "0.25/0.30/0.35; the thrust where vz~=0 is DCL's hover fraction. "
             "NOT for submission.",
    )
    args = parser.parse_args()
    if args.hover_probe is not None and not (0.0 <= args.hover_probe <= 1.0):
        parser.error("--hover-probe THRUST must be in [0, 1]")

    # Assign explicitly via globals() so this line remains a module-level
    # update even if a future refactor wraps the __main__ block in a
    # def main() (in which case a plain `_allow_stub_vision = ...` would
    # silently create a function-local that the guard never sees).
    # Tests exercise this via subprocess, not just module-attribute
    # patching — see TestPackageStructure.test_cli_flag_actually_enables_stub.
    globals()['_allow_stub_vision'] = args.allow_stub_vision

    asyncio.run(run(args.host, args.port, args.hz, args.vision_port,
                    args.log_trajectory, args.control_mode, args.device,
                    args.hover_probe, args.controller))
