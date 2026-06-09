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
import sys
import threading
import time
from typing import Optional

import numpy as np
from PIL import Image
from pymavlink import mavutil

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
CANONICAL_MODEL_PATH = _FINETUNE_PATH if os.path.exists(_FINETUNE_PATH) else _DISTILL_PATH


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
}


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

            if msg_type == "ATTITUDE":
                with _telem_lock:
                    _latest_telemetry["attitude"] = (
                        float(msg.roll), float(msg.pitch), float(msg.yaw)
                    )
                    _latest_telemetry["velocity"] = (
                        float(msg.rollspeed), float(msg.pitchspeed), float(msg.yawspeed)
                    )
            elif msg_type == "HIGHRES_IMU":
                with _telem_lock:
                    _latest_telemetry["velocity"] = (
                        float(msg.xgyro), float(msg.ygyro), float(msg.zgyro)
                    )
            elif msg_type == "LOCAL_POSITION_NED":
                with _telem_lock:
                    _latest_telemetry["position"] = (
                        float(msg.x), float(msg.y), float(msg.z)
                    )
                    _latest_telemetry["linear_velocity"] = (
                        float(msg.vx), float(msg.vy), float(msg.vz)
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
):
    global _vision_receiver, _mavlink_rx, _timesync, _trajectory_logger

    from dcl_mavlink_adapter import SCUBALabMAVLinkAdapter, DCLTimesync

    _check_model_path(CANONICAL_MODEL_PATH)
    logger.info("Model path verified: %s", CANONICAL_MODEL_PATH)

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

        # Start MAVLink receive loop to keep _latest_telemetry current.
        _mavlink_rx = _MAVLinkReceiver(sim_conn)
        _mavlink_rx.start()

    adapter = SCUBALabMAVLinkAdapter(
        model_path=CANONICAL_MODEL_PATH,
        udp_host=host,
        udp_port=port,
        target_hz=hz,
        sim_conn=sim_conn,
        control_mode=control_mode,
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
    args = parser.parse_args()

    # Assign explicitly via globals() so this line remains a module-level
    # update even if a future refactor wraps the __main__ block in a
    # def main() (in which case a plain `_allow_stub_vision = ...` would
    # silently create a function-local that the guard never sees).
    # Tests exercise this via subprocess, not just module-attribute
    # patching — see TestPackageStructure.test_cli_flag_actually_enables_stub.
    globals()['_allow_stub_vision'] = args.allow_stub_vision

    asyncio.run(run(args.host, args.port, args.hz, args.vision_port,
                    args.log_trajectory, args.control_mode))
