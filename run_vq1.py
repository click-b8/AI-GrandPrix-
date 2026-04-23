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
import time

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("run_vq1")

CANONICAL_MODEL_PATH = os.path.join(os.path.dirname(__file__), "models_release", "aigp_distill_final.zip")


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
    STUB: Returns a black frame until the DCL simulator image API is known.

    Guarded by the module-level _allow_stub_vision flag. By default, raises
    NotImplementedError — running the control loop against zero frames would
    emit a spec-compliant MAVLink stream of body-rate commands derived from
    constant-zero inference, which is the exact silent-failure pattern the
    project has been fighting (see obsidian/fragilities.md §Silent failure
    chain). Pass --allow-stub-vision on the command line for local dev only.

    When DCL ships the simulator, replace this function body with:
        frame = <DCL image API call>
        return frame  # shape (48, 48, 3), dtype uint8, RGB

    The policy was trained on 48x48 RGB + event channels. The adapter
    handles resizing and zero-pads the event channels automatically.
    """
    if not _allow_stub_vision:
        raise NotImplementedError(
            "Vision stream is a stub (returns zero-filled black frames). "
            "Running the control loop against this would emit a valid-looking "
            "MAVLink stream of constant body-rate commands derived from "
            "garbage inference — the silent-failure pattern the project has "
            "explicit guardrails against. Replace _get_vision_frame() with "
            "the real DCL image-API call when the simulator ships, or pass "
            "--allow-stub-vision on the command line for local dev only."
        )
    return np.zeros((48, 48, 3), dtype=np.uint8)


# ---------------------------------------------------------------------------
# Telemetry stub — replace with real MAVLink telemetry parsing
# ---------------------------------------------------------------------------

_latest_telemetry = {
    "attitude":  (0.0, 0.0, 0.0),
    "velocity":  (0.0, 0.0, 0.0),
    "position":  (0.0, 0.0, 0.0),
}


async def _get_telemetry() -> dict:
    """
    STUB: Returns zeroed telemetry until DCL telemetry parsing is wired up.

    Replace with actual MAVLink ATTITUDE / LOCAL_POSITION_NED parsing
    from the DCL simulator's telemetry stream.
    """
    return _latest_telemetry


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

async def run(host: str, port: int, hz: float):
    from dcl_mavlink_adapter import SCUBALabMAVLinkAdapter

    _check_model_path(CANONICAL_MODEL_PATH)
    logger.info("Model path verified: %s", CANONICAL_MODEL_PATH)

    adapter = SCUBALabMAVLinkAdapter(
        model_path=CANONICAL_MODEL_PATH,
        udp_host=host,
        udp_port=port,
        target_hz=hz,
    )

    logger.info("Starting control loop -> %s:%d at %.0f Hz", host, port, hz)
    if _allow_stub_vision:
        logger.warning(
            "Vision stream: STUB (black frames) — --allow-stub-vision is active. "
            "DO NOT use for VQ1 submission."
        )
    else:
        logger.info("Vision stream: real _get_vision_frame() implementation detected.")

    try:
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
        adapter.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SCUBA LAB VQ1 Entry Point")
    parser.add_argument("--host", default="127.0.0.1", help="DCL simulator UDP host")
    parser.add_argument("--port", type=int, default=14540, help="DCL simulator UDP port")
    parser.add_argument("--hz", type=float, default=50.0, help="Control command rate (50-120 Hz)")
    parser.add_argument(
        "--allow-stub-vision",
        action="store_true",
        help="Permit running against zero-filled vision frames. Local dev/testing ONLY — "
             "submission must NOT use this flag. See obsidian/fragilities.md.",
    )
    args = parser.parse_args()

    # Assign explicitly via globals() so this line remains a module-level
    # update even if a future refactor wraps the __main__ block in a
    # def main() (in which case a plain `_allow_stub_vision = ...` would
    # silently create a function-local that the guard never sees).
    # Tests exercise this via subprocess, not just module-attribute
    # patching — see TestPackageStructure.test_cli_flag_actually_enables_stub.
    globals()['_allow_stub_vision'] = args.allow_stub_vision

    asyncio.run(run(args.host, args.port, args.hz))
