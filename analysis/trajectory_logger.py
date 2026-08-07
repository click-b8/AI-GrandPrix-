"""
VQ1 Trajectory Logger — debugging tool, not part of submission.

Polls a telemetry source at a fixed rate and writes a CSV on stop().
Designed to run as a daemon thread inside run_vq1.py.

Telemetry source must return a dict with keys:
    position        — (x, y, z) m NED
    linear_velocity — (vx, vy, vz) m/s NED  (from LOCAL_POSITION_NED)
    attitude        — (roll, pitch, yaw) rad  (from ATTITUDE)

Standalone usage (binds its own MAVLink UDP socket):
    python3 trajectory_logger.py [--port 14550] [--hz 10] [--out trajectory_log.csv]
    NOTE: conflicts with run_vq1.py's telemetry receiver on the same port.
          Use standalone mode only without run_vq1.py running.
"""

import argparse
import csv
import logging
import socket
import threading
import time
from typing import Callable, Optional

logger = logging.getLogger(__name__)

CSV_HEADER = ["time_s", "x", "y", "z", "vx", "vy", "vz", "roll", "pitch", "yaw"]


class TrajectoryLogger:
    """Background polling logger — samples telemetry dict and writes CSV on stop()."""

    def __init__(
        self,
        telemetry_source: Callable[[], dict],
        hz: float = 10.0,
        output_path: str = "trajectory_log.csv",
    ):
        self._source = telemetry_source
        self._interval = 1.0 / hz
        self._output_path = output_path
        self._records: list = []
        self._thread: Optional[threading.Thread] = None
        self._running = False

    def start(self) -> None:
        self._running = True
        self._records = []
        self._thread = threading.Thread(
            target=self._log_loop, daemon=True, name="TrajectoryLogger"
        )
        self._thread.start()
        logger.info(
            "[TrajectoryLogger] Started — %.0f Hz -> %s",
            1.0 / self._interval, self._output_path,
        )

    def stop(self) -> None:
        self._running = False
        if self._thread:
            self._thread.join(timeout=3.0)
        self._save_csv()
        logger.info(
            "[TrajectoryLogger] Stopped — %d rows written to %s",
            len(self._records), self._output_path,
        )

    def _log_loop(self) -> None:
        t0 = time.monotonic()
        while self._running:
            t = time.monotonic()
            try:
                telem = self._source()
            except Exception as exc:
                logger.debug("[TrajectoryLogger] telemetry read error: %s", exc)
                time.sleep(self._interval)
                continue

            pos = telem.get("position",        (0.0, 0.0, 0.0))
            vel = telem.get("linear_velocity", (0.0, 0.0, 0.0))
            att = telem.get("attitude",         (0.0, 0.0, 0.0))

            self._records.append([
                round(t - t0, 4),
                round(float(pos[0]), 4), round(float(pos[1]), 4), round(float(pos[2]), 4),
                round(float(vel[0]), 4), round(float(vel[1]), 4), round(float(vel[2]), 4),
                round(float(att[0]), 4), round(float(att[1]), 4), round(float(att[2]), 4),
            ])
            time.sleep(self._interval)

    def _save_csv(self) -> None:
        with open(self._output_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(CSV_HEADER)
            writer.writerows(self._records)


# ---------------------------------------------------------------------------
# Standalone mode — own MAVLink UDP socket
# ---------------------------------------------------------------------------

def _run_standalone(port: int, hz: float, output_path: str) -> None:
    """Listen for MAVLink telemetry on a UDP socket and log to CSV."""
    try:
        from pymavlink.dialects.v20 import common as mav_common
    except ImportError:
        print("[TrajectoryLogger] pymavlink not installed.")
        return

    _state = {
        "position":        (0.0, 0.0, 0.0),
        "linear_velocity": (0.0, 0.0, 0.0),
        "attitude":        (0.0, 0.0, 0.0),
    }
    _lock = threading.Lock()

    def _recv_loop(sock: socket.socket) -> None:
        mav = mav_common.MAVLink(None)
        mav.robust_parsing = True
        while True:
            try:
                data, _ = sock.recvfrom(65535)
            except OSError:
                break
            try:
                msgs = mav.parse_buffer(data)
                if not msgs:
                    continue
                for msg in msgs:
                    t = msg.get_type()
                    with _lock:
                        if t == "ATTITUDE":
                            _state["attitude"] = (
                                float(msg.roll), float(msg.pitch), float(msg.yaw)
                            )
                        elif t == "LOCAL_POSITION_NED":
                            _state["position"] = (
                                float(msg.x), float(msg.y), float(msg.z)
                            )
                            _state["linear_velocity"] = (
                                float(msg.vx), float(msg.vy), float(msg.vz)
                            )
            except Exception:
                pass

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.settimeout(1.0)
    sock.bind(("0.0.0.0", port))
    print(f"[TrajectoryLogger] Listening on UDP :{port} — Ctrl+C to stop and save.")

    recv_thread = threading.Thread(target=_recv_loop, args=(sock,), daemon=True)
    recv_thread.start()

    def source() -> dict:
        with _lock:
            return dict(_state)

    log = TrajectoryLogger(telemetry_source=source, hz=hz, output_path=output_path)
    log.start()
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        pass
    finally:
        log.stop()
        sock.close()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s")
    parser = argparse.ArgumentParser(description="VQ1 Trajectory Logger (standalone)")
    parser.add_argument("--port", type=int, default=14550,
                        help="UDP port for MAVLink telemetry (default 14550)")
    parser.add_argument("--hz", type=float, default=10.0,
                        help="Sampling rate in Hz (default 10)")
    parser.add_argument("--out", default="trajectory_log.csv",
                        help="Output CSV path (default trajectory_log.csv)")
    args = parser.parse_args()
    _run_standalone(args.port, args.hz, args.out)
