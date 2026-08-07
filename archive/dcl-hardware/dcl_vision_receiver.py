"""
SCUBA LAB - DCL Vision Stream Receiver
=======================================
Receives the DCL simulator FPV camera stream over UDP and reconstructs
JPEG frames from chunked packets per VADR-TS-002 s4.6.

Usage (drop-in replacement for _get_vision_frame in run_vq1.py):

    from dcl_vision_receiver import DCLVisionReceiver

    receiver = DCLVisionReceiver(port=5600)
    receiver.start()

    # In _get_vision_frame():
    frame = receiver.get_latest_frame()  # (360, 640, 3) uint8 RGB
    return frame  # dcl_adapter.py resizes to 48x48

    # On shutdown:
    receiver.stop()

Packet structure (VADR-TS-002 s4.6):
    frame_id      uint32  4B  Unique sequence ID
    chunk_id      uint16  2B  Index of this chunk (0 to total_chunks-1)
    total_chunks  uint16  2B  Total chunks for this frame
    jpeg_size     uint32  4B  Total size of reconstructed JPEG
    payload_size  uint32  4B  Size of JPEG data in this packet
    sim_time_ns   uint64  8B  Simulation timestamp (nanoseconds)
    [payload]             variable  JPEG data slice
"""

import io
import logging
import socket
import struct
import threading
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

# Header format: little-endian, 24 bytes
# frame_id(4) chunk_id(2) total_chunks(2) jpeg_size(4) payload_size(4) sim_time_ns(8)
_HEADER_FMT  = "<IHHIIQ"
_HEADER_SIZE = struct.calcsize(_HEADER_FMT)  # 24 bytes per spec


class DCLVisionReceiver:
    """
    Threaded UDP receiver for the DCL FPV vision stream.

    Reassembles chunked JPEG packets into complete frames and
    decodes them to numpy arrays. Thread-safe latest-frame access.
    """

    def __init__(self, host: str = "0.0.0.0", port: int = 5600,
                 timeout: float = 2.0, max_udp_packet: int = 65535):
        self.host = host
        self.port = port
        self.timeout = timeout
        self.max_udp_packet = max_udp_packet

        self._sock: Optional[socket.socket] = None
        self._thread: Optional[threading.Thread] = None
        self._running = False

        self._latest_frame: Optional[np.ndarray] = None
        self._frame_lock = threading.Lock()

        self._buffer: dict = {}
        self._buffer_total: dict = {}

        self.frames_received = 0
        self.chunks_dropped = 0

    def start(self):
        """Bind UDP socket and start receiver thread."""
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._sock.settimeout(self.timeout)
        self._sock.bind((self.host, self.port))
        self._running = True
        self._thread = threading.Thread(target=self._receive_loop,
                                        daemon=True, name="DCLVisionReceiver")
        self._thread.start()
        logger.info("[DCL Vision] Listening on %s:%d", self.host, self.port)

    def stop(self):
        """Stop receiver thread and close socket."""
        self._running = False
        if self._sock:
            self._sock.close()
        if self._thread:
            self._thread.join(timeout=3.0)
        logger.info("[DCL Vision] Stopped. Frames received: %d, Chunks dropped: %d",
                    self.frames_received, self.chunks_dropped)

    def get_latest_frame(self) -> np.ndarray:
        """
        Return the most recently decoded frame.
        Returns black (360x640x3) if no frame has arrived yet.
        """
        with self._frame_lock:
            if self._latest_frame is None:
                logger.warning("[DCL Vision] No frame received yet - returning black frame")
                return np.zeros((360, 640, 3), dtype=np.uint8)
            return self._latest_frame.copy()

    def is_alive(self) -> bool:
        """True if the receiver thread is running."""
        return self._thread is not None and self._thread.is_alive()

    def _receive_loop(self):
        """Main receive loop - runs in background thread."""
        while self._running:
            try:
                data, _ = self._sock.recvfrom(self.max_udp_packet)
            except socket.timeout:
                continue
            except OSError:
                break
            self._process_packet(data)

    def _process_packet(self, data: bytes):
        """Parse one UDP packet and reassemble frame when complete."""
        header = data[:_HEADER_SIZE]
        payload = data[_HEADER_SIZE:]

        try:
            frame_id, chunk_id, total_chunks, jpeg_size, payload_size, sim_time_ns = \
                struct.unpack(_HEADER_FMT, header)
        except struct.error:
            self.chunks_dropped += 1
            return

        if len(data) < _HEADER_SIZE:
            self.chunks_dropped += 1
            return

        if len(payload) < payload_size:
            self.chunks_dropped += 1
            return

        payload = payload[:payload_size]

        if frame_id not in self._buffer:
            self._buffer[frame_id] = {}
            self._buffer_total[frame_id] = total_chunks

        self._buffer[frame_id][chunk_id] = payload

        if len(self._buffer[frame_id]) == total_chunks:
            self._reassemble_frame(frame_id, total_chunks, jpeg_size)
            old_ids = [fid for fid in self._buffer if fid < frame_id]
            for fid in old_ids:
                del self._buffer[fid]
                del self._buffer_total[fid]
                self.chunks_dropped += 1

    def _reassemble_frame(self, frame_id: int, total_chunks: int, jpeg_size: int):
        """Reassemble chunks into a JPEG and decode to numpy."""
        chunks = self._buffer[frame_id]
        jpeg_data = b"".join(chunks[i] for i in range(total_chunks))[:jpeg_size]

        try:
            from PIL import Image
            img = Image.open(io.BytesIO(jpeg_data)).convert("RGB")
            frame = np.array(img, dtype=np.uint8)
        except Exception as exc:
            logger.warning("[DCL Vision] JPEG decode failed for frame %d: %s",
                           frame_id, exc)
            return

        with self._frame_lock:
            self._latest_frame = frame

        self.frames_received += 1
        if self.frames_received % 300 == 0:
            logger.info("[DCL Vision] Frames received: %d", self.frames_received)

        del self._buffer[frame_id]
        del self._buffer_total[frame_id]