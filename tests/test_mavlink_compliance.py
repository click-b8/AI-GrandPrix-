"""
Test: MAVLink frame compliance.

Verifies that MAVLinkFrameBuilder produces frames that:
1. Pass CRC validation by a pymavlink reference parser
2. Have monotonically incrementing sequence bytes
3. Encode body rates at the correct CTBR scale (policy_output * MAX_BODY_RATE)
4. Set type_mask = 128 (IGNORE_ATTITUDE)
5. Match byte-for-byte against pymavlink's own encoder for a fixed input
"""

import socket
import struct
import threading
import time

import numpy as np
import pytest
from pymavlink.dialects.v20 import common as mav_common

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from dcl_mavlink_adapter import (
    MAVLinkFrameBuilder,
    MAX_BODY_RATE,
    TYPE_MASK_BODY_RATES_ONLY,
    _compute_crc16,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def parse_frames(raw: bytes):
    """Use pymavlink reference parser to decode all frames in raw bytes."""
    mlnk = mav_common.MAVLink(None)
    mlnk.robust_parsing = True
    msgs = []
    for b in raw:
        m = mlnk.parse_char(bytes([b]))
        if m:
            msgs.append(m)
    return msgs


def build_reference_frame(body_roll_rate, body_pitch_rate, body_yaw_rate, thrust, time_boot_ms=1000):
    """Build a reference SET_ATTITUDE_TARGET frame via pymavlink's own encoder."""
    ref_mav = mav_common.MAVLink(None, srcSystem=1, srcComponent=1)
    msg = mav_common.MAVLink_set_attitude_target_message(
        time_boot_ms=time_boot_ms,
        target_system=1,
        target_component=1,
        type_mask=TYPE_MASK_BODY_RATES_ONLY,
        q=[1.0, 0.0, 0.0, 0.0],
        body_roll_rate=body_roll_rate,
        body_pitch_rate=body_pitch_rate,
        body_yaw_rate=body_yaw_rate,
        thrust=thrust,
    )
    return msg.pack(ref_mav)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestCRC:
    def test_crc_matches_pymavlink(self):
        """Our CRC must match what pymavlink computes for the same bytes."""
        builder = MAVLinkFrameBuilder()
        frame = builder.set_attitude_target(1000, 3.6, -2.4, 1.2, 0.5)
        # Parse with pymavlink — it will reject if CRC is wrong
        msgs = parse_frames(frame)
        assert len(msgs) == 1, "pymavlink rejected frame (CRC or structure invalid)"

    def test_corrupted_crc_rejected(self):
        """Deliberately corrupt the CRC; pymavlink must not decode a valid message."""
        builder = MAVLinkFrameBuilder()
        frame = bytearray(builder.set_attitude_target(1000, 1.0, 0.0, 0.0, 0.5))
        frame[-1] ^= 0xFF  # flip last CRC byte
        msgs = parse_frames(bytes(frame))
        # pymavlink robust_parsing returns MAVLink_bad_data for CRC failures,
        # not a valid typed message. No SET_ATTITUDE_TARGET should be decoded.
        valid_msgs = [m for m in msgs if m.get_type() != 'BAD_DATA']
        assert len(valid_msgs) == 0, (
            f"pymavlink decoded a valid message from a corrupted frame: {valid_msgs}"
        )


class TestSequence:
    def test_sequence_increments(self):
        """Sequence byte must increment by 1 each frame, wrap at 255."""
        builder = MAVLinkFrameBuilder()
        seqs = []
        for _ in range(10):
            frame = builder.set_attitude_target(1000, 0.0, 0.0, 0.0, 0.5)
            # Sequence byte is at offset 4 in MAVLink v2 frame
            seqs.append(frame[4])
        assert seqs == list(range(10)), f"Sequence not incrementing: {seqs}"

    def test_sequence_wraps_at_256(self):
        builder = MAVLinkFrameBuilder()
        builder._seq = 254
        frames = [builder.set_attitude_target(i, 0.0, 0.0, 0.0, 0.5) for i in range(3)]
        seqs = [f[4] for f in frames]
        assert seqs == [254, 255, 0], f"Sequence wrap failed: {seqs}"


class TestSemantics:
    def test_body_rates_scale_correctly(self):
        """Decoded body rates must equal policy_output * MAX_BODY_RATE."""
        builder = MAVLinkFrameBuilder()
        policy_roll, policy_pitch, policy_yaw = 0.3, -0.2, 0.1
        frame = builder.set_attitude_target(
            time_boot_ms=500,
            body_roll_rate=policy_roll * MAX_BODY_RATE,
            body_pitch_rate=policy_pitch * MAX_BODY_RATE,
            body_yaw_rate=policy_yaw * MAX_BODY_RATE,
            thrust=0.5,
        )
        msgs = parse_frames(frame)
        assert len(msgs) == 1
        m = msgs[0]
        assert abs(m.body_roll_rate  - policy_roll  * MAX_BODY_RATE) < 1e-4
        assert abs(m.body_pitch_rate - policy_pitch * MAX_BODY_RATE) < 1e-4
        assert abs(m.body_yaw_rate   - policy_yaw   * MAX_BODY_RATE) < 1e-4

    def test_type_mask_is_ignore_attitude(self):
        builder = MAVLinkFrameBuilder()
        frame = builder.set_attitude_target(1000, 1.0, 0.0, 0.0, 0.5)
        msgs = parse_frames(frame)
        assert msgs[0].type_mask == TYPE_MASK_BODY_RATES_ONLY == 128

    def test_thrust_passes_through(self):
        builder = MAVLinkFrameBuilder()
        frame = builder.set_attitude_target(1000, 0.0, 0.0, 0.0, thrust=0.75)
        msgs = parse_frames(frame)
        assert abs(msgs[0].thrust - 0.75) < 1e-4

    def test_quaternion_is_identity(self):
        """Quaternion must be identity [1,0,0,0] since FC ignores it."""
        builder = MAVLinkFrameBuilder()
        frame = builder.set_attitude_target(1000, 0.0, 0.0, 0.0, 0.5)
        msgs = parse_frames(frame)
        q = msgs[0].q
        assert abs(q[0] - 1.0) < 1e-6
        assert abs(q[1]) < 1e-6
        assert abs(q[2]) < 1e-6
        assert abs(q[3]) < 1e-6


class TestByteExact:
    def test_frame_matches_pymavlink_reference(self):
        """Our encoded frame must match pymavlink's own encoder byte-for-byte.

        This is the key test from the VQ1 plan's red-team critique #5:
        'pymavlink agrees with itself' is vacuous; we need byte-level
        equality against an external reference.

        We can't compare the full frame byte-for-byte because sequence
        bytes differ between our builder and pymavlink's internal counter.
        We compare: STX, payload_len, incompat, compat, sysid, compid,
        msgid, payload (all except the seq byte at offset 4), and CRC.
        """
        roll, pitch, yaw, thrust = 3.6, -2.4, 1.2, 0.5
        t = 1000

        builder = MAVLinkFrameBuilder()
        builder._seq = 0  # force seq=0 to match pymavlink's initial state
        our_frame = builder.set_attitude_target(t, roll, pitch, yaw, thrust)

        ref_frame = build_reference_frame(roll, pitch, yaw, thrust, t)

        # Compare everything except the seq byte (offset 4)
        our_no_seq = our_frame[:4] + our_frame[5:]
        ref_no_seq = ref_frame[:4] + ref_frame[5:]

        assert our_no_seq == ref_no_seq, (
            f"Frame mismatch (excluding seq byte):\n"
            f"  ours: {our_frame.hex()}\n"
            f"  ref:  {ref_frame.hex()}"
        )


class TestHeartbeat:
    def test_heartbeat_parses(self):
        builder = MAVLinkFrameBuilder()
        frame = builder.heartbeat()
        msgs = parse_frames(frame)
        assert len(msgs) == 1
        assert msgs[0].get_type() == 'HEARTBEAT'
        assert msgs[0].type == 2  # MAV_TYPE_QUADROTOR


class TestMockReceiver:
    """End-to-end UDP round-trip: send 100 frames, verify all parse cleanly."""

    def test_udp_roundtrip_100_frames(self):
        # Bind a receiver on a random local port
        recv_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        recv_sock.bind(("127.0.0.1", 0))
        recv_sock.settimeout(2.0)
        port = recv_sock.getsockname()[1]

        send_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        builder = MAVLinkFrameBuilder()

        received = []
        errors = []

        def receiver():
            mlnk = mav_common.MAVLink(None)
            mlnk.robust_parsing = True
            try:
                while len(received) + len(errors) < 100:
                    data, _ = recv_sock.recvfrom(4096)
                    for b in data:
                        m = mlnk.parse_char(bytes([b]))
                        if m:
                            received.append(m)
            except socket.timeout:
                pass
            finally:
                recv_sock.close()

        t = threading.Thread(target=receiver, daemon=True)
        t.start()

        for i in range(100):
            frame = builder.set_attitude_target(
                time_boot_ms=i * 20,
                body_roll_rate=float(i % 10) * 0.1 * MAX_BODY_RATE,
                body_pitch_rate=0.0,
                body_yaw_rate=0.0,
                thrust=0.5,
            )
            send_sock.sendto(frame, ("127.0.0.1", port))
            time.sleep(0.001)

        t.join(timeout=3.0)
        send_sock.close()

        assert len(received) == 100, f"Only received {len(received)}/100 frames"
        assert len(errors) == 0

        # Verify monotonic sequence (wrapping)
        seqs = [m.get_header().seq for m in received]
        for i in range(1, len(seqs)):
            expected = (seqs[i - 1] + 1) & 0xFF
            assert seqs[i] == expected, f"Sequence gap at frame {i}: {seqs[i-1]} -> {seqs[i]}"

        # Verify body rate scaling on frame 5: roll=5*0.1=0.5 * MAX_BODY_RATE=6.0
        assert abs(received[5].body_roll_rate - 5 * 0.1 * MAX_BODY_RATE) < 1e-3


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
