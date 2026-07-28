#!/usr/bin/env python3
"""
sniff.py — raw UDP sniffer on port 14550.
No pymavlink. Bind, print every datagram (source addr + byte length), total at end.
Run while the DCL sim is actively streaming. Exits after 15 s of silence or 15 s total.
"""
import socket
import time

PORT = 14550
WINDOW = 15.0

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
sock.bind(("0.0.0.0", PORT))
print(f"Bound to 0.0.0.0:{PORT} — listening for {WINDOW:.0f} s...")

count = 0
deadline = time.time() + WINDOW
while True:
    remaining = deadline - time.time()
    if remaining <= 0:
        break
    sock.settimeout(remaining)
    try:
        data, addr = sock.recvfrom(65535)
    except socket.timeout:
        break
    except OSError as exc:
        print(f"  socket error: {exc}")
        break
    count += 1
    print(f"  [{count:>4}] src={addr[0]}:{addr[1]:<6}  len={len(data)}")

sock.close()
print(f"\nTotal datagrams received in {WINDOW:.0f} s: {count}")
