#!/usr/bin/env python3
"""UE4 `.pak` (version 11) reader — list / extract / diff, no external tools.

Why this exists: the AI-GP sim ships every level inside a single
`FlightSim-WindowsNoEditor.pak`. Each sim update we need to (a) pull the
`levels/MAP_*track*.umap`/`.uexp` for a course, and (b) diff a new build against
the committed course to decide whether `course_gates_cm.json` must be
re-extracted. This tool does both directly from the pak index, which in these
builds is UNENCRYPTED with compression method = None — so a payload compare is a
faithful proxy for "did the level change" (identical umap+uexp bytes ⇒ identical
serialized actor RootScene transforms ⇒ identical gates/spawn).

Validated against sim v1.0.3379 and v1.0.3385 (4.5 GB paks, 15086 entries).
Only pak version 11 is supported/verified; other versions abort loudly.

Usage:
    python tools/pak_extract.py list   <pak> [--filter SUBSTR]
    python tools/pak_extract.py extract <pak> <SUBSTR> [--out DIR]
    python tools/pak_extract.py diff   <pakA> <pakB> [--filter SUBSTR]

`diff` reports the full file-set delta (added/removed) plus a payload-hash
compare of every file whose path contains --filter (default: "track" maps).
"""
import argparse
import hashlib
import os
import struct
import sys

PAK_MAGIC = 0x5A6F12E1
SUPPORTED_VERSION = 11
FOOTER_SIZE = 221  # v11: guid16 + enc1 + magic4 + ver4 + idxoff8 + idxsz8 + hash20 + methods160
# In-file FPakEntry header for an uncompressed v11 entry:
#   int64 Offset, int64 Size, int64 UncompressedSize, int32 CompressionMethodIndex,
#   byte[20] Hash, uint8 bEncrypted, uint32 CompressionBlockSize
UNCOMPRESSED_HEADER_LEN = 8 + 8 + 8 + 4 + 20 + 1 + 4


def _read_fstring(buf, o):
    """UE FString: int32 length; positive=ASCII/latin1, negative=UTF-16LE. NUL-terminated."""
    (n,) = struct.unpack_from("<i", buf, o)
    o += 4
    if n == 0:
        return "", o
    if n < 0:
        n = -n
        s = buf[o:o + n * 2].decode("utf-16-le").rstrip("\x00")
        o += n * 2
    else:
        s = buf[o:o + n].decode("latin1").rstrip("\x00")
        o += n
    return s, o


def decode_entry(blob, off):
    """Decode one FPakEntry from the EncodedPakEntries blob (UE4 DecodePakEntry bitfield)."""
    o = off
    (val,) = struct.unpack_from("<I", blob, o); o += 4
    method = (val >> 23) & 0x3f
    off32 = val & (1 << 31)
    unc32 = val & (1 << 30)
    sz32 = val & (1 << 29)
    enc = (val >> 22) & 1
    nblocks = (val >> 6) & 0xffff
    if off32:
        (offset,) = struct.unpack_from("<I", blob, o); o += 4
    else:
        (offset,) = struct.unpack_from("<Q", blob, o); o += 8
    if unc32:
        (uncsize,) = struct.unpack_from("<I", blob, o); o += 4
    else:
        (uncsize,) = struct.unpack_from("<Q", blob, o); o += 8
    if method != 0:
        if sz32:
            (size,) = struct.unpack_from("<I", blob, o); o += 4
        else:
            (size,) = struct.unpack_from("<Q", blob, o); o += 8
    else:
        size = uncsize
    return dict(offset=offset, size=size, uncsize=uncsize, method=method,
                enc=enc, nblocks=nblocks)


def parse_pak(path):
    """Parse a v11 pak index -> dict(path, sz, mount, num_entries, files{path: enc_off}, encoded)."""
    with open(path, "rb") as f:
        f.seek(0, 2)
        sz = f.tell()
        f.seek(sz - FOOTER_SIZE)
        foot = f.read(FOOTER_SIZE)
        (magic,) = struct.unpack_from("<I", foot, 17)
        (ver,) = struct.unpack_from("<i", foot, 21)
        if magic != PAK_MAGIC:
            raise ValueError(f"{path}: not a UE pak (bad magic {magic:#x})")
        if ver != SUPPORTED_VERSION:
            raise ValueError(f"{path}: pak version {ver} unsupported (this tool is verified for v{SUPPORTED_VERSION} only)")
        enc_index = foot[16]
        if enc_index:
            raise ValueError(f"{path}: index is encrypted; this tool only handles unencrypted indices")
        (idx_off,) = struct.unpack_from("<q", foot, 25)
        (idx_sz,) = struct.unpack_from("<q", foot, 33)
        f.seek(idx_off)
        idx = f.read(idx_sz)

    o = 0
    mount, o = _read_fstring(idx, o)
    (num_entries,) = struct.unpack_from("<i", idx, o); o += 4
    o += 8  # PathHashSeed (uint64)
    (has_phi,) = struct.unpack_from("<i", idx, o); o += 4
    if has_phi:
        o += 8 + 8 + 20  # PathHashIndex offset, size, hash
    (has_fdi,) = struct.unpack_from("<i", idx, o); o += 4
    if not has_fdi:
        raise ValueError(f"{path}: no FullDirectoryIndex (cannot recover file paths)")
    (fdi_off,) = struct.unpack_from("<q", idx, o); o += 8
    (fdi_sz,) = struct.unpack_from("<q", idx, o); o += 8
    o += 20  # FDI hash
    (enc_sz,) = struct.unpack_from("<i", idx, o); o += 4
    encoded = idx[o:o + enc_sz]

    with open(path, "rb") as f:
        f.seek(fdi_off)
        fdi = f.read(fdi_sz)
    o2 = 0
    (ndirs,) = struct.unpack_from("<i", fdi, o2); o2 += 4
    files = {}
    for _ in range(ndirs):
        dname, o2 = _read_fstring(fdi, o2)
        (nfiles,) = struct.unpack_from("<i", fdi, o2); o2 += 4
        for _ in range(nfiles):
            fname, o2 = _read_fstring(fdi, o2)
            (enc_off,) = struct.unpack_from("<i", fdi, o2); o2 += 4
            files[mount + dname + fname] = enc_off
    return dict(path=path, sz=sz, mount=mount, num_entries=num_entries,
                files=files, encoded=encoded)


def read_payload(pak_path, entry):
    """Return the decompressed file bytes for an entry (uncompressed method=None only)."""
    if entry["method"] != 0:
        raise NotImplementedError("compressed entry: only method=None (uncompressed) is supported")
    with open(pak_path, "rb") as f:
        f.seek(entry["offset"] + UNCOMPRESSED_HEADER_LEN)
        return f.read(entry["uncsize"])


def _sha16(b):
    return hashlib.sha256(b).hexdigest()[:16]


def cmd_list(args):
    p = parse_pak(args.pak)
    keys = sorted(k for k in p["files"] if args.filter.lower() in k.lower())
    print(f"{p['path']}: {len(p['files'])} entries, mount={p['mount']!r}; "
          f"{len(keys)} match {args.filter!r}")
    for k in keys:
        print("  ", k)


def cmd_extract(args):
    p = parse_pak(args.pak)
    matches = [k for k in p["files"] if args.name.lower() in k.lower()]
    if not matches:
        sys.exit(f"no file matches {args.name!r}")
    os.makedirs(args.out, exist_ok=True)
    for k in matches:
        ent = decode_entry(p["encoded"], p["files"][k])
        data = read_payload(p["path"], ent)
        out = os.path.join(args.out, os.path.basename(k))
        with open(out, "wb") as f:
            f.write(data)
        print(f"  extracted {k} -> {out} ({len(data)} bytes, sha={_sha16(data)})")


def cmd_diff(args):
    a = parse_pak(args.pak_a)
    b = parse_pak(args.pak_b)
    sa, sb = set(a["files"]), set(b["files"])
    print(f"A {a['path']}\n  bytes={a['sz']} entries={a['num_entries']} files={len(sa)}")
    print(f"B {b['path']}\n  bytes={b['sz']} entries={b['num_entries']} files={len(sb)}")
    print(f"pak size delta B-A = {b['sz'] - a['sz']} bytes")

    added, removed = sorted(sb - sa), sorted(sa - sb)
    print(f"\nfull file-set: {len(added)} added, {len(removed)} removed")
    for x in added:
        print("   +", x)
    for x in removed:
        print("   -", x)

    flt = args.filter.lower()
    keys = sorted((sa | sb) & {k for k in (sa | sb) if flt in k.lower()})
    print(f"\npayload-hash compare of {len(keys)} files matching {args.filter!r}:")
    changed = 0
    for k in keys:
        ha = hb = None
        if k in a["files"]:
            ea = decode_entry(a["encoded"], a["files"][k]); ha = (ea["uncsize"], _sha16(read_payload(a["path"], ea)))
        if k in b["files"]:
            eb = decode_entry(b["encoded"], b["files"][k]); hb = (eb["uncsize"], _sha16(read_payload(b["path"], eb)))
        status = "IDENTICAL" if ha and hb and ha == hb else "CHANGED"
        if status != "IDENTICAL":
            changed += 1
        print(f"  [{status}] {k}")
        print(f"      A: {ha}   B: {hb}")
    print(f"\n{changed} of {len(keys)} matched files changed.")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    pl = sub.add_parser("list", help="list files in a pak")
    pl.add_argument("pak")
    pl.add_argument("--filter", default="", help="substring filter (case-insensitive)")
    pl.set_defaults(func=cmd_list)

    pe = sub.add_parser("extract", help="extract file(s) matching a substring")
    pe.add_argument("pak")
    pe.add_argument("name", help="substring of the path to extract")
    pe.add_argument("--out", default=".", help="output directory")
    pe.set_defaults(func=cmd_extract)

    pd = sub.add_parser("diff", help="diff two paks (file set + payload hashes)")
    pd.add_argument("pak_a")
    pd.add_argument("pak_b")
    pd.add_argument("--filter", default="track", help="substring of paths to hash-compare (default: track)")
    pd.set_defaults(func=cmd_diff)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
