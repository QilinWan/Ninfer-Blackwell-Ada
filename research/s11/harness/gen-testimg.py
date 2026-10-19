#!/usr/bin/env python3
"""Recreate the synthetic test images the harness uses (PPM + PNG, no dependencies)."""
import struct, zlib
from pathlib import Path

out = Path("testimg"); out.mkdir(exist_ok=True)

def ppm(path, w, h, rgb):
    with open(path, "wb") as f:
        f.write(f"P6\n{w} {h}\n255\n".encode())
        f.write(bytes(rgb) * (w * h))

def png(path, w, h, rgb):
    raw = bytearray()
    for _ in range(h):
        raw.append(0)
        raw += bytes(rgb) * w
    def chunk(t, d):
        c = t + d
        return struct.pack(">I", len(d)) + c + struct.pack(">I", zlib.crc32(c) & 0xFFFFFFFF)
    ihdr = struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0)
    Path(path).write_bytes(b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", ihdr)
                           + chunk(b"IDAT", zlib.compress(bytes(raw))) + chunk(b"IEND", b""))

ppm(out / "red.ppm", 768, 768, (220, 20, 20))
ppm(out / "blue.ppm", 768, 768, (20, 40, 220))
png(out / "red_rgb.png", 768, 768, (220, 20, 20))
png(out / "shot_638x285.png", 638, 285, (30, 120, 200))
print("wrote", sorted(p.name for p in out.iterdir()))
