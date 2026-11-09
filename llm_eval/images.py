"""Deterministic image fixtures with no imaging dependency."""
import base64
import struct
import zlib

COLORS = {"red": (220, 30, 30), "green": (30, 170, 60), "blue": (30, 60, 220), "yellow": (240, 210, 30)}


def _chunk(kind, data):
    body = kind + data
    return struct.pack(">I", len(data)) + body + struct.pack(">I", zlib.crc32(body) & 0xFFFFFFFF)


def quadrants_png(names, size=96):
    """PNG of four solid quadrants in reading order: top-left, top-right, bottom-left, bottom-right."""
    if len(names) != 4 or any(n not in COLORS for n in names):
        raise ValueError("quadrants need four known color names")
    half = size // 2
    rows = []
    for y in range(size):
        row = bytearray([0])
        for x in range(size):
            row.extend(COLORS[names[(2 if y >= half else 0) + (1 if x >= half else 0)]])
        rows.append(bytes(row))
    header = struct.pack(">IIBBBBB", size, size, 8, 2, 0, 0, 0)
    return (b"\x89PNG\r\n\x1a\n" + _chunk(b"IHDR", header)
            + _chunk(b"IDAT", zlib.compress(b"".join(rows), 9)) + _chunk(b"IEND", b""))


def data_uri(png):
    return "data:image/png;base64," + base64.b64encode(png).decode()
