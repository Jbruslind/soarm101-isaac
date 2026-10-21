#!/usr/bin/env python3
"""Generate a 128-byte EDID 1.3 blob for a virtual monitor.

The NVIDIA Xorg driver will only drive a display output if it believes a
monitor is attached. In a headless server/VM we fake one with
`ConnectedMonitor` + `CustomEDID`; this script produces that EDID for an
arbitrary resolution/refresh using CVT reduced-blanking (v1) timings.

Usage: make_edid.py WIDTH HEIGHT REFRESH OUTPUT_PATH
"""

from __future__ import annotations

import math
import struct
import sys


def cvt_rb_timing(width: int, height: int, refresh: int) -> dict:
    """CVT reduced blanking v1: fixed 160px h-blank, >=460us v-blank."""
    h_blank, h_front, h_sync = 160, 48, 32
    v_front, v_sync, v_back_min = 3, 5, 6
    min_vblank_us = 460.0

    # Approximate line period, then the number of lines needed for >=460us of v-blank.
    line_us = (1e6 / refresh - min_vblank_us) / height
    v_blank = max(math.ceil(min_vblank_us / line_us) + 1, v_front + v_sync + v_back_min)
    v_total = height + v_blank
    h_total = width + h_blank
    pixel_clock_khz = math.ceil(h_total * v_total * refresh / 1e3 / 250) * 250  # 0.25 MHz steps
    return dict(
        pclk_10khz=pixel_clock_khz // 10,
        h_active=width, h_blank=h_blank, h_front=h_front, h_sync=h_sync,
        v_active=height, v_blank=v_blank, v_front=v_front, v_sync=v_sync,
        h_freq_khz=pixel_clock_khz / h_total,
    )


def detailed_timing(t: dict, size_mm=(600, 340)) -> bytes:
    if t["pclk_10khz"] > 0xFFFF:
        raise ValueError("pixel clock exceeds EDID 1.3 limit (655.35 MHz); lower resolution or refresh")
    ha, hb, va, vb = t["h_active"], t["h_blank"], t["v_active"], t["v_blank"]
    hso, hsw, vso, vsw = t["h_front"], t["h_sync"], t["v_front"], t["v_sync"]
    w_mm, h_mm = size_mm
    return bytes([
        t["pclk_10khz"] & 0xFF, t["pclk_10khz"] >> 8,
        ha & 0xFF, hb & 0xFF, ((ha >> 8) << 4) | (hb >> 8),
        va & 0xFF, vb & 0xFF, ((va >> 8) << 4) | (vb >> 8),
        hso & 0xFF, hsw & 0xFF,
        ((vso & 0xF) << 4) | (vsw & 0xF),
        ((hso >> 8) << 6) | ((hsw >> 8) << 4) | ((vso >> 4) << 2) | (vsw >> 4),
        w_mm & 0xFF, h_mm & 0xFF, ((w_mm >> 8) << 4) | (h_mm >> 8),
        0, 0,
        0x1A,  # digital separate sync, +hsync, -vsync (CVT-RB convention)
    ])


def text_descriptor(tag: int, text: str) -> bytes:
    body = text.encode("ascii")[:13]
    if len(body) < 13:
        body += b"\n" + b" " * (12 - len(body))
    return bytes([0, 0, 0, tag, 0]) + body


def range_descriptor(refresh: int, t: dict) -> bytes:
    v_min, v_max = max(1, min(refresh, 24)), min(255, max(refresh, 60) + 5)
    h_min = max(1, int(t["h_freq_khz"] * 0.3))
    h_max = min(255, int(t["h_freq_khz"]) + 5)
    max_pclk_10mhz = min(255, t["pclk_10khz"] // 1000 + 1)
    return bytes([0, 0, 0, 0xFD, 0, v_min, v_max, h_min, h_max, max_pclk_10mhz,
                  0x00, 0x0A, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20])


def make_edid(width: int, height: int, refresh: int) -> bytes:
    t = cvt_rb_timing(width, height, refresh)
    mfg = (ord("S") - 64) << 10 | (ord("O") - 64) << 5 | (ord("A") - 64)  # "SOA"
    edid = bytearray()
    edid += b"\x00\xff\xff\xff\xff\xff\xff\x00"
    edid += struct.pack(">H", mfg) + struct.pack("<H", 0x0101) + struct.pack("<I", 0)
    edid += bytes([1, 2026 - 1990, 1, 3])                  # week, year, EDID 1.3
    edid += bytes([0x80, 60, 34, 0x78, 0x0A])               # digital, 60x34cm, gamma 2.2, RGB + preferred timing
    edid += bytes([0xEE, 0x91, 0xA3, 0x54, 0x4C, 0x99, 0x26, 0x0F, 0x50, 0x54])  # sRGB chromaticity
    edid += bytes([0, 0, 0])                                # no established timings
    edid += b"\x01\x01" * 8                                 # no standard timings
    edid += detailed_timing(t)
    edid += range_descriptor(refresh, t)
    edid += text_descriptor(0xFC, "SOARM-VDISP")
    edid += text_descriptor(0xFF, f"{width}x{height}@{refresh}")
    edid += bytes([0])                                      # no extension blocks
    edid += bytes([(-sum(edid)) & 0xFF])
    assert len(edid) == 128
    return bytes(edid)


def main() -> int:
    if len(sys.argv) != 5:
        print(__doc__.strip().splitlines()[-1], file=sys.stderr)
        return 2
    width, height, refresh = (int(v) for v in sys.argv[1:4])
    with open(sys.argv[4], "wb") as f:
        f.write(make_edid(width, height, refresh))
    return 0


if __name__ == "__main__":
    sys.exit(main())
