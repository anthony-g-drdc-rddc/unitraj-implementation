"""Parsing and formatting for PostGIS LINESTRING M values."""

from __future__ import annotations

import re
import struct
from dataclasses import dataclass
from typing import TypeAlias

import numpy as np
from numpy.typing import NDArray

PostGISLineStringM: TypeAlias = str | bytes | bytearray | memoryview

_FLOAT = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?"
_WKT_PATTERN = re.compile(
    rf"^\s*(?:SRID\s*=\s*(?P<srid>\d+)\s*;\s*)?"
    rf"LINESTRING\s*(?P<dim>M|ZM)?\s*\((?P<body>.*)\)\s*$",
    re.IGNORECASE | re.DOTALL,
)
_HEX_PATTERN = re.compile(r"^(?:\\x|0x)?[0-9a-fA-F]+$")


@dataclass(frozen=True, slots=True)
class ParsedLineStringM:
    """A parsed trajectory with columns [x, y, m]."""

    coordinates: NDArray[np.float64]
    srid: int | None = None

    def __post_init__(self) -> None:
        coordinates = np.asarray(self.coordinates, dtype=np.float64)
        if coordinates.ndim != 2 or coordinates.shape[1] != 3:
            raise ValueError("coordinates must have shape [N, 3] for X/Y/M")
        if coordinates.shape[0] < 2:
            raise ValueError("LINESTRING M must contain at least two points")
        if not np.isfinite(coordinates).all():
            raise ValueError("LINESTRING M contains a non-finite coordinate")
        object.__setattr__(self, "coordinates", coordinates)


def parse_linestringm(value: PostGISLineStringM) -> ParsedLineStringM:
    """Parse EWKT/WKT, WKB/EWKB bytes, or hex WKB for LINESTRING M.

    Supported textual forms include both ``LINESTRING M (...)`` and
    ``SRID=4326;LINESTRING M (...)``. Binary inputs support ISO WKB type 2002
    and PostGIS EWKB with the M and optional SRID flags.
    """

    if isinstance(value, str):
        stripped = value.strip()
        if _looks_like_hex_wkb(stripped):
            prefixless = stripped[2:] if stripped[:2].lower() in {"0x", "\\x"} else stripped
            return _parse_wkb(bytes.fromhex(prefixless))
        return _parse_wkt(stripped)
    if isinstance(value, (bytes, bytearray, memoryview)):
        raw = bytes(value)
        if not raw:
            raise ValueError("LINESTRING M binary value is empty")
        return _parse_wkb(raw)
    raise TypeError(
        "trajectory must be EWKT/WKT text, WKB/EWKB bytes, or hex WKB text"
    )


def to_ewkt(parsed: ParsedLineStringM, precision: int = 12) -> str:
    """Serialize coordinates to PostGIS-compatible EWKT."""

    if precision < 0:
        raise ValueError("precision must be non-negative")
    fmt = f".{{precision}}g"

    def render(value: float) -> str:
        return format(float(value), fmt.format(precision=precision))

    body = ", ".join(
        f"{render(x)} {render(y)} {render(m)}"
        for x, y, m in parsed.coordinates
    )
    prefix = f"SRID={parsed.srid};" if parsed.srid is not None else ""
    return f"{prefix}LINESTRING M ({body})"


def _looks_like_hex_wkb(value: str) -> bool:
    if not _HEX_PATTERN.fullmatch(value):
        return False
    prefixless = value[2:] if value[:2].lower() in {"0x", "\\x"} else value
    return len(prefixless) >= 10 and len(prefixless) % 2 == 0


def _parse_wkt(value: str) -> ParsedLineStringM:
    if re.search(r"\bEMPTY\b", value, re.IGNORECASE):
        raise ValueError("LINESTRING M EMPTY is not valid trajectory input")
    match = _WKT_PATTERN.fullmatch(value)
    if match is None:
        raise ValueError(
            "Expected LINESTRING M/EWKT text such as "
            "'SRID=4326;LINESTRING M (x y m, x y m)'"
        )
    dimension = (match.group("dim") or "").upper()
    if dimension != "M":
        raise ValueError("Input must explicitly be a 3-dimensional LINESTRING M")

    rows: list[tuple[float, float, float]] = []
    for point in match.group("body").split(","):
        values = point.strip().split()
        if len(values) != 3 or not all(re.fullmatch(_FLOAT, item) for item in values):
            raise ValueError(
                "Each LINESTRING M point must contain exactly X Y M numeric values"
            )
        rows.append((float(values[0]), float(values[1]), float(values[2])))
    srid_text = match.group("srid")
    return ParsedLineStringM(
        np.asarray(rows, dtype=np.float64),
        int(srid_text) if srid_text is not None else None,
    )


def _parse_wkb(raw: bytes) -> ParsedLineStringM:
    if len(raw) < 9:
        raise ValueError("WKB/EWKB value is too short")
    endian_marker = raw[0]
    if endian_marker == 0:
        endian = ">"
    elif endian_marker == 1:
        endian = "<"
    else:
        raise ValueError("Invalid WKB byte-order marker")

    offset = 1
    type_word = struct.unpack_from(endian + "I", raw, offset)[0]
    offset += 4

    ewkb_z = bool(type_word & 0x80000000)
    ewkb_m = bool(type_word & 0x40000000)
    ewkb_srid = bool(type_word & 0x20000000)
    if ewkb_z or ewkb_m or ewkb_srid:
        base_type = type_word & 0x0FFFFFFF
        has_z = ewkb_z
        has_m = ewkb_m
        has_srid = ewkb_srid
    else:
        has_srid = False
        if type_word >= 3000:
            base_type = type_word - 3000
            has_z = True
            has_m = True
        elif type_word >= 2000:
            base_type = type_word - 2000
            has_z = False
            has_m = True
        elif type_word >= 1000:
            base_type = type_word - 1000
            has_z = True
            has_m = False
        else:
            base_type = type_word
            has_z = False
            has_m = False

    if base_type != 2:
        raise ValueError("WKB/EWKB geometry must be a LINESTRING")
    if has_z:
        raise ValueError("LINESTRING ZM is not accepted; provide an XYM LINESTRING M")
    if not has_m:
        raise ValueError("WKB/EWKB LINESTRING must include an M ordinate")

    srid: int | None = None
    if has_srid:
        _require(raw, offset, 4)
        srid = struct.unpack_from(endian + "I", raw, offset)[0]
        offset += 4

    _require(raw, offset, 4)
    count = struct.unpack_from(endian + "I", raw, offset)[0]
    offset += 4
    if count < 2:
        raise ValueError("LINESTRING M must contain at least two points")

    dimensions = 3
    expected = count * dimensions * 8
    _require(raw, offset, expected)
    flat = np.asarray(
        struct.unpack_from(endian + ("d" * count * dimensions), raw, offset),
        dtype=np.float64,
    )
    offset += expected
    if offset != len(raw):
        raise ValueError("WKB/EWKB contains trailing bytes")
    return ParsedLineStringM(flat.reshape(count, dimensions), srid)


def _require(raw: bytes, offset: int, size: int) -> None:
    if offset + size > len(raw):
        raise ValueError("Truncated WKB/EWKB value")
