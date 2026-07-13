from __future__ import annotations

import struct

import numpy as np
import pytest

from unitraj.postgis import ParsedLineStringM, parse_linestringm, to_ewkt


def _little_endian_linestring_m(
    coordinates: list[tuple[float, float, float]], *, srid: int | None = None
) -> bytes:
    type_word = 2002 if srid is None else 0x40000000 | 0x20000000 | 2
    output = bytearray(struct.pack("<BI", 1, type_word))
    if srid is not None:
        output.extend(struct.pack("<I", srid))
    output.extend(struct.pack("<I", len(coordinates)))
    for point in coordinates:
        output.extend(struct.pack("<ddd", *point))
    return bytes(output)


def test_parse_ewkt_linestring_m() -> None:
    parsed = parse_linestringm(
        "SRID=4326;LINESTRING M (-73.5 45.5 100, -73.4 45.6 101)"
    )
    assert parsed.srid == 4326
    np.testing.assert_allclose(
        parsed.coordinates,
        [[-73.5, 45.5, 100.0], [-73.4, 45.6, 101.0]],
    )


def test_parse_linestringm_without_space() -> None:
    parsed = parse_linestringm("LINESTRINGM (1 2 3, 4 5 6)")
    assert parsed.srid is None
    np.testing.assert_allclose(parsed.coordinates[:, 2], [3.0, 6.0])


def test_parse_iso_wkb_m() -> None:
    raw = _little_endian_linestring_m([(1.0, 2.0, 3.0), (4.0, 5.0, 6.0)])
    parsed = parse_linestringm(raw)
    assert parsed.srid is None
    np.testing.assert_allclose(parsed.coordinates, [[1, 2, 3], [4, 5, 6]])


def test_parse_postgis_ewkb_m_with_srid() -> None:
    raw = _little_endian_linestring_m(
        [(1.0, 2.0, 3.0), (4.0, 5.0, 6.0)], srid=4326
    )
    parsed = parse_linestringm(raw.hex())
    assert parsed.srid == 4326
    np.testing.assert_allclose(parsed.coordinates, [[1, 2, 3], [4, 5, 6]])


def test_to_ewkt_round_trip() -> None:
    source = ParsedLineStringM(
        np.asarray([[1.25, 2.5, 3.75], [4.0, 5.0, 6.0]]), 4326
    )
    parsed = parse_linestringm(to_ewkt(source))
    assert parsed.srid == 4326
    np.testing.assert_allclose(parsed.coordinates, source.coordinates)


def test_reject_linestring_without_m() -> None:
    with pytest.raises(ValueError, match="explicitly"):
        parse_linestringm("LINESTRING (1 2, 3 4)")
