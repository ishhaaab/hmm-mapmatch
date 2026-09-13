"""Unit tests for src.geo shared geometry helpers."""
import pytest

from src.geo import M_PER_DEG_LAT, bearing_deg, haversine_m, project_to_segment


def test_haversine_zero():
    assert haversine_m(12.9, 77.6, 12.9, 77.6) == pytest.approx(0.0, abs=1e-9)


def test_haversine_equator_degree():
    # 1 deg of lon at the equator ~ 111.19 km.
    assert haversine_m(0.0, 0.0, 0.0, 1.0) == pytest.approx(111_194.9, rel=1e-3)


def test_bearing_quadrants():
    assert bearing_deg(0.0, 0.0, 0.0, 1.0) == pytest.approx(90.0)
    assert bearing_deg(0.0, 0.0, 0.0, -1.0) == pytest.approx(270.0)
    assert bearing_deg(0.0, 0.0, 1.0, 0.0) == pytest.approx(0.0)
    assert bearing_deg(0.0, 0.0, -1.0, 0.0) == pytest.approx(180.0)


def test_project_to_segment_midpoint_and_clamp():
    # Horizontal segment 0->1 deg lon at the equator.
    frac, dist, _lat_f, lon_f = project_to_segment(0.0, 0.5, 0.0, 0.0, 0.0, 1.0)
    assert frac == pytest.approx(0.5)
    assert dist == pytest.approx(0.0, abs=1e-6)
    assert lon_f == pytest.approx(0.5)

    # Beyond the far endpoint clamps to 1.0 and measures to the endpoint.
    # Planar projection (111_320 m/deg) vs great-circle haversine differ
    # ~0.1%, so the tolerance is loose by design.
    frac2, dist2, _, _ = project_to_segment(0.0, 2.0, 0.0, 0.0, 0.0, 1.0)
    assert frac2 == pytest.approx(1.0)
    assert dist2 == pytest.approx(haversine_m(0.0, 1.0, 0.0, 2.0), rel=2e-3)


def test_project_to_segment_perpendicular_offset():
    # 100 m north of the segment midpoint -> perpendicular distance ~100 m.
    _frac, dist, _lat_f, _lon_f = project_to_segment(
        100.0 / M_PER_DEG_LAT, 0.5, 0.0, 0.0, 0.0, 1.0
    )
    assert dist == pytest.approx(100.0, rel=1e-6)


def test_project_to_segment_degenerate():
    frac, dist, lat_f, lon_f = project_to_segment(0.1, 0.2, 0.0, 0.0, 0.0, 0.0)
    assert frac == pytest.approx(0.0)
    assert dist == pytest.approx(haversine_m(0.1, 0.2, 0.0, 0.0), rel=1e-6)
    assert (lat_f, lon_f) == (0.0, 0.0)