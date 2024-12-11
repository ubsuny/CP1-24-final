"""Unit tests for gps_distance module, testing distance calculation functions."""

from gps_distance import haversine, cumulative_distance, total_distance
import pytest

def test_haversine():
    """Test the Haversine distance function with multiple cases."""
    assert round(haversine(0, 0, 0, 1), 2) == 111194.93
    assert round(haversine(0, 0, 0, 1, unit="kilometers"), 2) == 111.19
    # Allow a small tolerance for pole-to-pole distance
    assert haversine(90, 0, -90, 0) == pytest.approx(20015114.35, rel=1e-5)
    try:
        haversine(0, 0, 1, 0, unit="feet")
    except ValueError:
        assert True

def test_cumulative_distance():
    """Test cumulative distance calculation across points."""
    latitudes = [0, 1, 2, 3]
    longitudes = [0, 1, 2, 3]
    distances = cumulative_distance(latitudes, longitudes, unit="meters")
    assert round(distances[-1], 2) == 471652.37

def test_total_distance():
    """Test total distance calculation across points."""
    latitudes = [0, 1, 2, 3]
    longitudes = [0, 1, 2, 3]
    assert round(total_distance(latitudes, longitudes, unit="meters"), 2) == 471652.37
