"""Test module for unitconversion.py"""
import pytest
from unitconversion import feet_to_meters, yards_to_meters

def test_feet_to_meters():
    """Test conversion from feet to meters."""
    assert feet_to_meters(1) == pytest.approx(0.3048, rel=1e-4)
    assert feet_to_meters(0) == 0.0
    assert feet_to_meters(10) == pytest.approx(3.048, rel=1e-4)
    assert feet_to_meters(-5) == pytest.approx(-1.524, rel=1e-4)

def test_yards_to_meters():
    """Test conversion from yards to meters."""
    assert yards_to_meters(1) == pytest.approx(0.9144, rel=1e-4)
    assert yards_to_meters(0) == 0.0
    assert yards_to_meters(5) == pytest.approx(4.572, rel=1e-4)
    assert yards_to_meters(-3) == pytest.approx(-2.7432, rel=1e-4)
