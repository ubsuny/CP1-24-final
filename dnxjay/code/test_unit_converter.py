"""Unit tests for unit_converter module, testing conversion functions."""

import math
from unit_converter import feet_to_meters, meters_to_feet

def test_feet_to_meters():
    """Test the feet_to_meters function with known values."""
    assert math.isclose(feet_to_meters(1), 0.3048, rel_tol=1e-5)
    assert math.isclose(feet_to_meters(10), 3.048, rel_tol=1e-5)

def test_meters_to_feet():
    """Test the meters_to_feet function with known values."""
    assert math.isclose(meters_to_feet(1), 3.28084, rel_tol=1e-5)
    assert math.isclose(meters_to_feet(10), 32.8084, rel_tol=1e-5)
