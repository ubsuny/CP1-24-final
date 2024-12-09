"""Test for accurate direction from acceleration"""

import pytest
from direction_from_acceleration import direction_from_acceleration

def test_direction_from_acceleration():
    """Tests for acceleration along the x and y axes"""
    # Test for acceleration along the x-axis
    theta, phi = direction_from_acceleration(1, 0, 0)
    assert pytest.approx(theta, 0.1) == 90  # Polar angle θ
    assert pytest.approx(phi, 0.1) == 0     # Azimuthal angle φ

    # Test for acceleration along the z-axis
    theta, phi = direction_from_acceleration(0, 0, 1)
    assert pytest.approx(theta, 0.1) == 0   # Polar angle θ

def test_direction_zero_acceleration():
    """Test for zero acceleratation"""
    with pytest.raises(ValueError):
        direction_from_acceleration(0, 0, 0)
