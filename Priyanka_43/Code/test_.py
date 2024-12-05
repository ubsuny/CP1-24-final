'''
unit testing module for unit_converter.py and motion_direction.py
'''
import pytest

from unit_converter import feet_to_meters, yards_to_meters

from motion_direction import calculate_direction

def test_feet_to_meters():
    '''
    function to test the foot to meter conversion
    '''
    assert feet_to_meters(1) == 0.3048   # checking if 1 foot = 0.3048 meter or not

def test_yards_to_meters():
    '''
    function to test the yards to meters conversion
    '''
    assert yards_to_meters(1) == 0.9144   # checking if 1 yard = 0.9144 meter or not

def test_calculate_direction():
    """Tests for acceleration along the x and z axes"""
    # Test for acceleration along the x-axis
    theta, phi = calculate_direction(1, 0, 0)
    assert pytest.approx(theta, 0.1) == 90  # Polar angle θ
    assert pytest.approx(phi, 0.1) == 0     # Azimuthal angle φ

    # Test for acceleration along the z-axis
    theta, phi = calculate_direction(0, 0, 1)
    assert pytest.approx(theta, 0.1) == 0   # Polar angle θ
