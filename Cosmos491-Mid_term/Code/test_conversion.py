"""
This module contains unit tests for the unit conversion 
functions feet_to_meters and yards_to_meters.
"""
from unit_converter_code import feet_to_meters, yards_to_meters
def test_feet_to_meters():
    """
    Tests the feet_to_meters function with positive, zero, and negative values.
    """
    assert feet_to_meters(1) == 0.3048  # 1 foot = 0.3048 meters
    assert feet_to_meters(10) == 3.048  # 10 feet = 3.048 meters
    assert feet_to_meters(0) == 0       # 0 feet should return 0 meters
    assert feet_to_meters(-5) == -1.524  # Negative value for edge case

def test_yards_to_meters():
    """
    Tests the yards_to_meters function with positive, zero, and negative values.
    """
    assert yards_to_meters(1) == 0.9144  # 1 yard = 0.9144 meters
    assert yards_to_meters(50) == 45.72  # 50 yards = 45.72 meters
    assert yards_to_meters(0) == 0       # 0 yards should return 0 meters
    assert yards_to_meters(-10) == -9.144  # Negative value for edge case
