"""Tests for unit converter module"""

import unit_converter

def test_feet_to_meters():
    """Tests conversion from feet to meters"""
    assert unit_converter.feet_to_meters(1) == 0.3048
    assert unit_converter.feet_to_meters(10) == 3.048

def test_yards_to_meters():
    """Tests conversion from yards to meters"""
    assert unit_converter.yards_to_meters(1) == 0.9144
    assert unit_converter.yards_to_meters(5) == 4.572
