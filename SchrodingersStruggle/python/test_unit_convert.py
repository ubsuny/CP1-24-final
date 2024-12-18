"""
This module provides unit testing for unit_conversion.py

Tests include:
- Converting 2 feet to meters.
- Converting 5 yards to meters.
- Converting 2 meters to yards.
- Invalid input form.
"""
import pytest
import unit_convert as uc

class TestUnitConvert:
    """
    Establishing class for testing unit_convert.py
    """
    def test_foot_to_meter(self):
        """
        Test conversion from 2 feet to meters.
        """
        feet = 2
        meters = uc.ft_m(feet)
        expected_meters = 0.6096
        assert meters == pytest.approx(expected_meters, rel=0.0001)

    def test_yards_to_meter(self):
        """
        Test conversion from 5 yards to meters.
        """
        yards = 5
        meters = uc.yrd_m(yards)
        expected_meters = 4.572
        assert meters == pytest.approx(expected_meters, rel=0.0001)

    def test_meters_to_yards(self):
        """
        Test conversion from 2 meters to yards.
        """
        meters = 2
        meters = uc.m_yrd(meters)
        expected_yards = 2.18723
        assert meters == pytest.approx(expected_yards, rel=0.0001)

    def test_invalid_input(self):
        """
        Test how code handles invalid input:
        """
        feet = 'not a number'
        with pytest.raises(TypeError):
            uc.ft_m(feet)
