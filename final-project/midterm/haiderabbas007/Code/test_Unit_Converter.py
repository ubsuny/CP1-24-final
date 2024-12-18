"""
This module contains unit tests for the `unit_converter` function, which converts lengths in feet or yards to meters.
It tests the valid conversion of lengths, as well as edge cases, including invalid input values and unsupported units.

Functions:
    test_valid_conversion(): Tests the `unit_converter` function with valid inputs.
    test_negative_value(): Tests the `unit_converter` function with negative values to verify error handling.
    test_invalid_units(): Tests the `unit_converter` function with unsupported units to verify error handling.
Run the tests in the terminal: pytest unit_converter_tests.py
"""

import pytest
from Unit_Converter import unit_converter

@pytest.fixture
def valid_conversion_data():
    """
    Fixture to create mock data for valid unit conversion scenarios.
    """
    return [
        (10, 'ft', 3.048),
        (5, 'yd', 4.572),
        (15, 'Ft', 4.572),  # Case insensitive test
        (2, 'Yd', 1.8288)   # Case insensitive test
    ]

@pytest.fixture
def negative_value_data():
    """
    Fixture to create mock data with negative values for testing error handling.
    """
    return [
        (-10, 'ft'),
        (-5, 'yd')
    ]

@pytest.fixture
def invalid_units_data():
    """
    Fixture to create mock data with unsupported units for testing error handling.
    """
    return [
        (10, 'm'),   # Unsupported unit
        (5, 'km'),   # Unsupported unit
        (20, 'inch') # Unsupported unit
    ]

def test_valid_conversion(valid_conversion_data):
    """
    Tests the `unit_converter` function with valid inputs.
    
    It checks whether the function correctly converts feet and yards to meters.
    """
    for val, unit, expected in valid_conversion_data:
        result = unit_converter(val, unit)
        assert abs(result - expected) < 0.001, f"Expected {expected}, but got {result}"

def test_negative_value(negative_value_data):
    """
    Tests the `unit_converter` function with negative values.
    
    It verifies that the function raises a ValueError when provided with negative lengths.
    """
    for val, unit in negative_value_data:
        with pytest.raises(ValueError, match="Only non-negative lengths please."):
            unit_converter(val, unit)

def test_invalid_units(invalid_units_data):
    """
    Tests the `unit_converter` function with unsupported units.
    
    It verifies that the function raises a ValueError when provided with units other than 'ft' or 'yd'.
    """
    for val, unit in invalid_units_data:
        with pytest.raises(ValueError, match="Only 'ft' or 'yd' please."):
            unit_converter(val, unit)
