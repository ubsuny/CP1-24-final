"""
This module is to test for the functions created in the final.py file
"""

import pytest
import final

def test_fahrenheit_to_kelvin():
    """
    This functions tests for multiple cases for temperature conversion
    """
    result = final.fahrenheit_to_kelvin(-459.67)
    expected = 0
    assert result == expected, f"Test failed: Expected {expected}, but got {result}."
    result = final.fahrenheit_to_kelvin(32)
    expected = 273.15
    assert result == expected, f"Test failed: Expected {expected}, but got {result}."
    result = final.fahrenheit_to_kelvin(0)
    expected = 255.37222222222222
    assert result == expected, f"Test failed: Expected {expected}, but got {result}."