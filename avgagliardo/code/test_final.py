
"""
test_final.py

Tests the final.py module's functionality.
"""
import pytest

from final import convert_f_to_k

def test_convert_f_to_k():
    """
    Test the convert_f_to_k function with multiple cases.

    Ensures that the function correctly converts temperatures
    from Fahrenheit to Kelvin using known inputs and expected outputs.

    Test Cases:
        - Freezing point of water (32°F -> 273.15K)
        - Boiling point of water (212°F -> 373.15K)
        - Absolute zero (-459.67°F -> 0K)
        - Arbitrary values for additional checks
    """
    # Test Cases: Input Fahrenheit, Expected Kelvin
    test_cases = [
        (32, 273.15),   # Freezing point of water
        (212, 373.15),  # Boiling point of water
        (-459.67, 0),   # Absolute zero
        (0, 255.372),   # Arbitrary test case
        (100, 310.928)  # Another test case
    ]

    # Tolerance for floating-point comparisons. Might be necessary.
    tolerance = 1e-3

    # Test each case.
    for fahrenheit, expected_kelvin in test_cases:
        # Invoke the function
        result = convert_f_to_k(fahrenheit)
        # Assert truth.
        assert abs(result - expected_kelvin) < tolerance, f"Failed for input: {fahrenheit}F"

if __name__ == "__main__":
    pytest.main()
