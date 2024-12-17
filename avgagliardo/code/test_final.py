
"""
test_final.py

Tests the final.py module's functionality.
"""
import tempfile
import os
import pytest

from final import convert_f_to_k
from final import parse_markdown

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

def create_temp_markdown_file(content):
    """
    Helper function to create a temporary markdown file.

    Parameters:
        content (str): The content to write into the file.

    Returns:
        str: The file path of the temporary file.
    """
    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".md") as temp_file:
        temp_file.write(content)
        return temp_file.name

def test_parse_markdown():
    """
    Test the parse_markdown function for extracting a specific field's value.
    """

    # Define test cases: content -> field_name -> expected result
    test_cases = [
        # Valid cases
        ("##Temperature\n33 F", "Temperature", 33.0),  # Basic valid case
        ("##Temperature\n-10 F", "Temperature", -10.0),  # Negative temperature
        ("##Temperature\n45.5 F", "Temperature", 45.5),  # Decimal temperature
        ("##Humidity\n75 F", "Humidity", 75.0),  # Extracting Humidity field
        ("##Pressure\n1013 F", "Pressure", 1013.0),  # Extracting Pressure field

        # Invalid cases
        ("##Temperature\nInvalid F", "Temperature", None),  # Non-numeric temperature
        ("##Temperature\n33", "Temperature", None),  # Missing 'F'
        ("##Temperature", "Temperature", None),  # No value after header
        ("No temperature here", "Temperature", None),  # Missing header
        ("##temperature\n33 F", "Temperature", None),  # Case-sensitive check
    ]

    for content, field_name, expected_result in test_cases:
        # Create a temporary markdown file
        temp_file_path = create_temp_markdown_file(content)

        try:
            # Run the function
            result = parse_markdown(temp_file_path, field_name)

            # Assert the result matches the expectation
            assert result == expected_result, (
                f"Failed for field: '{field_name}', content:\n{content}\n"
                f"Expected {expected_result}, got {result}."
            )
        finally:
            # Clean up: Delete the temporary file
            os.remove(temp_file_path)
# Pytest main hook
if __name__ == "__main__":
    pytest.main()
