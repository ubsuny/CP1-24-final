
"""
test_final.py

Tests the final.py module's functionality.
"""
import tempfile
import os
import pytest

# Task 1 tests
from final import convert_f_to_k
# Task 2 tests
from final import parse_markdown
# Task 3 tests
from final import filter_markdown_files
# Task 4 tests
from final import sort_filenames

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

def test_filter_markdown_files():
    """
    Test the filter_markdown_files function.
    """
    # create a temp directory with test markdown files
    with tempfile.TemporaryDirectory() as temp_dir:
        filenames = [
            "AVG001_gps_sine_walk_0.md",
            "AVG002_gps_sine_walk_1.md",
            "README.md",
            "NOTES_summary.md",
        ]

        # Create the files in the temp directory
        for filename in filenames:
            with open(os.path.join(temp_dir, filename), 'w', encoding="utf-8") as temp_file:
                temp_file.close()
            # open(os.path.join(temp_dir, filename), 'w').close()

        # Test filtering with 'sine_walk' string
        result = filter_markdown_files(temp_dir, "sine_walk")
        assert sorted(result) == ["AVG001_gps_sine_walk_0.md", "AVG002_gps_sine_walk_1.md"]

        # Test filtering with 'NOTES'
        result = filter_markdown_files(temp_dir, "NOTES")
        assert result == ["NOTES_summary.md"]

        # Test filtering with a non-matching string
        result = filter_markdown_files(temp_dir, "nonexistent")
        assert result == []

def test_sort_filenames_with_non_numeric_suffix():
    """
    Test that sort_filenames handles filenames with non-numeric suffixes correctly.
    """
    filenames = [
        "file_10.md",
        "file_2.md",
        "file_final.md",
        "file_1.md",
    ]
    expected_output = [
        "file_1.md",
        "file_2.md",
        "file_10.md",
        "file_final.md",  # Non-numeric filename appears last
    ]
    assert sort_filenames(filenames) == expected_output

def test_sort_filenames():
    """
    Test that sort_filenames sorts filenames based on their trailing numeric values.
    """
    # Sample unsorted filenames
    filenames = [
        "file_10.md",
        "file_2.md",
        "file_1.md",
        "file_20.md",
    ]
    # Expected sorted order
    expected_output = [
        "file_1.md",
        "file_2.md",
        "file_10.md",
        "file_20.md",
    ]
    # Test sorting
    assert sort_filenames(filenames) == expected_output

    # Edge case: empty list
    assert sort_filenames([]) == []

    # Edge case: single file
    assert sort_filenames(["file_42.md"]) == ["file_42.md"]

# Pytest main hook
if __name__ == "__main__":
    pytest.main()
