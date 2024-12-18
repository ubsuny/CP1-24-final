"""
This module contains unit tests for the `unix_time_converter` function which extracts date and time from metadata files and converts them into Unix time.

Functions:
    test_unix_time_converter_valid_metadata(): Tests the `unix_time_converter` function with mock valid metadata content.
    test_unix_time_converter_missing_fields(): Tests `unix_time_converter` when metadata files are missing 'Start' or 'End' fields.
    test_unix_time_converter_invalid_path(): Tests `unix_time_converter` with invalid file paths.
Run the tests in the terminal: pytest unix_time_converter_tests.py
"""

import pytest
from Unix_Converter import unix_time_converter

@pytest.fixture
def valid_metadata(tmp_path):
    """
    Fixture to create mock metadata files with valid content.
    """
    paths = []
    filenames = ["HA007_Elevator_Down.md", "HA007_Elevator_Up.md"]
    for filename in filenames:
        filepath = tmp_path / filename
        filepath.write_text(
            """
            | Start | 2024-10-25 | 12:00:00.000000 |
            | End | 2024-10-25 | 12:30:00.000000 |
            """
        )
        paths.append(str(filepath))
    return paths

@pytest.fixture
def missing_fields_metadata(tmp_path):
    """
    Fixture to create mock metadata files with missing 'Start' or 'End' fields.
    """
    filepath = tmp_path / "HA007_Missing_Field.md"
    filepath.write_text(
        """
        | Start | 2024-10-25 | 12:00:00.000000 |
        """
    )
    return [str(filepath)]

@pytest.fixture
def invalid_path():
    """
    Fixture to provide an invalid path.
    """
    return ["invalid/path/to/metadata.md"]

def test_unix_time_converter_valid_metadata(valid_metadata):
    """
    Tests the `unix_time_converter` function with mock valid metadata content.
    
    It checks whether the function correctly extracts Unix time for 'Start' and 'End' fields.
    """
    result = unix_time_converter(valid_metadata)
    assert len(result) == 2, f"Expected 2 metadata files to be processed, but got {len(result)}"
    for path, times in result.items():
        assert 'Start' in times, f"Missing 'Start' field in the result for {path}"
        assert 'End' in times, f"Missing 'End' field in the result for {path}"
        assert isinstance(times['Start'], int), f"Expected Unix time (int) for 'Start', but got {type(times['Start'])}"
        assert isinstance(times['End'], int), f"Expected Unix time (int) for 'End', but got {type(times['End'])}"

def test_unix_time_converter_missing_fields(missing_fields_metadata):
    """
    Tests the `unix_time_converter` function with mock metadata files missing 'End' field.
    
    It checks whether the function processes the available fields correctly.
    """
    result = unix_time_converter(missing_fields_metadata)
    path = missing_fields_metadata[0]
    assert 'Start' in result[path], f"Missing 'Start' field in the result for {path}"
    assert 'End' not in result[path], f"Unexpected 'End' field found in the result for {path}"

def test_unix_time_converter_invalid_path(invalid_path):
    """
    Tests the `unix_time_converter` function with invalid file paths.
    
    It verifies that the function handles invalid paths gracefully by raising a FileNotFoundError.
    """
    with pytest.raises(FileNotFoundError):
        unix_time_converter(invalid_path)
