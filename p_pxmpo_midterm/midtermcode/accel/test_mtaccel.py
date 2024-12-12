"""
Unit tests for functions in the `mtaccelfuncts` module.

This module includes tests for:
1. `read_acceleration`: Reads acceleration values from a CSV file.
2. `check_direction`: Determines the direction of movement based on acceleration values.
   
The tests use pytest fixtures to create temporary CSV files and provide sample data
for testing both functions.
"""

import sys
import csv
import tempfile
import pytest
sys.path.append('/workspaces/FORKCP1-24-final/p_pxmpo_midterm/midtermcode/accel')

from mtaccelfuncts import read_acceleration, check_direction

@pytest.fixture
def tmp_sample_csv_file():
    """
    Fixture to create a temporary CSV file for testing `read_acceleration`.
    
    Returns:
        str: The path to the temporary CSV file containing time, X, Y, and Z data.
    """
    data = [
        ["Time", "X", "Y", "Z"],
        ["0", "0", "0.5", "0"],
        ["1", "0", "2.0", "0"],
        ["2", "0", "-1.5", "0"],
        ["3", "0", "0.2", "0"]
    ]

    # Using tempfile to create a temporary CSV file
    with tempfile.NamedTemporaryFile(mode='w', delete=False, newline='') as temp_file:
        writer = csv.writer(temp_file)
        writer.writerows(data)
        temp_file_name = temp_file.name

    yield temp_file_name

@pytest.fixture
def sample_y_values():
    """
    Fixture to provide sample Y-values for testing `check_direction`.
    
    Returns:
        list: A list of Y-values.
    """
    return [0.5, 2.0, -1.5, 0.2]

def test_read_acceleration(tmp_sample_csv_file):
    """
    Test for `read_acceleration` function.
    
    Ensures that the function correctly reads Y-values from the third column of the CSV file.
    
    Args:
        sample_csv_file (str): The path to the temporary CSV file provided by the fixture.
    """
    y_values = read_acceleration(tmp_sample_csv_file)
    assert y_values == [0.5, 2.0, -1.5, 0.2]

def test_check_direction_up():
    """
    Test for `check_direction` when the direction is "up".
    
    Verifies that the function returns "up" when the Y-values indicate upward movement.
    """
    y_values = [0.5, 2.0, -1.5]
    result = check_direction(y_values)
    assert result == "up"

def test_check_direction_down():
    """
    Test for `check_direction` when the direction is "down".
    
    Verifies that the function returns "down" when there's an acceleration below -1,
    indicating downward movement.
    """
    y_values = [0.5, -2.0, 1.5]
    result = check_direction(y_values)
    assert result == "down"

def test_check_direction_no_significant_trend():
    """
    Test for `check_direction` when there's no significant trend.
    
    Ensures that the function returns "No significant trend" when all values are within [-1, 1].
    """
    y_values = [0.5, 0.3, -0.8]
    result = check_direction(y_values)
    assert result == "No significant trend"

# Run the tests if this file is executed directly
if __name__ == "__main__":
    pytest.main()
