"""
Unit tests for Direction_Calculator.py

This module contains unit tests for the `calculate_direction_of_motion` function in the Direction_Calculator.py module.
It tests both normal cases and edge cases (e.g., zero magnitude). The tests use mock data to ensure that the function returns correct unit vectors based on the provided data.

Functions:
    test_calculate_direction_of_motion(): Tests the function with regular mock acceleration data.
    test_zero_magnitude(): Tests the function with zero acceleration data to verify the handling of zero magnitudes.

To run these tests in terminal: pytest direction_calculator_tests.py

"""

import pytest
import pandas as pd

from Direction_Calculator import calculate_direction_of_motion

@pytest.fixture
def mock_data():
    """
    Fixture to create mock acceleration data for testing.
    """
    return pd.DataFrame({
        'Linear Acceleration x (m/s^2)': [1, 0, 0],
        'Linear Acceleration y (m/s^2)': [0, 1, 0],
        'Linear Acceleration z (m/s^2)': [0, 0, 1]
    })

@pytest.fixture
def zero_acceleration_data():
    """
    Fixture to create mock data with zero acceleration for testing.
    """
    return pd.DataFrame({
        'Linear Acceleration x (m/s^2)': [0, 0, 0],
        'Linear Acceleration y (m/s^2)': [0, 0, 0],
        'Linear Acceleration z (m/s^2)': [0, 0, 0]
    })

def test_calculate_direction_of_motion(mock_data, tmpdir):
    """
    Tests the `calculate_direction_of_motion` function with mock acceleration data.

    It checks whether the calculated unit vectors match the expected unit vectors for given (x, y, z) accelerations.
    """
    # Save mock data to a CSV file
    mock_file_path = tmpdir.join('mock_acceleration.csv')
    mock_data.to_csv(mock_file_path, index=False)

    # Test the function with the mock data
    result = calculate_direction_of_motion([str(mock_file_path)])

    # Expected unit vectors for the mock data
    expected_result = {
        str(mock_file_path): [
            (1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            (0.0, 0.0, 1.0)
        ]
    }

    # Assert that the results match the expected output
    assert result == expected_result, f"Expected {expected_result}, but got {result}"

def test_zero_magnitude(zero_acceleration_data, tmpdir):
    """
    Tests the `calculate_direction_of_motion` function with zero acceleration.

    It verifies that the function correctly handles zero magnitudes by returning (0, 0, 0) for unit vectors.
    """
    # Save mock data with zero acceleration to a CSV file
    mock_file_path = tmpdir.join('mock_zero_acceleration.csv')
    zero_acceleration_data.to_csv(mock_file_path, index=False)

    # Test the function with the mock data
    result = calculate_direction_of_motion([str(mock_file_path)])

    # Expected unit vectors for the mock data (all zeros)
    expected_result = {
        str(mock_file_path): [
            (0.0, 0.0, 0.0),
            (0.0, 0.0, 0.0),
            (0.0, 0.0, 0.0)
        ]
    }

    # Assert that the results match the expected output
    assert result == expected_result, f"Expected {expected_result}, but got {result}"
