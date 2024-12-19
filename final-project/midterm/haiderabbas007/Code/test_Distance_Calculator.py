"""
This module contains unit tests for the functions in the Distance_Calculator.py module.
It tests both normal cases (e.g., valid latitude and longitude points) and edge cases (e.g., identical points to verify zero distance).

Functions:
    test_calculate_distance(): Tests the `calculate_distance` function with regular latitude and longitude data.
    test_zero_distance(): Tests the `calculate_distance` function with identical coordinates to verify zero distance output.
    test_calculate_radius(): Tests the `calculate_radius` function to verify correct radius calculations.
    test_invalid_coordinates(): Tests the `calculate_distance` function with invalid coordinates to verify error handling.

To run the tests in terminal: pytest distance_calculator_tests.py
"""

import pytest

from Distance_Calculator import calculate_distance, calculate_radius

# Random GPS coordinates obtained from https://gps-coordinates.org/.

@pytest.fixture
def mock_coordinates():
    """
    Fixture to create mock coordinate data for testing.
    """
    return {
        'point_a': (40.748817, -73.985428),  # New York, Empire State Building
        'point_b': (34.052235, -118.243683)  # Los Angeles, City Hall
    }

@pytest.fixture
def identical_coordinates():
    """
    Fixture to create mock data with identical coordinates for testing zero distance.
    """
    return {
        'point_a': (51.507351, -0.127758),  # London, UK
        'point_b': (51.507351, -0.127758)   # Same point
    }

@pytest.fixture
def invalid_coordinates():
    """
    Fixture to create mock data with invalid coordinates for testing error handling.
    """
    return {
        'point_a': (100.0, -200.0),  # Invalid latitude and longitude
        'point_b': (34.052235, -118.243683)  # Valid point
    }

def test_calculate_distance(mock_coordinates):
    """
    Tests the `calculate_distance` function with mock coordinate data.

    It checks whether the calculated distance between two points matches the expected value.
    """
    point_a = mock_coordinates['point_a']
    point_b = mock_coordinates['point_b']
    result = calculate_distance(point_a, point_b)

    # The expected distance between New York and Los Angeles (approximately 3940 km)
    expected_result = 3940  # This is an approximate value in kilometers obtained through Google Maps

    # Assert that the result is within an acceptable range of the expected output
    assert abs(result - expected_result) < 50, f"Expected approximately {expected_result} km, but got {result} km"

def test_zero_distance(identical_coordinates):
    """
    Tests the `calculate_distance` function with identical coordinates.

    It verifies that the function correctly returns zero distance for identical points.
    """
    point_a = identical_coordinates['point_a']
    point_b = identical_coordinates['point_b']
    result = calculate_distance(point_a, point_b)

    # Expected result for identical points is 0 km
    expected_result = 0.0

    # Assert that the results match the expected output
    assert result == expected_result, f"Expected {expected_result} km, but got {result} km"

def test_calculate_radius():
    """
    Tests the `calculate_radius` function to verify correct radius calculations.

    It checks if the function returns the correct radius for the Earth.
    """

    # Earth's average radius in km (https://nssdc.gsfc.nasa.gov/planetary/factsheet/earthfact.html)
    expected_radius = 6371
    result = calculate_radius()

    # Assert that the calculated radius matches the expected value
    assert result == expected_radius, f"Expected radius {expected_radius} km, but got {result} km"

def test_invalid_coordinates(invalid_coordinates):
    """
    Tests the `calculate_distance` function with invalid coordinates.

    It verifies that the function raises a ValueError for invalid latitude and longitude values.
    """
    point_a = invalid_coordinates['point_a']
    point_b = invalid_coordinates['point_b']

    with pytest.raises(ValueError):
        calculate_distance(point_a, point_b)
