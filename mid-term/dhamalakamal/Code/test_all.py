"""
Unit Tests Module

This module provides a function `convert_to_meters` that converts various units
of length to meters. It supports common units such as kilometers, centimeters,
millimeters, inches, feet, yards, and miles.

The module also contains unit tests using `pytest` to verify the accuracy of
the conversion function. The tests cover a range of conversions and check for
correct error handling when an invalid unit is provided.

Functions:
    convert_to_meters(value, unit): Converts a given length from the specified
    unit to meters.

    import_csv

Test Cases:
    - Test conversion from yards to meters.
    - Test conversion from miles to meters.
    - Test that an invalid unit raises a ValueError.

"""
from io import StringIO
import os
import pytest
import pandas as pd
import numpy as np
from convert_to_meter import feet_to_meters, yards_to_meters
from distance_calculator import load_gps_data, haversine, calculate_distances
from acceleration_calculate import calculate_direction, calculate_directions_from_csv
from unix_time import extract_datetime_from_md, read_md_file_and_convert_to_unix

# Define the path to the existing Markdown file in the Data folder
DATA_FOLDER = os.path.join(os.path.dirname(__file__), '..', 'Data')  # Adjust the path as necessary
MD_FILE_PATH = os.path.join(DATA_FOLDER, 'KD_acceleration.md')  


# Testing feet to meter conversion
def test_feet():
    """
    Test conversion from feet to meters.
    1 foot should equal exactly 0.3048 meters.
    """
    input_val = 1
    expected = 0.3048

    output_val = feet_to_meters(input_val)
    assert output_val == expected, "The output value is not same: " + str(output_val)

# Testing yards to meter conversion
def test_yards():
    """
    Test conversion from yards to meters.
    1 yard should equal exactly 0.9144 meters.
    """
    input_val = 1
    expected = 0.9144

    output_val = yards_to_meters(input_val)
    assert output_val == expected, "The output value is not same: " + str(output_val)

def test_load_gps_data():
    """
    Test loading of gps data from csv file
    """
    # Simulate a CSV file with StringIO
    csv_data = """Latitude (°),Longitude (°)
    34.052235,-118.243683
    36.169941,-115.139832
    """
    # Use StringIO to mimic a file-like object for pandas
    test_file = StringIO(csv_data)

    # Call the function to load data
    gps_data = load_gps_data(test_file)

    # Check that the function returns a DataFrame with the expected columns
    assert isinstance(gps_data, pd.DataFrame)
    assert list(gps_data.columns) == ["Latitude (°)", "Longitude (°)"]
    assert len(gps_data) == 2

def test_haversine():
    """
    Testing haversine function 
    """
    # Sample coordinates for Paris and Rome
    lat1, lon1 = 48.8566, 2.3522  # Paris
    lat2, lon2 = 41.9028, 12.4964  # Rome
    # km to meter factor
    meter_fact = 1000

    result = haversine(lat1, lon1, lat2, lon2)
    expected = 1105 * meter_fact  # Approximate distance in meters between Paris and Rome
    assert pytest.approx(result, 0.1) == expected

    # Test with the same points (should return 0)
    result_same = haversine(lat1, lon1, lat1, lon1)
    assert result_same == 0

def test_calculate_distances(monkeypatch):
    """
    Test Calculate distances method
    """
    # Create sample GPS data
    data = {
        "Latitude (°)": [34.052235, 36.169941, 40.712776],
        "Longitude (°)": [-118.243683, -115.139832, -74.005974]
    }
    gps_data = pd.DataFrame(data)

    # Define a mock haversine function that returns predefined distances
    def mock_haversine(lat1, lon1, lat2, lon2):
        if (lat1, lon1, lat2, lon2) == (34.052235, -118.243683, 36.169941, -115.139832):
            return 367000  # Distance between LA and Vegas (approx.)
        elif (lat1, lon1, lat2, lon2) == (36.169941, -115.139832, 40.712776, -74.005974):
            return 393000  # Distance between Vegas and NY (approx.)

    # Use monkeypatch to replace the haversine function in the gps_module
    monkeypatch.setattr("distance_calculator.haversine", mock_haversine)

    # Expected distances based on the mocked haversine function
    expected_distances = [367000, 393000]

    # Call the calculate_distances function
    distances = calculate_distances(gps_data)

    # Check that the function returns the correct list of distances
    assert distances == expected_distances

def test_calculate_direction():
    """
    Test the calculate_direction function with various acceleration inputs.
    Each test case verifies if the function correctly computes azimuth and elevation angles
    for different configurations of acceleration data.
    """
    # Test case 1: Acceleration only in the X direction
    acc_x, acc_y, acc_z = 1.0, 0.0, 0.0
    azimuth, elevation = calculate_direction(acc_x, acc_y, acc_z)
    assert np.isclose(azimuth, 0.0, atol=1e-5)
    assert np.isclose(elevation, 0.0, atol=1e-5)

    # Test case 2: Acceleration only in the Y direction
    acc_x, acc_y, acc_z = 0.0, 1.0, 0.0
    azimuth, elevation = calculate_direction(acc_x, acc_y, acc_z)
    assert np.isclose(azimuth, 90.0, atol=1e-5)
    assert np.isclose(elevation, 0.0, atol=1e-5)

    # Test case 3: Acceleration in both X and Y directions (45 degrees in XY plane)
    acc_x, acc_y, acc_z = 1.0, 1.0, 0.0
    azimuth, elevation = calculate_direction(acc_x, acc_y, acc_z)
    assert np.isclose(azimuth, 45.0, atol=1e-5)
    assert np.isclose(elevation, 0.0, atol=1e-5)

    # Test case 4: Acceleration in Z direction, representing elevation
    acc_x, acc_y, acc_z = 0.0, 0.0, 1.0
    azimuth, elevation = calculate_direction(acc_x, acc_y, acc_z)
    assert np.isclose(azimuth, 0.0, atol=1e-5)
    assert np.isclose(elevation, 90.0, atol=1e-5)

    # Test case 5: Acceleration in all directions (arbitrary values)
    acc_x, acc_y, acc_z = 1.0, 1.0, 1.0
    azimuth, elevation = calculate_direction(acc_x, acc_y, acc_z)
    assert np.isclose(azimuth, 45.0, atol=1e-5)  # Expected azimuth in XY plane
    assert np.isclose(elevation, 35.264, atol=1e-3)  # Expected elevation angle

def test_calculate_directions_from_csv(tmp_path):
    """
    Test the calculate_directions_from_csv function by creating a mock CSV file 
    with known acceleration data and verifying that the returned directions 
    (azimuth and elevation angles) are calculated correctly.
    
    Args:
        tmp_path: A pytest fixture providing a temporary directory for file creation.
    """

    # Define test data with known azimuth and elevation results
    test_data = {
        'Linear Acceleration x (m/s^2)': [1.0, 0.0, 1.0, 0.0],
        'Linear Acceleration y (m/s^2)': [0.0, 1.0, 1.0, 0.0],
        'Linear Acceleration z (m/s^2)': [0.0, 0.0, 0.0, 1.0]
    }
    expected_directions = [
        (0.0, 0.0),  # Only X-axis acceleration
        (90.0, 0.0),  # Only Y-axis acceleration
        (45.0, 0.0),  # Equal X and Y acceleration
        (0.0, 90.0)  # Only Z-axis acceleration
    ]

    # Create a temporary CSV file with test data
    csv_file = tmp_path / "test_acceleration_data.csv"
    pd.DataFrame(test_data).to_csv(csv_file, index=False)

    # Run the function with the temporary file
    directions = calculate_directions_from_csv(csv_file)

    # Check that the directions returned match expected results
    for i, (azimuth, elevation) in enumerate(directions):
        expected_azimuth, expected_elevation = expected_directions[i]
        assert np.isclose(azimuth, expected_azimuth, atol=1e-5), f"Azimuth mismatch at index {i}"
        assert np.isclose(elevation, expected_elevation, atol=1e-5), f"Elevation mismatch at index {i}"

def test_extract_datetime_from_md():
    """
    Test the extract_datetime_from_md function to ensure it correctly extracts
    and converts date strings from Markdown content into Unix time.
    """

    # Test case: Valid date and time
    with open(MD_FILE_PATH, 'r', encoding='utf-8') as md_file:
        content = md_file.read()
        expected_unix_time = 1729521592
        assert extract_datetime_from_md(
            content,
            date_pattern=r'\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}'
            ) == expected_unix_time

    # Test case: Invalid date format
    invalid_content = "This content has no valid date."
    assert extract_datetime_from_md(invalid_content) is None

def test_read_md_file_and_convert_to_unix():
    """
    Test the read_md_file_and_convert_to_unix function to ensure it correctly
    reads a markdown file and converts date strings to Unix time.
    """
    # Read the valid file and get the Unix time
    expected_unix_time = 1729521592  # Updated expected Unix time
    assert read_md_file_and_convert_to_unix(
        MD_FILE_PATH,
        date_pattern=r'\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}'
        ) == expected_unix_time

    # Test case: Non-existent file handling
    with pytest.raises(FileNotFoundError):
        read_md_file_and_convert_to_unix('non_existent_file.md')
