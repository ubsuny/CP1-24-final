"""Test for direction_of_motion.py module"""
import numpy as np
import pytest
from direction_of_motion_elevator import (
    read_acceleration_data,
    calculate_position,
    process_acceleration_data
)

# Sample data for testing
SAMPLE_DATA = """time,x_acc,y_acc,z_acc
0.0,0.0,0.0,0.0
1.0,1.0,0.0,0.0
2.0,0.0,1.0,0.0
3.0,0.0,0.0,1.0
4.0,1.0,1.0,1.0
"""

@pytest.fixture
def sample_csv_file(tmpdir):

    """Create a temporary CSV file for testing."""

    csv_file = tmpdir.join("sample_data.csv")
    csv_file.write(SAMPLE_DATA)
    return str(csv_file)

def test_calculate_position(sample_csv_file):

    """Test calculating position based on acceleration data."""

    data = read_acceleration_data(sample_csv_file)
    x_position, y_position, z_position = calculate_position(data)

    # Print the calculated positions for debugging

    print("Calculated X Position:", x_position)
    print("Calculated Y Position:", y_position)
    print("Calculated Z Position:", z_position)

    # Check the shape of the position arrays

    assert x_position.shape == (5,)
    assert y_position.shape == (5,)
    assert z_position.shape == (5,)

    # Define expected values based on the integration of SAMPLE_DATA

    expected_x_position = np.array([0.0, 1., 2., 3., 5.])  # Adjusted based on calculations
    expected_y_position = np.array([0.0, 0.0, 1., 2., 4.])  # Adjusted based on calculations
    expected_z_position = np.array([0.0, 0.0, 0.0, 1.0, 3.0])  # Adjusted based on calculations

    # Print expected values for comparison

    print("Expected X Position:", expected_x_position)
    print("Expected Y Position:", expected_y_position)
    print("Expected Z Position:", expected_z_position)

    assert np.allclose(x_position,
                       expected_x_position,
                       atol=1e-2), f"X Position mismatch: {x_position} != {expected_x_position}"
    assert np.allclose(y_position,
                       expected_y_position,
                       atol=1e-2), f"Y Position mismatch: {y_position} != {expected_y_position}"
    assert np.allclose(z_position,
                       expected_z_position,
                       atol=1e-2), f"Z Position mismatch: {z_position} != {expected_z_position}"

def test_process_acceleration_data(sample_csv_file, tmpdir):

    """Test the complete processing of acceleration data."""

    gen_pic = tmpdir.join("final_plot.png")
    x_position, y_position, z_position = process_acceleration_data(sample_csv_file, str(gen_pic))

    # Check that positions are returned correctly

    expected_x_position = np.array([0.0, 1., 2., 3., 5.])  # Adjusted based on calculations
    expected_y_position = np.array([0.0, 0.0, 1., 2., 4.])  # Adjusted based on calculations
    expected_z_position = np.array([0.0, 0.0, 0.0, 1., 3.])  # Adjusted based on calculations

    assert np.allclose(x_position,
                       expected_x_position,
                       atol=1e-2), f"X Position mismatch: {x_position} != {expected_x_position}"
    assert np.allclose(y_position,
                       expected_y_position,
                       atol=1e-2), f"Y Position mismatch: {y_position} != {expected_y_position}"
    assert np.allclose(z_position,
                       expected_z_position,
                       atol=1e-2), f"Z Position mismatch: {z_position} != {expected_z_position}"

    # Check if the generated image file exists by using the path returned by the fixture

    assert tmpdir.join("final_plot.png").exists(), "Generated picture file does not exist."
