"""
This module is to test for the functions created in the final.py file
"""

import tempfile
import pandas as pd
import numpy as np
import pytest
import final

# Testing Temperature Conversion
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

# Testing Temperature Reading
def test_read_temp():
    """
    Test the read_temp function with a valid file and an invalid format.
    """
    # Test case 1: Valid file
    valid_content = "Temperature (F): 28\nDate: 12-5-2024\nTime: 15:40:48.266\n"
    with tempfile.NamedTemporaryFile(mode="w+", delete=True) as temp_file:
        temp_file.write(valid_content)
        temp_file.seek(0)  # Reset file pointer to the beginning
        result = final.read_temp(temp_file.name)
        expected = 28.0
        assert result == expected, f"Test failed: Expected {expected}, but got {result}."
    # Test case 2: Invalid file format (missing temperature line)
    invalid_content = "Date: 12-5-2024\nTime: 15:40:48.266\n"
    with tempfile.NamedTemporaryFile(mode="w+", delete=True) as temp_file:
        temp_file.write(invalid_content)
        temp_file.seek(0)  # Reset file pointer to the beginning
        with pytest.raises(ValueError, match="could not convert string to float"):
            final.read_temp(temp_file.name)
    # Test case 3: Empty file
    with tempfile.NamedTemporaryFile(mode="w+", delete=True) as temp_file:
        with pytest.raises(IndexError):
            final.read_temp(temp_file.name)
    # Test case 4: Actual file
    fpath = "/workspaces/CP1-24-final/HeshamElsaman/Data/GPSSine/gs01_gps_sin/gs01_gps_sin.md"
    temp = final.read_temp(fpath)
    expected = 28
    assert temp == expected, f"Test failed: Expected {expected}, but got {temp}."

#Testing Listing The Files
def test_list_markdown_files(tmp_path):
    """
    Test the list_markdown_files function with a temporary directory.
    """
    # Set up a temporary directory with test files
    markdown_files = ["experiment1.md", "experiment2.md", "notes.md"]
    other_files = ["experiment1.txt", "readme.md", "experiment3.log"]
    for file_name in markdown_files + other_files:
        file_path = tmp_path / file_name
        file_path.touch()  # Create the file
    # Test case 1: Matching files
    result = final.list_markdown_files(tmp_path, "experiment")
    expected = ["experiment1.md", "experiment2.md"]
    assert sorted(result) == sorted(expected), f"Test failed: Expected {expected}, but got {result}"
    # Test case 2: No matching files
    result = final.list_markdown_files(tmp_path, "nonexistent")
    expected = []
    assert result == expected, f"Test failed: Expected {expected}, but got {result}."
    # Test case 3: Empty directory
    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()
    result = final.list_markdown_files(empty_dir, "experiment")
    expected = []
    assert result == expected, f"Test failed: Expected {expected}, but got {result}."
    #Test case 4: Actual directory
    directory = "/workspaces/CP1-24-final/HeshamElsaman/Data/GPSSine/gs01_gps_sin"
    expected = ["gs01_gps_sin.md"]
    result = final.list_markdown_files(directory, "gps_sin")
    assert result == expected, f"Test failed: Expected {expected}, but got {result}."

# Testing the Non-linear Fitting
def test_fit_sinusoidal():
    """
    Testing the Non-linear Fitting function
    """
    # Test case: simple sine wave
    x_data = [0, 1, 2, 3, 4, 5, 6]
    y_data = [0, 0.8415, 0.9093, 0.1411, -0.7568, -0.9589, -0.2794]
    initial_guess = (1, 1, 0)
    steps = 64
    result = final.fit_sinusoidal(x_data, y_data, steps, initial_guess)
    expected = (1, 1, 0)
    errmsg = f"Test failed: Expected {expected}, got {result}"
    assert pytest.approx(result, rel=1e-512) == expected, errmsg

# Testing FFT/IFFT
def test_check_equidistant():
    """
    Test the check_equidistant function for equidistant and non-equidistant data.
    """
    # Equidistant data
    df = pd.DataFrame({'x': np.linspace(0, 10, 100)})
    assert final.check_equidistant(df, 'x'), "Test failed: Data wasn't recognized as equidistant."

    # Non-equidistant data
    df = pd.DataFrame({'x': [0, 1, 2, 4, 7]})
    assert not final.check_equidistant(df, 'x'), "Test failed: It's not nonequidistant data."

def test_fft():
    """
    Test the fft function for correct FFT computation with equidistant data.
    """
    # Equidistant data
    df = pd.DataFrame({'x': np.linspace(0, 10, 100), 'y': np.sin(np.linspace(0, 10, 100))})
    result = final.fft(df, 'x', 'y')
    assert isinstance(result, pd.Series), "Test failed: FFT did not return a pandas Series."
    assert len(result) == len(df), "Test failed: FFT result length does not match input data."

    # Non-equidistant data
    df = pd.DataFrame({'x': [0, 1, 2, 4, 7], 'y': [0, 1, 0, -1, 0]})
    with pytest.raises(ValueError, match="Column 'x' contains non-equidistant data."):
        final.fft(df, 'x', 'y')

def test_inv_fft():
    """
    Test the inv_fft function for correct inverse FFT computation.
    """
    # Equidistant data
    df = pd.DataFrame({'x': np.linspace(0, 10, 100), 'y': np.sin(np.linspace(0, 10, 100))})
    y_fft = final.fft(df, 'x', 'y')
    result = final.inv_fft(df, 'x', y_fft)
    assert isinstance(result, pd.Series), "Test failed: Inverse FFT did not return a pandas Series."
    assert len(result) == len(df), "Test failed: Inverse FFT result length does not match the data."
    assert np.allclose(df['y'], np.real(result)), "Test failed: Original data wasn't reconstructed."

    # Non-equidistant data
    df = pd.DataFrame({'x': [0, 1, 2, 4, 7], 'y': [0, 1, 0, -1, 0]})
    with pytest.raises(ValueError, match = "Column 'x' contains non-equidistant data."):
        final.inv_fft(df, 'x', y_fft)

# Testing Calculating the Frequency Axis
def test_calculate_frequency_axis():
    """
    Test the calculate_frequency_axis function.
    """
    # Test case 1: Small dataset
    sample_rate = 2000
    num_samples = 4
    result = list(final.calculate_frequency_axis(sample_rate, num_samples))
    expected = [0.0, 500.0, 1000.0, 1500.0]
    assert result == expected, f"Test failed: Expected {expected}, but got {result}."
    # Test case 2: Larger dataset
    sample_rate = 1000
    num_samples = 8
    result = list(final.calculate_frequency_axis(sample_rate, num_samples))
    expected = [0.0, 125.0, 250.0, 375.0, 500.0, 625.0, 750.0, 875.0]
    assert result == expected, f"Test failed: Expected {expected}, but got {result}."
    # Test case 3: Edge case with 1 sample
    sample_rate = 500
    num_samples = 1
    result = list(final.calculate_frequency_axis(sample_rate, num_samples))
    expected = [0.0]
    assert result == expected, f"Test failed: Expected {expected}, but got {result}."
