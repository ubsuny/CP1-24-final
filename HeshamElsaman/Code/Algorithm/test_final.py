"""
This module is to test for the functions created in the final.py file
"""

import tempfile
import pandas as pd
import numpy as np
import pytest
import final

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
