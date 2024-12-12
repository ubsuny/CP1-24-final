"""
This module contains unit tests for functions in the `final` module.
It includes tests for reading CSV and markdown data, converting temperatures,
fitting nonlinear sine models, and performing Fourier transforms.
"""
from unittest.mock import mock_open, patch
import numpy as np
from final import (read_first_two_columns, extract_temperature_from_markdown,
                    fahrenheit_to_kelvin, nonlinear_sine_fit, compute_fft,
                    compute_inverse_fft)

# Test for read_first_two_columns function
def test_read_first_two_columns():
    """
    Test the `read_first_two_columns` function to ensure it correctly reads 
    the first two columns from a CSV file.
    """
    mock_csv = "header1,header2\n1,2\n3,4\n5,6\n"
    with patch("builtins.open", mock_open(read_data=mock_csv)):
        first_column, second_column = read_first_two_columns("test.csv")
        assert first_column == ["1", "3", "5"]
        assert second_column == ["2", "4", "6"]

def test_read_first_two_columns_file_not_found():
    """
    Test the `read_first_two_columns` function to handle the case where 
    the file is not found.
    """
    with patch("builtins.open", side_effect=FileNotFoundError):
        first_column, second_column = read_first_two_columns("nonexistent.csv")
        assert not first_column
        assert not second_column

def test_extract_temperature_from_markdown():
    """
    Test the `extract_temperature_from_markdown` function to ensure it 
    correctly extracts temperature and unit from a markdown file.
    """
    mock_markdown = "Temperature: 25.5 °C"
    with patch("builtins.open", mock_open(read_data=mock_markdown)):
        temp, unit = extract_temperature_from_markdown("test.md")
        assert temp == 25.5
        assert unit == "°C"

def test_extract_temperature_from_markdown_no_temp():
    """
    Test the `extract_temperature_from_markdown` function to handle the case 
    where no temperature is found in the markdown file.
    """
    mock_markdown = "Some text without temperature."
    with patch("builtins.open", mock_open(read_data=mock_markdown)):
        temp, unit = extract_temperature_from_markdown("test.md")
        assert temp is None
        assert unit is None

def test_fahrenheit_to_kelvin():
    """
    Test the `fahrenheit_to_kelvin` function to ensure it correctly converts 
    Fahrenheit to Kelvin.
    """
    assert np.isclose(fahrenheit_to_kelvin(32), 273.15)
    assert np.isclose(fahrenheit_to_kelvin(0), 255.372)

def test_nonlinear_sine_fit():
    """
    Test the `nonlinear_sine_fit` function to ensure it correctly performs 
    nonlinear sine fitting on data from a CSV file.
    """
    mock_csv = "x,y\n0,0\n1,1\n2,0\n3,-1\n4,0\n"
    with patch("builtins.open", mock_open(read_data=mock_csv)):
        popt, _, x_data, y_data = nonlinear_sine_fit("test.csv")
        assert len(popt) == 4  # We expect 4 parameters from sine fit
        assert popt is not None
        assert x_data is not None
        assert y_data is not None

def test_nonlinear_sine_fit_invalid_data():
    """
    Test the `nonlinear_sine_fit` function to handle the case with invalid data 
    in the CSV file (non-numeric values).
    """
    mock_csv = "x,y\n0,0\n1,non-numeric\n2,0\n"
    with patch("builtins.open", mock_open(read_data=mock_csv)):
        popt, _, x_data, y_data = nonlinear_sine_fit("test.csv")
        assert popt is None
        assert x_data is None
        assert y_data is None

def test_compute_fft():
    """
    Test the `compute_fft` function to ensure it correctly computes the 
    FFT of given data.
    """
    y_data = np.array([0, 1, 0, -1])
    x_data = np.array([0, 1, 2, 3])
    fft_freq, fft_magnitude, fft_result = compute_fft(y_data, x_data)

    assert len(fft_freq) == len(y_data)
    assert len(fft_magnitude) == len(y_data)
    assert len(fft_result) == len(y_data)
    assert np.iscomplexobj(fft_result)

def test_compute_inverse_fft():
    """
    Test the `compute_inverse_fft` function to ensure it correctly computes 
    the inverse FFT of a given filtered FFT result.
    """
    fft_result = np.array([1+1j, 2+2j, 3+3j])
    filter_mask = np.array([1, 0, 1])

    inverse_fft_result = compute_inverse_fft(fft_result, filter_mask)

    assert len(inverse_fft_result) == len(fft_result)
    assert np.isrealobj(inverse_fft_result)
    # Allow for small tolerance due to floating-point precision
    assert np.allclose(inverse_fft_result, [1.33333333, 0.69935874, -1.03269207], atol=1e-5)
