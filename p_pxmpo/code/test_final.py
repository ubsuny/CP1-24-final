"""
This module contains unit tests for functions in the `final` module.
It includes tests for reading CSV and markdown data, converting temperatures,
fitting nonlinear sine models, and performing Fourier transforms.
"""
from unittest.mock import patch, mock_open
import numpy as np
import pandas as pd
import pytest

from final import (
    load_csv_data,
    sine_model_function,
    fit_curve,
    fahrenheit_to_kelvin,
    extract_temperature_from_markdown,
    compute_fft,
    compute_inverse_fft
)

def test_load_csv_data_valid():
    """
    Test loading valid CSV data with mock content.
    """
    # Mock valid CSV data
    mock_csv = """Time (s),Latitude (°)
                 1,34.05
                 2,36.12"""
    with patch("builtins.open", mock_open(read_data=mock_csv)):
        time, latitude = load_csv_data("mock_file.csv")
        assert time.tolist() == [1, 2]
        assert latitude.tolist() == [34.05, 36.12]

def test_load_csv_data_file_not_found():
    """
    Test loading CSV data when the file is not found.
    """
    with patch('pandas.read_csv') as mock_read_csv:
        mock_read_csv.side_effect = FileNotFoundError
        with pytest.raises(ValueError, match="File valid.csv not found."):
            load_csv_data('valid.csv')

def test_load_csv_data_empty_file():
    """
    Test loading CSV data when the file is empty.
    """
    with patch('pandas.read_csv') as mock_read_csv:
        mock_read_csv.side_effect = pd.errors.EmptyDataError
        with pytest.raises(ValueError, match="File valid.csv is empty."):
            load_csv_data('valid.csv')

def test_load_csv_data_parser_error():
    """
    Test loading CSV data when there is a parser error.
    """
    with patch('pandas.read_csv') as mock_read_csv:
        mock_read_csv.side_effect = pd.errors.ParserError
        with pytest.raises(ValueError, match="Error parsing valid.csv."):
            load_csv_data('valid.csv')

def test_load_csv_data_generic_exception():
    """
    Test loading CSV data when an unexpected exception occurs.
    """
    with patch('pandas.read_csv') as mock_read_csv:
        mock_read_csv.side_effect = Exception("Some unexpected error")
        with pytest.raises(ValueError, match="Error loading valid.csv: Some unexpected error"):
            load_csv_data('valid.csv')

# Test sine_model_function
def test_sine_model_function():
    """
    Test the sine model function against expected sine values.
    """
    x = np.linspace(0, 2 * np.pi, 100)
    result = sine_model_function(x, 1, 1, 0)
    expected_result = np.sin(x)
    np.testing.assert_almost_equal(result, expected_result, decimal=6)

# Test fit_curve function
def test_fit_curve():
    """
    Test the curve fitting function using noisy sine data.
    """
    x = np.linspace(0, 2 * np.pi, 100)
    y = np.sin(x) + 0.1 * np.random.normal(size=x.size)
    params = fit_curve(x, y, sine_model_function)
    assert len(params) == 3

# Test fahrenheit_to_kelvin function
def test_fahrenheit_to_kelvin():
    """
    Test conversion from Fahrenheit to Kelvin.
    """
    assert fahrenheit_to_kelvin(32) == 273.15
    assert fahrenheit_to_kelvin(212) == 373.15

# Test compute_fft function
def test_compute_fft():
    """
    Test computation of FFT for a simple sine wave.
    """
    x = np.linspace(0, 2 * np.pi, 100)
    y = np.sin(x)
    freq, magnitude, fft_result = compute_fft(y, x)
    assert len(freq) == len(magnitude) == len(fft_result)

# Test compute_inverse_fft function
def test_compute_inverse_fft():
    """
    Test computation of inverse FFT with a simple low-pass filter.
    """
    x = np.linspace(0, 2 * np.pi, 100)
    y = np.sin(x)
    freq, _, fft_result = compute_fft(y, x)
    filter_mask = np.abs(freq) < 10  # Simple low-pass filter
    result = compute_inverse_fft(fft_result, filter_mask)
    assert len(result) == len(y)

def test_extract_temperature_valid():
    """
    Test that temperature data is successfully extracted from a valid markdown file.
    Ensures that the extracted data is not empty.
    """
    try:
        data = extract_temperature_from_markdown(
            '/workspaces/FORKCP1-24-final/p_pxmpo/data/pp05_sinewalk.md'
        )
        assert len(data) > 0, "Extracted temperature data is empty."
    except FileNotFoundError:
        pytest.fail("File pp03_sinewalk.md not found.")

def test_extract_temperature_invalid_file():
    """
    Test that a `FileNotFoundError` is raised when trying to extract temperature 
    from a nonexistent markdown file.
    """
    with pytest.raises(FileNotFoundError):
        extract_temperature_from_markdown('nonexistent.md')

def test_extract_temperature_no_data():
    """
    Test that no temperature data is returned when the markdown file contains no temperature
    information.
    """
    data = extract_temperature_from_markdown('no_temp.md')
    assert len(data) == 0  # Ensure no data is returned

def test_extract_temperature_f_to_k():
    """
    Test that temperatures are correctly converted from Fahrenheit to Kelvin.
    Ensures all extracted temperatures are above or equal to 273.15 K.
    """
    data = extract_temperature_from_markdown(
            '/workspaces/FORKCP1-24-final/p_pxmpo/data/pp05_sinewalk.md'
        )
    assert all(temp >= 273.15 for temp in data)  # Ensure temperatures are converted to Kelvin

pytest.main()
