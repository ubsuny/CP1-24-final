"""
This module contains unit tests for functions in the `final` module.
It includes tests for reading CSV and markdown data, converting temperatures,
fitting nonlinear sine models, and performing Fourier transforms.
"""
from unittest.mock import mock_open, patch
import numpy as np
from final import (curve_fit, read_first_two_columns, extract_temperature_from_markdown,
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

# Define a simple model function for testing
def linear_model(x, m, c):
    """
    Linear model function.
    """
    return m * x + c

def test_curve_fit():
    """
    Unit test for fitting a linear model to synthetic data using curve_fit.

    This function generates synthetic noisy data based on a known linear model, fits the model 
    to the data using the curve_fit function, and performs assertions to verify that the fitted 
    parameters and covariance matrix are as expected.

    Raises:
        AssertionError: If the fitted parameters deviate significantly from the true parameters 
                        or if the covariance matrix is None or has an incorrect shape.
    """
    # Synthetic data
    np.random.seed(42)  # Set a fixed seed for reproducibility
    xdata = np.linspace(0, 10, 50)
    true_params = [2.0, 1.0]  # Slope (m) and intercept (c)
    noise = np.random.normal(0, 0.5, size=xdata.shape)  # Add some noise
    ydata = linear_model(xdata, *true_params) + noise

    # Initial guess for parameters
    initial_guess = [1.0, 0.0]

    # Call the curve_fit function
    popt, pcov = curve_fit(linear_model, xdata, ydata, p0=initial_guess)

    # Assert the fitted parameters are close to the true parameters
    assert np.allclose(popt, true_params, atol=0.2), f"Fitted parameters {popt} deviate from true parameters {true_params}"

    # Assert the covariance matrix is not None and has the correct shape
    assert pcov is not None, "Covariance matrix None"
    assert pcov.shape == (len(true_params), len(true_params)), "Covariance matrix shape incorrect"
