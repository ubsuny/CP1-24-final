"""
Unit tests for the functions in final.py.
Includes:
- Conversion tests for fahrenheit_to_kelvin.
- Parsing tests for parse_temperature.
- Markdown file listing tests for list_markdown_files.
"""

import pytest
import numpy as np
import pandas as pd
from final import (
    fahrenheit_to_kelvin,
    parse_temperature,
    list_markdown_files,
    generate_data,
    non_linear_fit,
    plot_fit,
    check_equidistant,
    compute_fft,
    compute_ifft,
    calculate_frequency_axis,
    convert_to_khz,
    convert_to_mhz,
)
def test_freezing_point():
    """Test conversion of the freezing point of water."""
    assert abs(fahrenheit_to_kelvin(32.0) - 273.15) < 1e-6


def test_boiling_point():
    """Test conversion of the boiling point of water."""
    assert abs(fahrenheit_to_kelvin(212.0) - 373.15) < 1e-6


def test_absolute_zero():
    """Test conversion of absolute zero."""
    assert abs(fahrenheit_to_kelvin(-459.67) - 0) < 1e-6


def test_below_absolute_zero():
    """Test that temperatures below absolute zero raise a ValueError."""
    with pytest.raises(ValueError, match="Temperature in Kelvin cannot be below absolute zero."):
        fahrenheit_to_kelvin(-500.0)


def test_parse_temperature_valid_data():
    """Test parse_temperature with valid DataFrame."""
    data = pd.DataFrame({"Temperature": [32.0, 212.0, 100.0]})
    result = parse_temperature(data)
    expected = (273.15 + 373.15 + 310.9277778) / 3
    assert abs(result - expected) < 1e-6


def test_parse_temperature_missing_column():
    """Test parse_temperature with a DataFrame missing the 'Temperature' column."""
    data = pd.DataFrame({"OtherColumn": [32.0, 212.0, 100.0]})
    with pytest.raises(ValueError, match="The DataFrame must contain a 'Temperature' column."):
        parse_temperature(data)


def test_parse_temperature_empty_column():
    """Test parse_temperature with an empty 'Temperature' column."""
    data = pd.DataFrame({"Temperature": []})
    with pytest.raises(ValueError, match="The 'Temperature' column is empty."):
        parse_temperature(data)


def test_list_markdown_files_valid_dataframe():
    """Test listing markdown files with valid DataFrame input."""
    files = pd.DataFrame({
        "filename": [
            "VS001 sinewalk.md", "VS002 sinewalk.md", "VS003 sinewalk.md",
            "VS010 sinewalk.md", "VS020 sinewalk.md", "other_file.md",
            "VS001 data.csv", "random_file.txt"
        ]
    })
    result = list_markdown_files(files, "VS")
    expected = [
        "VS001 sinewalk.md", "VS002 sinewalk.md", "VS003 sinewalk.md",
        "VS010 sinewalk.md", "VS020 sinewalk.md"
    ]
    assert sorted(result) == sorted(expected)


def test_list_markdown_files_no_matches():
    """Test listing markdown files when no matches are found."""
    files = pd.DataFrame({
        "filename": [
            "other_file.md", "random_file.txt", "data.csv"
        ]
    })
    result = list_markdown_files(files, "VS")
    expected = []
    assert result == expected


def test_list_markdown_files_missing_column():
    """Test listing markdown files when the DataFrame lacks the 'filename' column."""
    files = pd.DataFrame({
        "other_column": [
            "VS001 sinewalk.md", "VS002 sinewalk.md", "VS003 sinewalk.md"
        ]
    })
    with pytest.raises(ValueError, match="The DataFrame must contain a 'filename' column."):
        list_markdown_files(files, "VS")

# Define a simple quadratic model for testing
def quadratic_model(x, a, b, c):
    """
    A quadratic model function.

    Parameters:
        x (float or numpy.ndarray): The input value(s) for the independent variable.
        a (float): The coefficient of the quadratic term.
        b (float): The coefficient of the linear term.
        c (float): The constant term.

    Returns:
        float or numpy.ndarray: The computed value(s) of the quadratic function a*x^2 + b*x + c.
    """
    return a * x**2 + b * x + c

def test_generate_data():
    """
    Test the generate_data function to ensure it returns the correct format and values.
    """
    data = generate_data(quadratic_model, (0, 10, 100), (1, 2, 3), noise_level=0.1)
    assert isinstance(data, pd.DataFrame)
    assert 'x' in data.columns and 'y' in data.columns
    assert len(data) == 100

def test_non_linear_fit():
    """
    Test the non_linear_fit function to ensure it performs non-linear fitting correctly.
    """
    # Generate clean data using a quadratic model
    data = generate_data(quadratic_model, (0, 10, 100), (1, 2, 3), noise_level=0)

    # Introduce NaN values for testing
    data.loc[10, 'x'] = np.nan
    data.loc[20, 'y'] = np.nan

    # Remove NaN values
    clean_data = data.dropna()

    # Normalize data
    clean_data.loc[:, 'x'] = clean_data['x'] / max(clean_data['x'])
    clean_data.loc[:, 'y'] = clean_data['y'] / max(clean_data['y'])

    # Configuration for non_linear_fit
    config = {"step_power": 1, "max_iter": 1000, "tol": 1e-6}

    # Perform non-linear fitting
    params, residuals = non_linear_fit(
        clean_data, quadratic_model, initial_guess=[1, 1, 1], config=config
    )

    # Assertions
    assert len(params) == 3, "Expected 3 fitted parameters."
    assert isinstance(residuals, list), "Residuals should be returned as a list."
    assert not any(np.isnan(params)), f"Parameters should not contain NaN values. Got: {params}"
    assert np.isclose(params[0], 1, atol=0.1), f"Expected parameter 0 close to 1, got {params[0]}"
    assert np.isclose(params[1], 2, atol=0.1), f"Expected parameter 1 close to 2, got {params[1]}"
    assert np.isclose(params[2], 3, atol=0.1), f"Expected parameter 2 close to 3, got {params[2]}"

def test_plot_fit():
    """
    Test the plot_fit function to ensure it generates a plot without errors.
    """
    data = generate_data(quadratic_model, (0, 10, 100), (1, 2, 3), noise_level=0)

    # Add config for non_linear_fit
    config = {"step_power": 4, "max_iter": 1000, "tol": 1e-6}

    # Scale the data
    data['x'] = data['x'] / max(data['x'])
    data['y'] = data['y'] / max(data['y'])

    # Use config for non_linear_fit
    params, _ = non_linear_fit(data, quadratic_model, initial_guess=[0, 0, 0], config=config)

    # Ensure no exceptions during plotting
    try:
        plot_fit(data, quadratic_model, params)
    except (ValueError, TypeError) as e:
        pytest.fail(f"Plotting failed with exception: {e}")

def test_check_equidistant():
    """
    Tests the check_equidistant function to ensure it correctly identifies
    whether the input data is equidistant or not.

    Verifies:
    - Equidistant data returns True.
    - Non-equidistant data returns False.
    """
    index = pd.date_range(start='2023-01-01', periods=5, freq='D')
    data = pd.Series([1, 2, 3, 4, 5], index=index)
    assert check_equidistant(data) is True

    # Test non-equidistant data
    index = pd.to_datetime(['2023-01-01', '2023-01-02', '2023-01-04', '2023-01-07'])
    data = pd.Series([1, 2, 3, 4], index=index)
    assert check_equidistant(data) is False

def test_compute_fft():
    """
    Tests the compute_fft function to ensure it computes the FFT correctly.

    Verifies:
    - The output is a numpy array.
    - The length of the FFT result matches the length of the input data.
    """
    data = np.array([1, 2, 3, 4])
    fft_result = compute_fft(data)
    assert isinstance(fft_result, np.ndarray)
    assert len(fft_result) == len(data)

def test_compute_ifft():
    """
    Tests the compute_ifft function to ensure it correctly computes the
    inverse FFT and reconstructs the original data.

    Verifies:
    - The output is a numpy array.
    - The inverse FFT result closely matches the original input data.
    """
    data = np.array([1, 2, 3, 4])
    fft_result = compute_fft(data)
    ifft_result = compute_ifft(fft_result)
    assert isinstance(ifft_result, np.ndarray)
    np.testing.assert_array_almost_equal(ifft_result, data, decimal=6)

def test_calculate_frequency_axis():
    """
    Test the calculate_frequency_axis function.

    This test verifies that the frequency axis is calculated correctly for a given sample rate
    and number of samples. It checks:
    - Correct positive and negative frequency calculations.
    - Handling of invalid inputs (sample_rate <= 0 or num_samples <= 0).
    """
    sample_rate = 1000  # Hz
    num_samples = 10
    expected = [0.0, 100.0, 200.0, 300.0, 400.0, 500.0, -400.0, -300.0, -200.0, -100.0]
    result = calculate_frequency_axis(sample_rate, num_samples)
    assert result == expected, f"Expected {expected} but got {result}"

    with pytest.raises(ValueError):
        calculate_frequency_axis(0, 10)

    with pytest.raises(ValueError):
        calculate_frequency_axis(1000, 0)

def test_convert_to_khz():
    """
    Test the convert_to_khz function.

    This test verifies that the frequency axis is correctly converted from Hz to kHz.
    It checks:
    - Conversion of positive and negative frequencies from Hz to kHz.
    """
    freq_axis = [0.0, 100.0, 200.0, 300.0, 400.0, 500.0, -400.0, -300.0, -200.0, -100.0]
    expected = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, -0.4, -0.3, -0.2, -0.1]
    result = convert_to_khz(freq_axis)
    assert result == expected, f"Expected {expected} but got {result}"

def test_convert_to_mhz():
    """
    Test the convert_to_mhz function.

    This test verifies that the frequency axis is correctly converted from Hz to MHz.
    It checks:
    - Conversion of positive and negative frequencies from Hz to MHz.
    """
    freq_axis = [0.0, 100.0, 200.0, 300.0, 400.0, 500.0, -400.0, -300.0, -200.0, -100.0]
    expected = [0.0, 0.0001, 0.0002, 0.0003, 0.0004, 0.0005, -0.0004, -0.0003, -0.0002, -0.0001]
    result = convert_to_mhz(freq_axis)
    assert result == expected, f"Expected {expected} but got {result}"
