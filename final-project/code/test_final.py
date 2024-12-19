"""
Unit tests for the final project module.

This script tests various functions such as temperature conversion, file parsing, 
data fitting, and Fourier Transform utilities.
"""

# Standard library imports
import os

# Third-party imports
import pytest
import numpy as np
import pandas as pd

# Local imports
from final import (
    fahrenheit_to_kelvin,
    parse_temperature_from_markdown,
    list_markdown_files,
    sine_function,
    non_linear_fitting,
    _compute_gradients,
    check_equidistant,
    perform_fft,
    perform_inverse_fft,
    fft_wrapper,
    calculate_frequency_axis,
)

# Test fahrenheit_to_kelvin
def test_fahrenheit_to_kelvin():
    """
    Test the fahrenheit_to_kelvin function for accuracy.
    """
    assert pytest.approx(fahrenheit_to_kelvin(32), 0.1) == 273.15  # Freezing point of water
    assert pytest.approx(fahrenheit_to_kelvin(212), 0.1) == 373.15  # Boiling point of water
    assert pytest.approx(fahrenheit_to_kelvin(-40), 0.1) == 233.15  # Negative value

# Test parse_temperature_from_markdown
def test_parse_temperature_from_markdown(tmp_path):
    """
    Test parsing temperature from a markdown file.
    """
    markdown_content = (
        "| Parameter       | Value    |\n"
        "|-----------------|----------|\n"
        "| Temperature     | 72°F     |\n"
    )

    file_path = tmp_path / "test.md"
    file_path.write_text(markdown_content, encoding="utf-8")

    assert pytest.approx(parse_temperature_from_markdown(file_path), 0.1) == 72.0

# Test list_markdown_files
def test_list_markdown_files(tmp_path):
    """
    Test listing markdown files with a specific filter.
    """
    file1 = tmp_path / "test1.md"
    file2 = tmp_path / "test2.md"
    file3 = tmp_path / "other_file.txt"
    file1.write_text("")
    file2.write_text("")
    file3.write_text("")

    files = list_markdown_files(tmp_path, "test")
    assert len(files) == 2
    assert str(file1) in files
    assert str(file2) in files

# Test sine_function
def test_sine_function():
    """
    Test the sine function for correct outputs.
    """
    x_data = np.linspace(0, 1, 100)
    amplitude, frequency, phase, offset = 2.0, 1.0, 0.0, 1.0
    y_data = sine_function(x_data, amplitude, frequency, phase, offset)

    assert len(y_data) == len(x_data)
    assert pytest.approx(max(y_data), 0.1) == amplitude + offset

# Test non_linear_fitting
def test_non_linear_fitting():
    """
    Test non-linear fitting using synthetic sine function data.
    """
    x_data = np.linspace(0, 1, 100)
    y_data = sine_function(x_data, 2.0, 1.0, 0.0, 1.0)

    params = {
        "x_data": x_data,
        "y_data": y_data,
        "fit_function": sine_function,
        "initial_params": [1.0, 0.5, 0.0, 0.0],
        "step_power": 5,
        "learning_rate": 0.001,
        "max_iterations": 2000,
    }

    optimized_params = non_linear_fitting(params)
    assert pytest.approx(optimized_params[0], rel=0.1) == 2.0  # Amplitude
    assert pytest.approx(optimized_params[1], rel=0.1) == 1.0  # Frequency
    assert pytest.approx(optimized_params[2], abs=0.1) == 0.0  # Phase
    assert pytest.approx(optimized_params[3], rel=0.1) == 1.0  # Offset

def test_compute_gradients():
    """
    Test the _compute_gradients function to ensure gradients are calculated correctly.
    """
    # Define test inputs
    x_sampled = np.linspace(0, 1, 10)  # Independent variable
    true_params = [2.0, 1.0, 0.0, 1.0]  # True sine wave parameters
    y_pred = sine_function(x_sampled, *true_params)  # Predicted values using sine function
    residuals = y_pred - y_pred  # Residuals should be zero (since y_pred matches the input)
    initial_params = [1.5, 0.8, 0.1, 0.5]  # Initial guess for parameters

    # Compute gradients using the helper function
    gradients = _compute_gradients(sine_function, initial_params, x_sampled, y_pred, residuals)

    # Expected gradients should be close to zero since residuals are zero
    assert np.allclose(gradients, 0.0, atol=1e-6), f"Gradients should be close to zero but got {gradients}"

    # Add non-zero residuals to simulate updates
    y_pred = sine_function(x_sampled, *initial_params)
    residuals = y_pred - sine_function(x_sampled, *true_params)  # Residuals based on difference from true params

    # Compute gradients again
    gradients = _compute_gradients(sine_function, initial_params, x_sampled, y_pred, residuals)

    # Check that gradients are not zero and align with updates
    assert not np.allclose(gradients, 0.0, atol=1e-6), "Gradients should not be zero for non-zero residuals"
    assert len(gradients) == len(initial_params), "Gradients length mismatch with initial parameters"

# Test check_equidistant
def test_check_equidistant():
    """
    Test equidistant and non-equidistant data checks.
    """
    equidistant_data = np.linspace(0, 10, 100)
    non_equidistant_data = np.array([0, 1, 2, 4, 8])
    assert check_equidistant(equidistant_data) is True
    assert check_equidistant(non_equidistant_data) is False

# Test perform_fft and perform_inverse_fft
def test_fft_and_inverse_fft():
    """
    Test FFT and inverse FFT for correctness.
    """
    signal = np.sin(2 * np.pi * np.linspace(0, 1, 100))
    fft_result = perform_fft(signal)
    inverse_fft_result = perform_inverse_fft(fft_result)

    assert len(fft_result) == len(signal)
    assert np.allclose(signal, inverse_fft_result.real, atol=1e-6)

# Test fft_wrapper
def test_fft_wrapper(tmp_path):
    """
    Test the fft_wrapper function for correct FFT and inverse FFT outputs.
    """
    data = {
        "time": np.linspace(0, 1, 100),
        "signal": np.sin(2 * np.pi * np.linspace(0, 1, 100)),
    }
    file_path = tmp_path / "test.csv"
    pd.DataFrame(data).to_csv(file_path, index=False)

    result = fft_wrapper(file_path, "time", "signal")
    assert result["equidistant"] is True
    assert "fft" in result
    assert "inverse_fft" in result

# Test calculate_frequency_axis
def test_calculate_frequency_axis():
    """
    Test the frequency axis calculation for FFT.
    """
    samples = 100
    rate = 1000
    frequencies = calculate_frequency_axis(samples, rate)
    assert len(frequencies) == samples // 2 + 1
    assert frequencies[0] == 0
    assert frequencies[-1] == rate / 2
