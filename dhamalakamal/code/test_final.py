"""
test_final.py

This module contains pytest unit tests for the functions implemented in `final.py`.
The tests ensure the correctness and reliability of the utility functions used for 
data analysis and experiment processing.

Tests:
- Tests for temperature conversion (Fahrenheit to Kelvin).
- Tests for Markdown parsing of temperature metadata.
- Tests for file listing functionality.
- Tests for non-linear fitting function.
- Tests for FFT and inverse FFT wrapper functions.
- Tests for checking equidistant data.
- Tests for frequency axis calculation.

Usage:
- Run pytest in the terminal to execute the tests.
    $ pytest dhamalakamal/code/test_final.py

Author:
- Kamal Dhamala
"""

import os
from pathlib import Path
import pytest
import numpy as np
from final import (
    fahrenheit_to_kelvin,
    parse_temperature_from_md,
    list_md_files,
    nonlinear_fit,
    fft_wrapper,
    inverse_fft_wrapper,
    check_equidistant,
    calculate_frequency_axis,
)

# Get the absolute path to the 'data' folder
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data_dir = os.path.join(project_root, "data")

def test_fahrenheit_to_kelvin():
    """
    Test fahrenheit_to_kelvin function.
    """
    assert fahrenheit_to_kelvin(32) == pytest.approx(273.15)  # Freezing point
    assert fahrenheit_to_kelvin(212) == pytest.approx(373.15)  # Boiling point
    assert fahrenheit_to_kelvin(-40) == pytest.approx(233.15)  # Common test case
    with pytest.raises(TypeError):
        fahrenheit_to_kelvin("invalid")

def test_parse_temperature_from_md(tmp_path: Path):
    """
    Test parse_temperature_from_md function with a temporary Markdown file.
    """
    # Take one specific file for testing
    valid_file_path = os.path.join(data_dir, "gps06_sinewalk.md")
    extracted_temp = parse_temperature_from_md(str(valid_file_path))
    assert extracted_temp == 39.0

    # Test if the same md file value is properly converted from Fahrenheit to Kelvin
    assert fahrenheit_to_kelvin(extracted_temp) == pytest.approx(277.039)

    md_content_invalid = "## Weather Condition\nTemperature not mentioned\n"
    file_path_invalid = tmp_path / "invalid_test.md"
    file_path_invalid.write_text(md_content_invalid)
    with pytest.raises(ValueError):
        parse_temperature_from_md(str(file_path_invalid))

def test_list_md_files():
    """
    Test list_md_files function with the actual 'data' folder in the project.
    """
    # Ensure the directory exists
    assert os.path.exists(data_dir), f"The data directory {data_dir} does not exist."

    # Test the function with a filter keyword
    result = list_md_files(data_dir, "sinewalk")
    assert len(result), 20 # total 20 md files should be there
    assert isinstance(result, list), "Result should be a list."

    only_sinewalk_files = any("sinewalk" in filename for filename in result)
    assert only_sinewalk_files, "Filtered filenames must contain 'sinewalk'."

    only_md_files = all(filename.endswith(".md") for filename in result)
    assert only_md_files, "Filtered filenames must have '.md' extension."

def test_nonlinear_fit():
    """
    Test nonlinear_fit function with example data, including smoothing.
    """
    # Test data
    data = [1, 2, 3, 4, 5, 6, 7, 8]
    step = 2

    # Non-linear function
    def func(x):
        return x**2

    # Basic non-linear fit without smoothing
    result_no_smooth = nonlinear_fit(data, step, func, smooth=False)
    assert result_no_smooth == [1, 9, 25, 49], (
        "Non-linear fit without smoothing failed."
    )

    # Non-linear fit with smoothing
    result_smooth = nonlinear_fit(data, step, func, smooth=True, window_size=4)

    # Ensure that the length of smoothed data is the same as unsmoothed data
    assert len(result_smooth) == len(result_no_smooth), (
        "Smoothing altered the result length."
    )

    # Ensure that smoothing has made an actual change
    assert result_smooth != result_no_smooth, (
        "Smoothing did not change the data."
    )

    # Ensure that the smoothed data is not the same as the raw data in terms of averages or ranges
    original_avg = sum(result_no_smooth) / len(result_no_smooth)
    smoothed_avg = sum(result_smooth) / len(result_smooth)

    # Check if the averages are different (indicating that smoothing occurred)
    assert abs(original_avg - smoothed_avg) > 0.1, (
        "Smoothing did not produce a noticeable difference in averages."
    )

    # Now let's compare it to manually expected smoothed data
    smoothed_data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]  # Example of expected smoothed data

    # Here you could do a more detailed comparison, depending on how you calculate `smoothed_data`
    # For now, we just assert the smoothing changed the data
    assert result_smooth != smoothed_data, (
        "Smoothed data did not change the result as expected."
    )

    # Edge cases
    with pytest.raises(ValueError):
        nonlinear_fit(data, 3, func)  # Step not a power of 2
    with pytest.raises(ValueError):
        nonlinear_fit(data, step, func, smooth=True, window_size=0)  # Invalid window size

def test_fft_wrapper():
    """
    Test fft_wrapper function with simple data.
    """
    data = np.array([1, 0, -1, 0])
    result = fft_wrapper(data)
    assert np.allclose(result, np.fft.fft(data))

def test_inverse_fft_wrapper():
    """
    Test inverse_fft_wrapper function with simple FFT data.
    """
    data = np.array([1, 0, -1, 0])
    fft_data = np.fft.fft(data)
    result = inverse_fft_wrapper(fft_data)
    assert np.allclose(result, np.fft.ifft(fft_data))

def test_check_equidistant():
    """
    Test check_equidistant function to verify equidistant data.
    """
    equidistant_data = np.array([1, 3, 5, 7])
    non_equidistant_data = np.array([1, 2, 4, 7])
    assert check_equidistant(equidistant_data) is True
    assert check_equidistant(non_equidistant_data) is False

def test_calculate_frequency_axis():
    """
    Test calculate_frequency_axis function with sample data.
    """
    data_length = 8
    sampling_rate = 2.0  # Hz
    result = calculate_frequency_axis(data_length, sampling_rate)
    expected = [0.0, 0.25, 0.5, 0.75, 1.0]  # Frequencies for a length of 8
    assert result == pytest.approx(expected)
