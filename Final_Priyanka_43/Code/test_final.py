"""
test_final.py
This module contains pytest unit tests for all function in final.py
Functions:
-test function that converts Fahrenheit to Kelvin
-test parser that reads out the temperature of one markdown file
-test filename lister that generates programmatically (using the python os library) a list of your markdown file based on a filename filter.
-test non-linear fitting in pure python which includes the functionality to specify the step number of 2^n
-test numpy wrapper for fft, inverse fft, including functionality that checks for non-equidistant data.
-test pure python (no numpy) to calculate the frequency axis in useful units.
"""

import pytest
import numpy as np
import os
from final import fahrenheit_to_kelvin, extract_temperature_from_markdown, list_markdown_files, non_linear_fit, fft_wrapper, calculate_frequency_axis

# Test the fahrenheit_to_kelvin function
def test_fahrenheit_to_kelvin():
    assert abs(fahrenheit_to_kelvin(32) - 273.15) < 1e-6
    assert abs(fahrenheit_to_kelvin(212) - 373.15) < 1e-6
    assert abs(fahrenheit_to_kelvin(0) - 255.372) < 1e-6

# Test the extract_temperature_from_markdown function
def test_extract_temperature_from_markdown(tmpdir):
    md_file = tmpdir.join("PR001sinewalk.md")
    md_file.write("# Temperature: 35.6\n")

    assert abs(extract_temperature_from_markdown(str(md_file)) - 35.6) < 1e-6

    with pytest.raises(ValueError):
        extract_temperature_from_markdown(str(tmpdir.join("nonexistent.md")))

# Test the list_markdown_files function
def test_list_markdown_files(tmpdir):
    # Create 20 markdown files with names PR001sinewalk.md to PR020sinewalk.md
    for i in range(1, 21):
        md_file = tmpdir.join(f"PR{i:03d}sinewalk.md")
        md_file.write("# Data")

    # Also create a non-markdown file for testing
    tmpdir.join("otherfile.txt").write("# Data")

    # Call the function to list markdown files with "sinewalk" in the name
    files = list_markdown_files(str(tmpdir), "sinewalk")
    
    # Assert there are 20 markdown files
    assert len(files) == 20
    
    # Assert each of the 20 files is in the list
    for i in range(1, 21):
        assert f"PR{i:03d}sinewalk.md" in files

# Test the non_linear_fit function (simple quadratic example)
def test_non_linear_fit():
    def model(x, a, b, c):
        return a * x**2 + b * x + c

    x_data = [1, 2, 3]
    y_data = [6, 11, 18]
    initial_params = [1, 1, 1]

    fitted_params = non_linear_fit(model, x_data, y_data, initial_params)

    assert abs(fitted_params[0] - 2) < 1e-6
    assert abs(fitted_params[1] - 1) < 1e-6
    assert abs(fitted_params[2] - 0) < 1e-6

# Test the fft_wrapper function
def test_fft_wrapper():
    data = np.array([1, 2, 3, 4])

    # Check for normal FFT
    fft_result = fft_wrapper(data)
    assert np.allclose(fft_result, np.fft.fft(data))

    # Check for inverse FFT
    ifft_result = fft_wrapper(data, inverse=True)
    assert np.allclose(ifft_result, np.fft.ifft(data))

    # Test for non-equidistant data (raises ValueError)
    with pytest.raises(ValueError):
        fft_wrapper(np.array([1, 3, 2]), inverse=True)

# Test the calculate_frequency_axis function
def test_calculate_frequency_axis():
    length = 4
    sample_rate = 1000
    frequencies = calculate_frequency_axis(length, sample_rate)

    assert len(frequencies) == length
    assert abs(frequencies[0] - 0) < 1e-6
    assert abs(frequencies[1] - 250) < 1e-6
    assert abs(frequencies[2] - 500) < 1e-6
    assert abs(frequencies[3] - 750) < 1e-6
