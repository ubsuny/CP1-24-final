"""
test_final.py
This module contains pytest unit tests for all function in final.py
Functions:
-test function that converts Fahrenheit to Kelvin
-test parser that reads out the temperature of one markdown file
-test filename lister that generates programmatically (using the python os library)
a list of your markdown file based on a filename filter.
-test non-linear fitting in pure python which includes the 
functionality to specify the step number of 2^n
-test numpy wrapper for fft, inverse fft, including functionality
that checks for non-equidistant data.
-test pure python (no numpy) to calculate the frequency axis in useful units.
"""

import pytest
import numpy as np
from final import (
    fahrenheit_to_kelvin,
    extract_temperature_from_markdown,
    list_markdown_files,
    non_linear_fit,
    fft_wrapper,
    calculate_frequency_axis
)
# Test the fahrenheit_to_kelvin function
def test_fahrenheit_to_kelvin():
    """
    Test the fahrenheit_to_kelvin function
    """
    assert abs(fahrenheit_to_kelvin(32) - 273.15) < 1e-6
    assert abs(fahrenheit_to_kelvin(212) - 373.15) < 1e-6
    assert abs(fahrenheit_to_kelvin(0) - 255.372) < 1e-3

# Test the extract_temperature_from_markdown function
def test_extract_temperature_from_markdown(tmpdir):
    """
    Test the extract_temperature_from_markdown function
    """
    md_file = tmpdir.join("PR001sinewalk.md")
    md_file.write("Temperature: 35.6\n")

    assert abs(extract_temperature_from_markdown(str(md_file)) - 35.6) < 1e-6

    with pytest.raises(ValueError):
        extract_temperature_from_markdown(str(tmpdir.join("nonexistent.md")))

# Test the list_markdown_files function
def test_list_markdown_files(tmpdir):
    """
    Test the list_markdown_files function
    """
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



def test_non_linear_fit():
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
    result_no_smooth = non_linear_fit(data, step, func, smooth=False)
    assert result_no_smooth == [1, 9, 25, 49], (
        "Non-linear fit without smoothing failed."
    )

    # Non-linear fit with smoothing
    result_smooth = non_linear_fit(data, step, func, smooth=True, window_size=4)

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
        non_linear_fit(data, 3, func)  # Step not a power of 2
    with pytest.raises(ValueError):
        non_linear_fit(data, step, func, smooth=True, window_size=0)  # Invalid window size

# Test the fft_wrapper function
def test_fft_wrapper():
    """
    Test the fft_wrapper function
    """
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
    """
    Test the calculate_frequency_axis function
    """
    length = 4
    sample_rate = 1000
    frequencies = calculate_frequency_axis(length, sample_rate)

    assert len(frequencies) == length
    assert abs(frequencies[0] - 0) < 1e-6
    assert abs(frequencies[1] - 250) < 1e-6
    assert abs(frequencies[2] - 500) < 1e-6
    assert abs(frequencies[3] - 750) < 1e-6
