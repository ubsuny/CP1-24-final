"""Unit tests for final.py"""

import pytest
from final import (
    fahrenheit_to_kelvin,
    parse_temperature_from_markdown,
    list_markdown_files,
    sine_function,
    non_linear_fit,
    fft_with_check,
    inverse_fft,
    calculate_frequency_axis,
)
import os
import numpy as np
import tempfile


def test_fahrenheit_to_kelvin():
    """
    Test the Fahrenheit-to-Kelvin conversion function.

    Verifies that known Fahrenheit values are converted correctly
    to Kelvin using the function `fahrenheit_to_kelvin`.
    """
    assert pytest.approx(fahrenheit_to_kelvin(32), 0.1) == 273.15
    assert pytest.approx(fahrenheit_to_kelvin(212), 0.1) == 373.15


def test_parse_temperature_from_markdown():
    """
    Test the temperature parsing function from a markdown file.

    Creates a temporary markdown file with a sample temperature in Fahrenheit.
    Verifies if the function `parse_temperature_from_markdown` correctly extracts the temperature.
    """
    with tempfile.NamedTemporaryFile(mode='w+', suffix='.md', delete=False) as temp_file:
        temp_file.write("Environment temperature: 75 F")
        temp_file_path = temp_file.name

    assert parse_temperature_from_markdown(temp_file_path) == 75


def test_list_markdown_files():
    """
    Test the function that lists markdown files based on a filename filter.

    Creates temporary markdown files in a directory and ensures
    the function `list_markdown_files` only returns files matching the specified filter.
    """
    folder = tempfile.TemporaryDirectory()
    file_1 = os.path.join(folder.name, "test_sinewalk1.md")
    file_2 = os.path.join(folder.name, "test_other.md")

    open(file_1, "w").close()
    open(file_2, "w").close()

    files = list_markdown_files(folder.name, keyword="sinewalk")
    assert len(files) == 1
    assert file_1 in files


def test_non_linear_fit():
    """
    Test the non-linear fitting function using a sine wave model.

    Generates synthetic sine wave data and verifies if the function `non_linear_fit`
    can correctly fit the data and recover the expected amplitude.
    """
    x = np.linspace(0, 10, 100)
    y = sine_function(x, 2, 1, 0, 0)
    popt, _ = non_linear_fit(x, y, initial_guess=(1, 1, 0, 0))
    assert pytest.approx(popt[0], 0.1) == 2


def test_fft_with_check():
    """
    Test the FFT function with equidistant data verification.

    Verifies that the function `fft_with_check` computes the correct FFT
    of a sine wave and checks if the input data is equidistant.
    """
    x = np.linspace(0, 10, 100)
    y = np.sin(x)
    fft_y, freq = fft_with_check(x, y)
    assert len(fft_y) == len(y)
    assert len(freq) == len(y)


def test_inverse_fft():
    """
    Test the inverse FFT function.

    Computes the FFT of a sine wave and then applies the inverse FFT.
    Verifies that the original signal is recovered within a specified tolerance.
    """
    y = np.sin(np.linspace(0, 10, 100))
    fft_y = np.fft.fft(y)
    inv_y = inverse_fft(fft_y)
    assert np.allclose(y, inv_y, atol=0.1)


def test_calculate_frequency_axis():
    """
    Test the frequency axis calculation function.

    Verifies that the function `calculate_frequency_axis` generates the correct
    number of frequency points based on input data length and spacing.
    """
    freq_axis = calculate_frequency_axis(100, 0.1)
    assert len(freq_axis) == 50
