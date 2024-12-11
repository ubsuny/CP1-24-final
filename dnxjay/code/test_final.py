"""
Unit tests for the functions in final.py.
"""
import numpy as np
import pytest
from final import (
    fahrenheit_to_kelvin,
    parse_temperature_from_markdown,
    list_markdown_files,
    sine_wave,
    fft_with_check,
    calculate_frequency_axis,
)


def test_fahrenheit_to_kelvin():
    """
    Test the fahrenheit_to_kelvin function for correct conversions.
    """
    assert fahrenheit_to_kelvin(32) == 273.15
    assert fahrenheit_to_kelvin(212) == pytest.approx(373.15, 0.01)


def test_parse_temperature_from_markdown(tmp_path):
    """
    Test the parse_temperature_from_markdown function to extract temperature.
    """
    markdown_content = """
    **Temperature (°F):** 75
    """
    file_path = tmp_path / "test.md"
    file_path.write_text(markdown_content)
    assert parse_temperature_from_markdown(file_path) == 75.0

    # Test with different formats
    markdown_content_outdoor = """
    **Temperature (Outdoor):** 75°F
    """
    file_path.write_text(markdown_content_outdoor)
    assert parse_temperature_from_markdown(file_path) == 75.0


def test_list_markdown_files(tmp_path):
    """
    Test the list_markdown_files function for correct filtering.
    """
    (tmp_path / "file1_sinewalk.md").write_text("Test file")
    (tmp_path / "file2.md").write_text("Test file")
    assert list_markdown_files(tmp_path, "sinewalk") == ["file1_sinewalk.md"]


def test_sine_wave():
    """
    Test the sine_wave function for correct sine wave computation.
    """
    assert sine_wave(0, 1, 1, 0) == 0
    assert sine_wave(0.25, 1, 1, 0) == pytest.approx(1.0, 0.01)


def test_fft_with_check():
    """
    Test the fft_with_check function for correct FFT computation.
    """
    data = [1, 2, 3, 4, 5]
    result = fft_with_check(data)
    assert len(result) == len(data)


def test_calculate_frequency_axis():
    """
    Test the calculate_frequency_axis function for correct frequency axis calculation.
    """
    expected = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    result = calculate_frequency_axis(10, 0.1)
    assert np.allclose(result, expected), f"Expected {expected}, but got {result}"
