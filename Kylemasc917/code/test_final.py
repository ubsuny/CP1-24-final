"""This is the unit test functions for Kylemasc917 final algorithm"""

import tempfile
import os
from unittest import mock
import pytest
import numpy as np
from final import fahrenheit_to_kelvin, extract_temp_from_markdown, list_markdown_files
from final import (
    sine_wave, nonlinear_fit, is_non_equidistant, fft, ifft,
    calculate_frequency_axis, load_csv_data, shift_and_fft,
    plot_fft, plot_ifft, plot_filtered_ifft
)



def test_conversion():
    """Test standard conversions"""
    assert pytest.approx(fahrenheit_to_kelvin(32), 0.01) == 273.15
    assert pytest.approx(fahrenheit_to_kelvin(212), 0.01) == 373.15
    assert pytest.approx(fahrenheit_to_kelvin(-459.67), 0.01) == 0

def test_absolute_zero():
    """Test boundary condition at absolute zero"""
    assert fahrenheit_to_kelvin(-459.67) == 0

def test_below_absolute_zero():
    """Test input below absolute zero"""
    with pytest.raises(ValueError):
        fahrenheit_to_kelvin(-500)

def test_non_numeric_input():
    """Test non-numeric input handling"""
    with pytest.raises(TypeError):
        fahrenheit_to_kelvin("invalid input")

def test_extract_temp_no_temperatures():
    """Test case for a file containing no temperatures"""
    file_content = """This is a markdown file with no temperatures."""

    with mock.patch("builtins.open", mock.mock_open(read_data=file_content)):
        result = extract_temp_from_markdown("test_file.md")

    assert result == []

def test_extract_temp_invalid_format():
    """ Test case for invalid temperature format"""
    file_content = """This is a test with invalid temperature values.
    Invalid temp: 72A, another invalid: 85X."""

    with mock.patch("builtins.open", mock.mock_open(read_data=file_content)):
        result = extract_temp_from_markdown("test_file.md")

    assert result == []


def test_list_markdown_files_valid():
    """Test case for a valid directory with matching markdown files"""
    mock_files = ['experimentname_results.md', 'experimentname_data.md',
    'otherfile.txt', 'README.md']
    with mock.patch('os.listdir', return_value=mock_files):
        result = list_markdown_files('/mock/directory', 'experimentname')

    assert result == ['experimentname_results.md', 'experimentname_data.md']

def test_list_markdown_files_no_match():
    """Test case for a directory with no matching markdown files"""
    mock_files = ['experiment1_results.md', 'experiment2_data.md', 'README.md']
    with mock.patch('os.listdir', return_value=mock_files):
        result = list_markdown_files('/mock/directory', 'experimentname')

    assert result == []

def test_list_markdown_files_no_md_files():
    """Test case for a directory with no markdown files"""
    mock_files = ['otherfile.txt', 'image.jpg', 'README.md']
    with mock.patch('os.listdir', return_value=mock_files):
        result = list_markdown_files('/mock/directory', 'experimentname')

    assert result == []

def test_list_markdown_files_directory_not_found():
    """Test case for a non-existent directory"""
    with mock.patch('os.listdir', side_effect=FileNotFoundError):
        result = list_markdown_files('/mock/nonexistent_directory', 'experimentname')

    assert result == 'Directory not found: /mock/nonexistent_directory'







def test_sine_wave():
    """Test sine wave function"""
    x_data = np.linspace(0, 1, 100)
    amplitude, frequency, phase, offset = 2.0, 1.0, np.pi/4, 1.0
    y_data = sine_wave(x_data, amplitude, frequency, phase, offset)

    assert y_data.shape == x_data.shape
    assert np.isclose(y_data[0], amplitude * np.sin(phase) + offset)

def test_nonlinear_fit():
    """Test non-linear function"""
    x_data = np.linspace(0, 1, 100)
    true_params = (2.0, 1.0, np.pi/4, 1.0)
    y_data = sine_wave(x_data, *true_params)
    initial_guess = [1, 1, 0, 0]

    fitted_params = nonlinear_fit(x_data, y_data, initial_guess)

    for fitted, true in zip(fitted_params, true_params):
        assert np.isclose(fitted, true, rtol=1e-2)

def test_is_non_equidistant():
    """Test equidistant data function"""
    equidistant_data = [0, 1, 2, 3, 4]
    non_equidistant_data = [0, 1, 3, 6, 10]

    assert not is_non_equidistant(equidistant_data)
    assert is_non_equidistant(non_equidistant_data)

def test_fft():
    """test fft function"""
    x_data = np.linspace(0, 1, 128)  # Equidistant x_data
    y_data = np.sin(2 * np.pi * 5 * x_data)  # Sine wave signal
    data = (x_data, y_data)  # Tuple as expected by `fft`

    fft_result, freqs = fft(data)

    # Ensure the length of FFT output matches the number of samples
    assert len(fft_result) == len(y_data), "FFT result length mismatch"
    assert len(freqs) == len(y_data), "Frequency length mismatch"
    # Check that the dominant frequency corresponds to the signal frequency (5 Hz)
    dominant_frequency = freqs[np.argmax(np.abs(fft_result))]
    assert np.isclose(dominant_frequency, 5, atol=0.1), "Dominant frequency mismatch"


def test_ifft():
    """Test ifft function"""
    x_data = np.linspace(0, 1, 128)
    y_data = np.sin(2 * np.pi * 5 * x_data)
    data = (x_data, y_data)  # Tuple as expected by the `fft` function

    fft_result, _ = fft(data)
    y_reconstructed = ifft(fft_result)

    # Allow for small numerical differences
    assert np.allclose(y_data, np.real(y_reconstructed), atol=10), "IFFT failed"


def test_calculate_frequency_axis():
    """Test for frequency axis function"""
    x_data = np.linspace(0, 10, 128)
    total_time = 10

    freq_axis = calculate_frequency_axis(x_data, total_time)

    assert len(freq_axis) == len(x_data) // 2
    assert np.isclose(freq_axis[0], 0)

def test_load_csv_data():
    """Test function that loads data from csv files"""
    csv_content = "Latitude (°),Longitude (°)\n0,1\n1,2\n2,3\n3,4\n"

    # Create a temporary CSV file
    with tempfile.NamedTemporaryFile(
     delete=False, mode='w', encoding='utf-8', suffix=".csv"
    ) as temp_csv:

        temp_csv.write(csv_content)
        temp_csv.close()

        # Pass the temporary file path to the function
        data = load_csv_data([temp_csv.name])

        # Validate the results
        assert len(data) == 1, "Unexpected number of datasets"
        assert data[0][0] == [0.0, 1.0, 2.0, 3.0], "X values mismatch"
        assert data[0][1] == [1.0, 2.0, 3.0, 4.0], "Y values mismatch"

    # Clean up the temporary file
    os.remove(temp_csv.name)

def test_shift_and_fft():
    """Test the shift of data for plots"""
    x_data = [0, 1, 2, 3, 4]
    y_data = [1, 2, 3, 4, 5]

    x_shifted, y_fft, freqs = shift_and_fft(x_data, y_data)

    assert np.isclose(x_shifted[0], 0)
    assert len(y_fft) == len(y_data)
    assert len(freqs) == len(y_data)

def test_plot_fft(tmp_path):
    """Test plot fft"""
    data = [([0, 1, 2, 3], [1, 2, 3, 4])]
    output_image = tmp_path / "fft_plot.png"

    plot_fft(data, output_image)

    assert output_image.exists()

def test_plot_ifft(tmp_path):
    """Test plot ifft"""
    data = [([0, 1, 2, 3], [1, 2, 3, 4])]
    output_image = tmp_path / "ifft_plot.png"

    plot_ifft(data, output_image)

    assert output_image.exists()

def test_plot_filtered_ifft(tmp_path):
    """Test plot filtered ifft"""
    data = [([0, 1, 2, 3], [1, 2, 3, 4])]
    output_image = tmp_path / "filtered_ifft_plot.png"

    plot_filtered_ifft(data, output_image)

    assert output_image.exists()
