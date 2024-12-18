
### **test_final.py (Unit Tests)**
import sys
sys.path.append("/workspaces/CP1-24-final-Forked")
import pytest
from Tolani4.Code.final import fahrenheit_to_kelvin, parse_temperature, list_md_files, read_csv_data, compute_fft, calculate_frequency_axis
import os
import numpy as np
import pandas as pd
import tempfile


# Test fahrenheit_to_kelvin function
def test_fahrenheit_to_kelvin():
    assert fahrenheit_to_kelvin(32) == 273.15  # Freezing point of water
    assert fahrenheit_to_kelvin(212) == 373.15  # Boiling point of water


# Test parse_temperature function with a mock markdown file
@pytest.fixture
def mock_md_file():
    "creates a mock markdown file to test"
    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        temp_file.write(b"Temperature = 70 F\n")
        temp_file.close()
        yield temp_file.name
        os.remove(temp_file.name)


# Tests parse temperature function to make sure temperature value is correctly being extracted from markdown files for each walk
def test_parse_temperature(mock_md_file):
    temperature = parse_temperature(mock_md_file)
    assert np.isclose(temperature, 294.26, atol=1e-2)  # Adjust tolerance to account for precision
  # 32°F should convert to 273.15K


# Test list_md_files function
@pytest.fixture
def mock_md_directory():
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create two mock markdown files with the "sinwalk" filter
        with open(os.path.join(temp_dir, "file1_sinwalk.md"), "w") as f:
            f.write("Temperature = 32 F")
        with open(os.path.join(temp_dir, "file2_sinwalk.md"), "w") as f:
            f.write("Temperature = 32 F")
        yield temp_dir


def test_list_md_files(mock_md_directory):
    md_files = list_md_files(mock_md_directory, "sinwalk")
    print (len(md_files))
    assert len(md_files) == 2  # Should return two files matching the filter


# Test read_csv_data function with a mock CSV file
@pytest.fixture
def mock_csv_file():
    data = {
        'Time (s)': [0.0, 1.0, 2.0],
        'Latitude (°)': [40.7128, 40.7129, 40.7130],
        'Longitude (°)': [-74.0060, -74.0061, -74.0062],
    }
    df = pd.DataFrame(data)
    with tempfile.NamedTemporaryFile(delete=False, mode='w', newline='') as temp_csv_file:
        df.to_csv(temp_csv_file, index=False)
        temp_csv_file.close()
        yield temp_csv_file.name
        os.remove(temp_csv_file.name)


def test_read_csv_data(mock_csv_file):
    time, latitude, longitude = read_csv_data(mock_csv_file)
    assert len(time) == 3
    assert time[0] == 0.0
    assert latitude[1] == 40.7129


# Test FFT computation with mock data
def test_compute_fft():
    time = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    signal = np.sin(time) - np.mean(np.sin(time))  # Zero-center the signal
    freqs, fft_vals = compute_fft(signal, time)
    assert len(freqs) == len(fft_vals)
    assert np.isclose(fft_vals[0], 0, atol=1e-1)  # Check if DC component is small

# Test calculate_frequency_axis function
# Test for calculate_frequency_axis function
def test_calculate_frequency_axis():
    # Test case with a simple displacement signal (sinusoidal signal)
    time = [0.0, 1.0, 2.0, 3.0, 4.0]  # Sample time array (5 time steps)
    displacement = [0.0, 1.0, 0.0, -1.0, 0.0]  # Simple sinusoidal signal for displacement
    
    # Compute the frequency axis using the function
    freqs_100m = calculate_frequency_axis(displacement, time)
    
    # Check that the length of the frequency axis is half of the displacement signal length
    assert len(freqs_100m) == len(displacement) // 2, f"Expected {len(displacement) // 2} frequencies, got {len(freqs_100m)}"
    
    # Check that the first frequency is zero (DC component)
    assert freqs_100m[0] == 0, f"Expected first frequency to be 0, got {freqs_100m[0]}"
    
    # Check that all frequencies are non-negative (they should be positive frequencies)
    assert all(f >= 0 for f in freqs_100m), f"Frequencies contain negative values: {freqs_100m}"
    
    # Check that the spacing between the frequencies is consistent
    frequency_spacing = freqs_100m[1] - freqs_100m[0]
    assert all(abs(freqs_100m[i+1] - freqs_100m[i] - frequency_spacing) < 1e-10 for i in range(len(freqs_100m)-1)), "Frequencies are not evenly spaced"
    
    