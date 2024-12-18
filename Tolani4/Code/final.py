import os
import numpy as np

def fahrenheit_to_kelvin(temp_f):
    """Convert temperature from Fahrenheit to Kelvin."""
    return (temp_f - 32) * 5/9 + 273.15

def parse_temperature(markdown_file):
    """Extract temperature in Fahrenheit from a markdown file."""
    with open(markdown_file, "r") as file:
        for line in file:
            if "Environment temperature" in line:
                temp_f = float(line.split(":")[1].strip().replace("F", ""))
                return temp_f
    return None

def list_markdown_files(directory, filter_str):
    """List markdown files in a directory with a specific filter."""
    return [f for f in os.listdir(directory) if f.endswith(".md") and filter_str in f]

def nonlinear_fit(x, y, step=2):
    """Non-linear fitting using a simple polynomial (step^n)."""
    return np.polyfit(x, y, step)

def fft_numpy(data):
    """Compute FFT and check for non-equidistant data."""
    if not np.allclose(np.diff(data[:, 0]), np.diff(data[:, 0])[0]):
        raise ValueError("Data is not equidistant.")
    fft_result = np.fft.fft(data[:, 1])
    return fft_result

def frequency_axis(data_length, dx):
    """Calculate frequency axis in 1/100m (pure Python)."""
    freq = [(i / (dx * data_length)) * 100 for i in range(data_length // 2)]
    return freq