import os
import re
from math import sin, pi


def fahrenheit_to_kelvin(temp_f):
    """
    Converts a temperature from Fahrenheit to Kelvin.
    
    Args:
        temp_f (float): Temperature in Fahrenheit.

    Returns:
        float: Temperature in Kelvin.
    """
    return (temp_f - 32) * 5 / 9 + 273.15


def parse_temperature_from_markdown(file_path):
    """
    Parses the temperature (Fahrenheit) from a markdown file.

    Args:
        file_path (str): Path to the markdown file.

    Returns:
        float: Temperature in Fahrenheit.
    """
    with open(file_path, "r") as file:
        for line in file:
            if "Temperature (Outdoor):" in line:
                match = re.search(r"(\d+)", line)
                if match:
                    return float(match.group(1))
    raise ValueError("Temperature not found in the markdown file.")


def list_markdown_files(directory, keyword="sinewalk"):
    """
    Lists markdown files in a directory that contain a specific keyword.

    Args:
        directory (str): Path to the directory.
        keyword (str): Keyword to filter filenames.

    Returns:
        list: List of filenames containing the keyword.
    """
    return [f for f in os.listdir(directory) if f.endswith(".md") and keyword in f]


def sine_wave(x, amplitude, frequency, phase):
    """
    Computes a sine wave value at a given x.
    
    Args:
        x (float): Input value.
        amplitude (float): Amplitude of the sine wave.
        frequency (float): Frequency of the sine wave.
        phase (float): Phase shift of the sine wave.

    Returns:
        float: Computed sine wave value.
    """
    return amplitude * sin(2 * pi * frequency * x + phase)


def fft_with_check(data, sampling_interval):
    """
    Computes the FFT and checks for non-equidistant data.

    Args:
        data (list of float): Time series data.
        sampling_interval (float): Sampling interval.

    Returns:
        list: FFT result.
    """
    import numpy as np

    # Check if data is equidistant
    intervals = [data[i + 1] - data[i] for i in range(len(data) - 1)]
    if not all(abs(interval - intervals[0]) < 1e-6 for interval in intervals):
        raise ValueError("Data is not equidistant.")

    return np.fft.fft(data)


def calculate_frequency_axis(data_length, sampling_interval):
    """
    Calculates the frequency axis in useful units.

    Args:
        data_length (int): Length of the data.
        sampling_interval (float): Sampling interval.

    Returns:
        list: Frequency axis values.
    """
    return [(i / (data_length * sampling_interval)) for i in range(data_length // 2)]
