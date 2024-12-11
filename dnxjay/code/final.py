"""
This module contains functions for the CP1-24 final project, including:
- Temperature conversion (Fahrenheit to Kelvin)
- Parsing temperature from markdown files
- Listing markdown files
- Non-linear sine wave fitting
- FFT wrapper and frequency axis calculations
"""

import os
import re
import numpy as np
from scipy.interpolate import interp1d


def fahrenheit_to_kelvin(temp_f):
    """
    Converts a temperature from Fahrenheit to Kelvin.
    
    Args:
        temp_f (float): Temperature in Fahrenheit.

    Returns:
        float: Temperature in Kelvin.
    """
    return (temp_f - 32) * 5 / 9 + 273.15


def parse_temperature_from_markdown(filepath):
    """
    Parses the temperature (in Fahrenheit) from a markdown file.
    """
    with open(filepath, 'r', encoding='utf-8') as file:
        content = file.read()
    # Match various formats of temperature
    match = re.search(r"\*\*Temperature.*?\*\*.*?(\d+)", content)
    if match:
        return float(match.group(1))  # Convert matched value to float
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
    Computes a sine wave value for an array-like input x.

    Args:
        x (array-like): Input values (e.g., time or spatial coordinates).
        amplitude (float): Amplitude of the sine wave.
        frequency (float): Frequency of the sine wave.
        phase (float): Phase shift of the sine wave.

    Returns:
        numpy.ndarray: Computed sine wave values for the input.
    """
    return amplitude * np.sin(2 * np.pi * frequency * x + phase)


def fft_with_check(data):
    """
    Computes the FFT and resamples data if necessary to ensure equidistant intervals.

    Args:
        data (array-like): Input data.

    Returns:
        numpy.ndarray: FFT result.
    """
    # Check intervals
    intervals = np.diff(data)
    if not np.allclose(intervals, intervals[0]):
        # Resample the data to make it equidistant
        original_x = np.arange(len(data))
        new_x = np.linspace(original_x[0], original_x[-1], len(original_x))
        interpolator = interp1d(original_x, data, kind='linear')
        data = interpolator(new_x)
    return np.fft.fft(data)


def calculate_frequency_axis(data_length, sampling_interval):
    """
    Calculates the frequency axis in useful units.

    Args:
        data_length (int): Length of the data.
        sampling_interval (float): Sampling interval.

    Returns:
        numpy.ndarray: Frequency axis values.
    """
    return np.fft.fftfreq(data_length, d=sampling_interval)[:data_length // 2]
