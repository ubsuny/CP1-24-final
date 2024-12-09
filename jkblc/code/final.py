"""Algorithms file"""

import os
import re
from datetime import datetime
from scipy.optimize import curve_fit
import numpy as np


def fahrenheit_to_kelvin(fahrenheit):
    """
    Convert Fahrenheit to Kelvin.

    Parameters:
    fahrenheit (float): Temperature in Fahrenheit.

    Returns:
    float: Temperature in Kelvin.
    """
    return (fahrenheit - 32) * 5 / 9 + 273.15


def parse_temperature_from_markdown(file_path):
    """
    Extract the temperature in Fahrenheit from a markdown file.

    Parameters:
    file_path (str): Path to the markdown file.

    Returns:
    float: Temperature in Fahrenheit.
    """
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()
        match = re.search(r"Environment temperature: (\d+\.?\d*) F", content)
        if match:
            return float(match.group(1))
    raise ValueError("Temperature not found in file.")


def list_markdown_files(folder_path, keyword="sinewalk"):
    """
    List all markdown files containing a specific keyword.

    Parameters:
    folder_path (str): Path to the folder containing files.
    keyword (str): Keyword to filter filenames.

    Returns:
    list: List of matching markdown file paths.
    """
    return [
        os.path.join(folder_path, f)
        for f in os.listdir(folder_path)
        if f.endswith(".md") and keyword in f
    ]


def sine_function(x, amplitude, frequency, phase, offset):
    """
    Sine wave function for non-linear fitting.

    Parameters:
    x (array): Input values (x-axis).
    amplitude (float): Sine wave amplitude.
    frequency (float): Sine wave frequency.
    phase (float): Phase shift of the sine wave.
    offset (float): Vertical offset.

    Returns:
    array: Calculated y values.
    """
    return amplitude * np.sin(frequency * x + phase) + offset


def non_linear_fit(x_data, y_data, initial_guess, steps=1000):
    """
    Perform non-linear fitting using curve_fit from SciPy.

    Parameters:
    x_data (array): x-axis data.
    y_data (array): y-axis data.
    initial_guess (tuple): Initial guess for parameters.
    steps (int): Number of steps for fitting.

    Returns:
    tuple: Optimal parameters and covariance.
    """
    popt, pcov = curve_fit(
        sine_function, x_data, y_data, p0=initial_guess, maxfev=steps
    )
    return popt, pcov


def fft_with_check(x_data, y_data):
    """
    Perform FFT with non-equidistant data checking.

    Parameters:
    x_data (array): x-axis data.
    y_data (array): y-axis data.

    Returns:
    tuple: FFT of y_data and corresponding frequencies.
    """
    dx = np.diff(x_data)
    if not np.allclose(dx, dx[0]):
        raise ValueError("Data is not equidistant.")

    fft_y = np.fft.fft(y_data)
    freq = np.fft.fftfreq(len(y_data), d=dx[0])
    return fft_y, freq


def inverse_fft(fft_y):
    """
    Perform inverse FFT.

    Parameters:
    fft_y (array): FFT-transformed data.

    Returns:
    array: Inverse FFT result.
    """
    return np.fft.ifft(fft_y).real


def calculate_frequency_axis(n_points, dx):
    """
    Calculate the frequency axis in units of 1/100m.

    Parameters:
    n_points (int): Number of points.
    dx (float): Spacing between points.

    Returns:
    array: Frequency axis values.
    """
    return [(i / (dx * n_points)) * 100 for i in range(n_points // 2)]
