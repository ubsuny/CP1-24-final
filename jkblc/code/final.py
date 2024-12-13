"""Algorithms file"""

import os
from typing import Tuple
import numpy as np


# Temperature Conversion Function
def fahrenheit_to_kelvin(fahrenheit):
    """
    Convert Fahrenheit to Kelvin.

    Parameters:
    fahrenheit (float): Temperature in Fahrenheit.

    Returns:
    float: Temperature in Kelvin.
    """
    return (fahrenheit - 32) * 5 / 9 + 273.15


# Markdown Temperature Parser
def parse_temperature_from_markdown(file_path):
    """
    Extracts the environment temperature from a Markdown file.

    Parameters:
    file_path (str): Path to the Markdown file.

    Returns:
    float: Extracted temperature in Fahrenheit.

    Raises:
    ValueError: If the temperature is not found or formatted incorrectly.
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            for line in file:
                if "Environment Temperature:" in line:
                    clean_line = line.replace("*", "").replace("**", "").strip()
                    temp_str = clean_line.split(":")[1].strip().replace("F", "").strip()
                    return float(temp_str)
    except (IndexError, ValueError) as exc:
        raise ValueError(f"Invalid or missing temperature in file: {file_path}") from exc

    raise ValueError(f"Temperature not found in file: {file_path}")


# List Markdown Files
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


# Sine Wave Function for Fitting
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


# Non-Linear Fitting Using Gradient Descent
def non_linear_fit(x_data, y_data, initial_guess, steps=20000, lr=0.0001):
    """
    Custom non-linear fitting using gradient descent.

    Parameters:
    x_data (array): x-axis data.
    y_data (array): y-axis data.
    initial_guess (tuple): Initial guess for parameters.
    steps (int): Number of gradient descent steps.
    lr (float): Learning rate for parameter updates.

    Returns:
    tuple: Optimized parameters (amplitude, frequency, phase, offset).
    """
    amplitude, frequency, phase, offset = initial_guess

    for _ in range(steps):
        y_pred = sine_function(x_data, amplitude, frequency, phase, offset)
        error = y_data - y_pred

        # Calculate gradients
        grad_amplitude = -2 * np.sum(error * np.sin(frequency * x_data + phase))
        grad_frequency = -2 * np.sum(error * amplitude * x_data * np.cos(frequency * x_data + phase))
        grad_phase = -2 * np.sum(error * amplitude * np.cos(frequency * x_data + phase))
        grad_offset = -2 * np.sum(error)

        # Update parameters
        amplitude -= lr * grad_amplitude
        frequency -= lr * grad_frequency
        phase -= lr * grad_phase
        offset -= lr * grad_offset

    return amplitude, frequency, phase, offset


# Perform FFT with Equidistant Data Check
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


# Perform Inverse FFT
def inverse_fft(fft_y):
    """
    Perform inverse FFT.

    Parameters:
    fft_y (array): FFT-transformed data.

    Returns:
    array: Inverse FFT result.
    """
    return np.fft.ifft(fft_y).real


# Calculate Frequency Axis
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
