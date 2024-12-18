"""
Final.py
This module contains some importable implementable functions to be use
with experiment data stored in CSV and Markdown files. 
Functions:
-function that converts Fahrenheit to Kelvin
-parser that reads out the temperature of one markdown file
-filename lister that generates programmatically (using the python os library)
a list of your markdown file based on a filename filter.
-non-linear fitting in pure python which includes the functionality
to specify the step number of 2^n
-numpy wrapper for fft, inverse fft, including functionality that checks for non-equidistant data.
=pure python (no numpy) to calculate the frequency axis in useful units.
"""

import os
import re
from typing import List
from typing import Callable
import numpy as np

# Function to convert Fahrenheit to Kelvin
def fahrenheit_to_kelvin(fahrenheit: float) -> float:
    """
    Converts temperature from Fahrenheit to Kelvin.

    Parameters:
        fahrenheit (float): Temperature in Fahrenheit.

    Returns:
        float: Temperature in Kelvin.
    """
    return (fahrenheit - 32) * 5/9 + 273.15


# Parser to extract temperature from a markdown file

def extract_temperature_from_markdown(file_path: str) -> float:
    """
    Parses a markdown file to extract the temperature value.

    Assumes the markdown file has a line with 'Temperature: <value>' format.

    Parameters:
        file_path (str): Path to the markdown file.

    Returns:
        float: Extracted temperature value.
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            for line in file:
                if 'Temperature:' in line:
                    # Extracting the temperature value after 'Temperature:'
                    temp_part = line.split(':')[1].strip()
                    # Use regex to extract the numeric part
                    temp_value = float(re.findall(r"[-+]?\d*\.\d+|\d+", temp_part)[0])
                    return temp_value
    except FileNotFoundError as exc:
        raise ValueError(f"Temperature not found in the markdown file: {file_path}") from exc

    raise ValueError(f"Temperature not found in the markdown file: {file_path}")


# Function to list markdown files based on a filename filter
def list_markdown_files(directory: str, filter_string: str) -> List[str]:
    """
    Generates a list of markdown files from a specified directory
    that contains a given filter string in their filename.

    Parameters:
        directory (str): Directory to search for markdown files.
        filter_string (str): Substring to filter the filenames.

    Returns:
        List[str]: List of markdown filenames that match the filter.
    """
    files = [f for f in os.listdir(directory) if f.endswith('.md') and filter_string in f]
    return files


def non_linear_fit(
        data: List[float],
        step: int,
        func: Callable[[float], float],
        smooth: bool = False,
        window_size: int = 5
    ) -> List[float]:
    """
    Applies a non-linear fitting function to data with optional smoothing.
    
    Parameters:
        data: Input data points.
        step: Step size, must be a power of 2.
        func: Function to apply.
        smooth: Whether to smooth the data using moving average. Default is False.
        window_size: Window size for smoothing. Default is 5.
    
    Returns: Processed data.
    """
    if not (step > 0 and (step & (step - 1)) == 0):  # Power of 2 check
        raise ValueError("Step must be a power of 2.")

    # Smooth data if enabled
    if smooth:
        if window_size < 1:
            raise ValueError("Window size must be at least 1.")
        # Apply moving average smoothing
        data = np.convolve(data, np.ones(window_size) / window_size, mode="same").tolist()

    # Apply the function to the data (non-linear fitting)
    result = []
    for i in range(0, len(data), step):
        result.append(func(data[i]))
    return result



# Numpy wrapper for FFT, Inverse FFT with checks for non-equidistant data
def fft_wrapper(data: np.ndarray, inverse: bool = False) -> np.ndarray:
    """
    Computes the FFT or Inverse FFT for equidistant or non-equidistant data.
    
    Parameters:
        data (np.ndarray): Input data array (must be 1D).
        inverse (bool): If True, performs the inverse FFT. Defaults to False (regular FFT).
    
    Returns:
        np.ndarray: FFT or Inverse FFT result.
    """
    if len(data.shape) != 1:
        raise ValueError("Input data must be a 1D array.")
    # Check if the data is equidistant
    diff = np.diff(data)
    if not np.allclose(diff, diff[0]):  # Check if all the differences are the same
        raise ValueError("Input data must be equidistant.")

    if inverse:
        return np.fft.ifft(data)
    return np.fft.fft(data)

# Calculate frequency axis in useful units (e.g., Hz) without using numpy
def calculate_frequency_axis(length: int, sample_rate: float) -> List[float]:
    """
    Calculates the frequency axis in Hz for a given data length and sample rate.

    Parameters:
        length (int): Number of samples in the data.
        sample_rate (float): Sample rate in samples per second.

    Returns:
        List[float]: Frequency axis values in Hz.
    """
    return [i * sample_rate / length for i in range(length)]
