"""
Final.py
This module contains some importable implementable functions to be used with experiment data stored in CSV and Markdown files. 
Functions:
-function that converts Fahrenheit to Kelvin
-parser that reads out the temperature of one markdown file
-filename lister that generates programmatically (using the python os library) a list of your markdown file based on a filename filter.
-non-linear fitting in pure python which includes the functionality to specify the step number of 2^n
-numpy wrapper for fft, inverse fft, including functionality that checks for non-equidistant data.
=pure python (no numpy) to calculate the frequency axis in useful units.
"""

import os
import numpy as np
import math
from typing import List, Tuple

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
    with open(file_path, 'r') as file:
        for line in file:
            if line.startswith('Temperature:'):
                # Extracting the temperature value after 'Temperature:'
                temp_value = float(line.split(':')[1].strip())
                return temp_value
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

# Non-linear fitting in pure Python (using least squares)
def non_linear_fit(func, x_data: List[float], y_data: List[float], initial_params: List[float], steps: int = 1000, tol: float = 1e-6) -> List[float]:
    """
    Performs non-linear fitting using a least squares method with a given function.

    Parameters:
        func (callable): The model function to fit (takes parameters and x_data).
        x_data (List[float]): The independent data points.
        y_data (List[float]): The dependent data points.
        initial_params (List[float]): Initial guess for the parameters.
        steps (int): Number of iterations for the fitting process (default is 1000).
        tol (float): Tolerance level to stop the fitting process (default is 1e-6).

    Returns:
        List[float]: Fitted parameters.
    """
    params = initial_params
    for _ in range(steps):
        residuals = [y - func(x, *params) for x, y in zip(x_data, y_data)]
        jacobian = []
        for i in range(len(params)):
            jacobian.append([2 * res * (x ** i) for res, x in zip(residuals, x_data)])
        jacobian_matrix = np.array(jacobian).T
        delta_params = np.linalg.pinv(jacobian_matrix).dot(residuals)
        params = params - delta_params
        if np.linalg.norm(delta_params) < tol:
            break
    return params

# Numpy wrapper for FFT, Inverse FFT with checks for non-equidistant data
def fft_wrapper(data: np.ndarray, inverse: bool = False) -> np.ndarray:
    """
    Computes the FFT or Inverse FFT for equidistant or non-equidistant data.

    Parameters:
        data (np.ndarray): Input data array (must be 1D).
        inverse (bool): If True, performs the inverse FFT. Defaults to False (regular FFT).

    Returns:
        np.ndarray: FFT or Inverse FFT result.

    Raises:
        ValueError: If data is non-equidistant.
    """
    if len(data.shape) != 1:
        raise ValueError("Input data must be a 1D array.")

    # Check for non-equidistant data
    if np.any(np.diff(data[0]) != np.mean(np.diff(data[0]))):
        raise ValueError("Input data is non-equidistant.")

    if inverse:
        return np.fft.ifft(data)
    else:
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
