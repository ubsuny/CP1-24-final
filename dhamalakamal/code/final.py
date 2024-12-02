"""
final.py

This module contains a collection of utility functions for data analysis and experiment processing. 
It is designed to be used with experiment data stored in CSV and Markdown files, 
providing functionality for temperature conversion, parsing experiment metadata, 
file handling, non-linear fitting, FFT analysis, and frequency axis calculation.

Functions:
- fahrenheit_to_kelvin(fahrenheit: float) -> float:
    Converts a temperature from Fahrenheit to Kelvin.
- parse_temperature_from_md(md_filepath: str) -> float:
    Extracts the temperature in Fahrenheit from a Markdown file.
- list_files(directory: str, filter_keyword: str) -> List[str]:
    Lists files in a directory filtered by a keyword.
- nonlinear_fit(data: List[float], step: int, func: Callable[[float], float]) -> List[float]:
    Applies a non-linear fitting function to data with a specific step size.
- fft_wrapper(data: np.ndarray) -> np.ndarray:
    Computes the FFT of equidistant data.
- inverse_fft_wrapper(data: np.ndarray) -> np.ndarray:
    Computes the inverse FFT.
- check_equidistant(data: np.ndarray) -> bool:
    Checks if the data points are equidistant.
- calculate_frequency_axis(data_length: int, sampling_rate: float) -> List[float]:
    Calculates the frequency axis in Hz.

Author:
- Kamal Dhamala
"""

import os
from typing import Callable, List
import numpy as np

# Fahrenheit to Kelvin Converter
def fahrenheit_to_kelvin(fahrenheit: float) -> float:
    """
    Converts a temperature from Fahrenheit to Kelvin.

    Parameters:
        fahrenheit: Temperature in Fahrenheit.

    Returns: Temperature in Kelvin.
    """
    return (fahrenheit - 32) * 5 / 9 + 273.15

# Markdown Temperature Parser
def parse_temperature_from_md(md_filepath: str) -> float:
    """
    Parses the temperature in Fahrenheit from a Markdown file.

    Args:
        md_filepath: Path to the Markdown file.

    Returns: Temperature in Fahrenheit.

    Raises:
        ValueError: If temperature is not found.
    """
    with open(md_filepath, "r", encoding='utf-8') as file:
        content = file.read()

    # Find the line that contains "Temperature:"
    lines = content.splitlines()
    for line in lines:
        if "Temperature:" in line:
            # Extract the part after "Temperature:" and clean it
            temp_part = line.split("Temperature:")[1].strip()

            # Remove LaTeX formatting like $39^\circ\ \text{F}$
            # Extract the numeric part, which should be the first sequence of digits
            temp_value = ''.join(filter(str.isdigit, temp_part))

            if temp_value:
                return float(temp_value)

    raise ValueError(f"Temperature not found in {md_filepath}")


# Filename lister
def list_md_files(directory: str, filter_keyword: str) -> List[str]:
    """
    Lists files in a directory filtered by a keyword.

    Parameters:
        directory: Path to the directory.
        filter_keyword: Keyword to filter filenames.

    Returns:
        List: List of filtered filenames.
    """
    return [
        f for f in os.listdir(directory)
        if filter_keyword in f and f.endswith(".md")
    ]

# Non-linear fitting
def nonlinear_fit(data: List[float], step: int, func: Callable[[float], float]) -> List[float]:
    """
    Applies a non-linear fitting function to data with a specific step size.

    Parameters:
        data (List[float]): Input data points.
        step (int): Step size, must be a power of 2.
        func (Callable[[float], float]): Function to apply.

    Returns:
        List[float]: Processed data.
    """
    if not (step > 0 and (step & (step - 1)) == 0):  # Power of 2 check
        raise ValueError("Step must be a power of 2.")
    return [func(data[i]) for i in range(0, len(data), step)]

# Numpy Wrapper for FFT
def fft_wrapper(data: np.ndarray) -> np.ndarray:
    """
    Computes the FFT of equidistant data.

    Parameters:
        data (np.ndarray): Input data array.

    Returns:
        np.ndarray: FFT result.
    """
    return np.fft.fft(data)

def inverse_fft_wrapper(data: np.ndarray) -> np.ndarray:
    """
    Computes the inverse FFT.

    Parameters:
        data (np.ndarray): Input FFT data array.

    Returns:
        np.ndarray: Inverse FFT result.
    """
    return np.fft.ifft(data)

def check_equidistant(data: np.ndarray) -> bool:
    """
    Checks if the data points are equidistant.

    Parameters:
        data (np.ndarray): Array of data points.

    Returns:
        bool: True if equidistant, False otherwise.
    """
    differences = np.diff(data)
    return np.allclose(differences, differences[0])

# Pure Python Frequency Axis
def calculate_frequency_axis(data_length: int, sampling_rate: float) -> List[float]:
    """
    Calculates the frequency axis in Hz.

    Parameters:
        data_length: Length of the data.
        sampling_rate: Sampling rate in Hz.

    Returns: List of Frequency axis in Hz.
    """
    return [(i / (data_length * (1 / sampling_rate))) for i in range(data_length // 2 + 1)]
