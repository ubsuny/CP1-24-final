"""
This module contains functions for processing and analyzing 
GPS motion data as part of the CP1-24 final project.

The module provides functionality for:
- Temperature conversion between Fahrenheit and Kelvin
- Parsing markdown files for temperature data 
- File listing and filtering for experiment data
- Non-linear fitting capabilities
- FFT analysis tools
- Frequency axis calculations

All functions include comprehensive unit tests and follow pylint standards.
"""

import os
import re
import math as mt
import numpy as np

def fahrenheit_to_kelvin(temp_f):
    """
    Converts temperature from Fahrenheit to Kelvin.
    
    The conversion uses the standard formula:
    K = (°F - 32) × 5/9 + 273.15

    Input:
        temp_f (float): Temperature in Fahrenheit

    Output:
        float: Temperature in Kelvin

    Exceptions:
        TypeError: If input is not a number
        ValueError: If temperature is below absolute zero (-459.67°F)
    """
    if not isinstance(temp_f, (int, float)):
        raise TypeError("Temperature must be a number")

    # Check if temperature is below absolute zero
    if temp_f < -459.67:
        raise ValueError("Temperature cannot be below absolute zero (-459.67°F)")

    return (temp_f - 32) * 5/9 + 273.15

def parse_temperature_from_markdown(filepath):
    """
    Extracts temperature in Fahrenheit from a markdown file.
    
    Expects markdown file to contain a line with temperature in format of
    'Environment temperature: XX°F'

    Input:
        filepath (str): Path to markdown file

    Output:
        float: Temperature value in Fahrenheit

    Exceptions:
        FileNotFoundError: If file doesn't exist
        ValueError: If temperature cannot be found or parsed
    """
    try:
        with open(filepath, 'r', encoding='utf-8') as file:
            content = file.read()

        # Look for temperature using various possible formats
        patterns = [
            r'Environment temperature:\s*(-?\d+\.?\d*)°?F',
            r'Temperature:\s*(-?\d+\.?\d*)°?F',
            r'Temp:\s*(-?\d+\.?\d*)°?F'
        ]

        for pattern in patterns:
            match = re.search(pattern, content)
            if match:
                temp = float(match.group(1))
                return temp

        raise ValueError("No temperature found in markdown file")

    except FileNotFoundError as exc:
        raise FileNotFoundError(f"Could not find markdown file: {filepath}") from exc
    except ValueError as e:
        raise ValueError(f"Error parsing temperature: {str(e)}") from e

def list_markdown_files(directory, pattern):
    """
    Lists markdown files in directory that match a given pattern.

    Uses os library to find all .md files in specified directory that contain
    the pattern in their filename. Case-sensitive matching.

    Input:
        directory (str): Path to search for markdown files
        pattern (str): Pattern to match in filenames (e.g. 'sinewalk')

    Output:
        list: List of matching markdown filenames with full paths

    Exceptions:
        FileNotFoundError: If directory doesn't exist
        ValueError: If pattern is empty
    """

    if not os.path.exists(directory):
        raise FileNotFoundError(f"Directory not found: {directory}")

    if not pattern:
        raise ValueError("Search pattern cannot be empty")

    markdown_files = []

    for filename in os.listdir(directory):
        if filename.endswith('.md') and pattern in filename:
            full_path = os.path.join(directory, filename)
            markdown_files.append(full_path)

    return sorted(markdown_files)  # Sort for consistent ordering

def fit_nonlinear(data_x, data_y, n: int):
    """
    Perform non-linear fitting to data using a sine wave model.

    Input:
        data_x: List of x values (e.g., GPS-displacement x-axis data)
        data_y: List of y values (e.g., GPS-displacement y-axis data)
        n: Specifies the step size as 2^n.

    Output:
        Best-fit parameters (amplitude, frequency, phi, constant).
    """

    # Calculate step size based on 2^n
    step_size = 1 / (2 ** n)

    # Check for empty data
    if len(data_x) == 0 or len(data_y) == 0:
        raise ValueError("Input data cannot be empty")

    # Estimate constants
    data_stats = {
        'mean': sum(data_y) / len(data_y),
        'range': max(data_y) - min(data_y),
        'span': data_x[-1] - data_x[0]
    }

    # Estimate frequency from zero crossings
    zero_crossings = sum(
        1 for i in range(1, len(data_y))
        if (data_y[i-1] - data_stats['mean']) * (data_y[i] - data_stats['mean']) < 0
    )

    # Define parameter ranges
    param_ranges = {
        'amplitude': (0, data_stats['span']),
        'frequency': (0, zero_crossings / (2 * data_stats['span']) or 1 / data_stats['span'] * 1.5),
        'phi': (-mt.pi, mt.pi),
        'constant': (data_stats['mean'] - abs(data_stats['mean']), 
                     data_stats['mean'] + abs(data_stats['mean']))
    }

    # Generate parameter values based on step size
    # Step size is int((1 / step_size) + 1), but I ran out of variables for linting.
    param_values = {
        key: [
            start + i * step_size * (end - start)
            for i in range(int((1 / step_size) + 1))
        ] for key, (start, end) in param_ranges.items()
    }

    min_error = float('inf')
    best_params = {}

    # Brace yourself for a horrible nested for loop, since I'm trying to get pure python.
    # Search over parameter space
    for amp in param_values['amplitude']:
        for freq in param_values['frequency']:
            for phi in param_values['phi']:
                for const in param_values['constant']:
                    # Compute residuals
                    error = sum(
                        (y - (amp * mt.sin(2 * mt.pi * freq * x + phi) + const)) ** 2
                        for x, y in zip(data_x, data_y)
                    )
                    # Update best parameters if error is smaller
                    if error < min_error:
                        min_error = error
                        best_params = {
                            'amplitude': amp,
                            'frequency': freq,
                            'phi': phi,
                            'constant': const
                        }

    return best_params

def fft_wrapper(data, positions):
    """
    Wrapper for numpy FFT with data validation and spatial frequency axis generation.

    Validates that input data points are equidistant in space before performing FFT.
    Handles conversion to proper numpy arrays, calculates appropriate spatial frequency axis,
    and centers the frequency spectrum around zero.

    Input:
        data (array-like): Signal amplitude data to transform
        positions (array-like): Spatial positions corresponding to data samples.
                              Must be equidistant.

    Output:
        tuple: (frequencies, fft_result)
            - frequencies: Array of spatial frequency points (1/distance), centered around 0
            - fft_result: FFT of input data, shifted to match frequency axis

    Exceptions:
        TypeError: If data or positions are not numeric arrays
        ValueError: If positions are not equidistant within tolerance (1e-5)
    """

    if not isinstance(data, (list, np.ndarray)) or not isinstance(positions, (list, np.ndarray)):
        raise TypeError("Data and positions must be arrays")

    # Convert to numpy arrays if needed
    data = np.array(data)
    positions = np.array(positions)

    # Check for equidistant positions
    dx = np.diff(positions)
    if not np.allclose(dx, dx[0], rtol=1e-5):
        raise ValueError("Data points must be equidistant in space")

    # Calculate spatial sampling rate (1/distance)
    sample_rate = 1.0 / dx[0]

    # Perform FFT and get spatial frequencies
    fft_result = np.fft.fft(data)
    freqs = np.fft.fftfreq(len(data), d=1/sample_rate)

    # Shift both frequencies and FFT result to center the spectrum
    freqs = np.fft.fftshift(freqs)
    fft_result = np.fft.fftshift(fft_result)

    return freqs, fft_result

def ifft_wrapper(fft_result):
    """
    Wrapper for numpy inverse FFT.

    Input:
        fft_result (array-like): FFT data to inverse transform

    Output:
        array: Inverse FFT of input data

    Exceptions:
        TypeError: If input is not a numeric array
    """

    if not isinstance(fft_result, (list, np.ndarray)):
        raise TypeError("FFT result must be an array")

    # Used .real to only focus on real component, as FFT may introduce small imaginary parts
    return np.fft.ifft(np.fft.ifftshift(fft_result)).real

def calculate_frequency_axis(sample_rate, n_points):
    """
    Calculate frequency axis for FFT in pure Python.

    Input:
        sample_rate (float): Sampling rate in oscilations per meter
        n_points (int): Number of points in the signal (must be positive integer)

    Output:
        list: List of frequencies centered around 0

    Exceptions:
        TypeError: If n_points is not an integer
        ValueError: If n_points <= 0 or sample_rate <= 0
    """
    # Input validation
    if not isinstance(n_points, int):
        raise TypeError("n_points must be an integer")
    if n_points <= 0:
        raise ValueError("n_points must be positive")
    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive")

    # Calculate frequency step
    df = sample_rate / n_points

    # Generate frequency points centered around 0
    freqs = []
    if n_points % 2 == 0:
        for i in range(n_points):
            if i < n_points // 2:
                freq = i * df
            else:
                freq = (i - n_points) * df
            freqs.append(freq)
    else:
        for i in range(n_points):
            if i <= n_points // 2:
                freq = i * df
            else:
                freq = (i - n_points) * df
            freqs.append(freq)

    return freqs

def rotate_to_horizontal(x, y):
    """
    Rotates data to align with x-axis based on start and end points.
    
    Input:
        x (list/array): x coordinates
        y (list/array): y coordinates
    
    Output:
        x_rot, y_rot: Rotated coordinates with main direction along x-axis
    """
    # Validate inputs
    if not (isinstance(x, (list, np.ndarray)) and isinstance(y, (list, np.ndarray))):
        raise TypeError("Inputs x and y must be lists or numpy arrays")
    if (not all(isinstance(i, (int, float)) for i in x) or
            not all(isinstance(i, (int, float)) for i in y)):
        raise TypeError("All elements in x and y must be numeric")

    # Convert to numpy arrays if needed
    x = np.array(x)
    y = np.array(y)

    # Find angle from start to end point
    dx = x[-1] - x[0]
    # Convert to numpy arrays if needed
    x = np.array(x)
    y = np.array(y)

    # Find angle from start to end point
    dx = x[-1] - x[0]
    dy = y[-1] - y[0]
    angle = np.arctan2(dy, dx)

    # Add special case for vertical lines
    if np.allclose(dx, 0):
        x_rot = np.zeros_like(x)
        y_rot = y - y[0]  # Translate to start at origin
        return x_rot, y_rot

    # Calculate original spacing
    original_length = x[-1] - x[0]

    # Create rotation matrix
    rot_matrix = np.array([
        [np.cos(-angle), -np.sin(-angle)],
        [np.sin(-angle), np.cos(-angle)]
    ])

    # Center data at origin and rotate
    x_centered = x - x[0]
    y_centered = y - y[0]
    xy = np.vstack((x_centered, y_centered))
    x_rot, y_rot = rot_matrix @ xy

    # Scale x values to match original spacing
    x_rot = x_rot * (original_length / (x_rot[-1] - x_rot[0]))

    if y_rot[0] > y_rot[1] and y_rot[1] > y_rot[2]:
        y_rot = -y_rot

    return x_rot, y_rot
