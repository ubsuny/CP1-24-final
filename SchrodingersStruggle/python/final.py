"""
This module contains functions for processing and analyzing GPS motion data as part of the CP1-24 final project.

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
import numpy as np

def fahrenheit_to_kelvin(temp_f):
    """
    Converts temperature from Fahrenheit to Kelvin.
    
    The conversion uses the standard formula:
    K = (°F - 32) × 5/9 + 273.15

    Parameters:
        temp_f (float): Temperature in Fahrenheit

    Returns:
        float: Temperature in Kelvin

    Raises:
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
    
    Expects markdown file to contain a line with temperature in format:
    'Environment temperature: XX°F' or similar

    Parameters:
        filepath (str): Path to markdown file

    Returns:
        float: Temperature value in Fahrenheit

    Raises:
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
        
    except FileNotFoundError:
        raise FileNotFoundError(f"Could not find markdown file: {filepath}")
    except ValueError as e:
        raise ValueError(f"Error parsing temperature: {str(e)}")
    
def list_markdown_files(directory, pattern):
    """
    Lists markdown files in directory that match a given pattern.

    Uses os library to find all .md files in specified directory that contain
    the pattern in their filename. Case-sensitive matching.

    Parameters:
        directory (str): Path to search for markdown files
        pattern (str): Pattern to match in filenames (e.g. 'sinewalk')

    Returns:
        list: List of matching markdown filenames with full paths

    Raises:
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

def fft_wrapper(data, timesteps):
    """
    Wrapper for numpy FFT with data validation and frequency axis generation.

    Validates that input data points are equidistant in time before performing FFT.
    Handles conversion to proper numpy arrays, calculates appropriate frequency axis,
    and centers the frequency spectrum around zero.

    Parameters:
        data (array-like): Signal amplitude data to transform
        timesteps (array-like): Time points corresponding to data samples.
                                Must be equidistant.

    Returns:
        tuple: (frequencies, fft_result)
            - frequencies: Array of frequency points, centered around 0
            - fft_result: FFT of input data, shifted to match frequency axis

    Raises:
        TypeError: If data or timesteps are not numeric arrays
        ValueError: If timesteps are not equidistant within tolerance (1e-5)
    """
    
    if not isinstance(data, (list, np.ndarray)) or not isinstance(timesteps, (list, np.ndarray)):
        raise TypeError("Data and timesteps must be arrays")
        
    # Convert to numpy arrays if needed
    data = np.array(data)
    timesteps = np.array(timesteps)
    
    # Check for equidistant timesteps
    dt = np.diff(timesteps)
    if not np.allclose(dt, dt[0], rtol=1e-5):
        raise ValueError("Data points must be equidistant in time")
    
    # Calculate sample rate
    sample_rate = 1.0 / dt[0]
    
    # Perform FFT and get frequencies
    fft_result = np.fft.fft(data)
    freqs = np.fft.fftfreq(len(data), d=1/sample_rate)
    
    # Shift both frequencies and FFT result to center the spectrum
    freqs = np.fft.fftshift(freqs)
    fft_result = np.fft.fftshift(fft_result)
    
    return freqs, fft_result

def ifft_wrapper(fft_result):
    """
    Wrapper for numpy inverse FFT.
    
    Parameters:
        fft_result (array-like): FFT data to inverse transform

    Returns:
        array: Inverse FFT of input data

    Raises:
        TypeError: If input is not a numeric array
    """
    import numpy as np
    
    if not isinstance(fft_result, (list, np.ndarray)):
        raise TypeError("FFT result must be an array")
        
    # Used .real to only focus on real component, as FFT may introduce small imaginary parts
    return np.fft.ifft(np.fft.ifftshift(fft_result)).real

if __name__ == "__main__":
    path = 'SchrodingersStruggle/data/final/CJY001_sinewalk.md'
    print(parse_temperature_from_markdown(path))
    print(fahrenheit_to_kelvin(32))
    print(list_markdown_files('SchrodingersStruggle/data/final', 'sinewalk'))
