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

import re

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

def calculate_frequency_axis(sample_rate, num_points):
    """
    Calculates frequency axis points for FFT analysis without using numpy.
    
    For N points, creates a frequency axis from -fs/2 to +fs/2,
    where fs is the sampling frequency.

    Parameters:
        sample_rate (float): Sampling frequency in Hz
        num_points (int): Number of data points

    Returns:
        list: Frequency points in Hz

    Raises:
        TypeError: If inputs are not numeric
        ValueError: If sample_rate <= 0 or num_points <= 0
    """
    if not isinstance(sample_rate, (int, float)) or not isinstance(num_points, int):
        raise TypeError("Sample rate must be numeric and num_points must be integer")
    
    if sample_rate <= 0 or num_points <= 0:
        raise ValueError("Sample rate and num_points must be positive")

    freq_step = sample_rate / num_points
    num_freqs = num_points
    
    # Generate frequencies from 0 to Nyquist frequency
    freqs = []
    for i in range(num_freqs):
        if i <= num_freqs // 2:
            freq = i * freq_step
        else:
            freq = (i - num_freqs) * freq_step
        freqs.append(freq)

    return freqs

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
                temp = int(match.group(1))
                return temp
                
        raise ValueError("No temperature found in markdown file")
        
    except FileNotFoundError:
        raise FileNotFoundError(f"Could not find markdown file: {filepath}")
    except ValueError as e:
        raise ValueError(f"Error parsing temperature: {str(e)}")
    
if __name__ == "__main__":
    path = 'SchrodingersStruggle/data/final/CJY001_sinewalk.md'
    print(parse_temperature_from_markdown(path))
    print(fahrenheit_to_kelvin(32))
    print(calculate_frequency_axis(100, 8))
