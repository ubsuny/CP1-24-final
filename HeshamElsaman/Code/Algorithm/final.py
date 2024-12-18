"""
This module is to provide all the needed functionality for the final exam of CP1-24
"""

import os
import math
import numpy as np
import pandas as pd
import unit_converter_module as ucm
from mercator_projection_functions import xy_on_earth

# Temperature Conversion
def fahrenheit_to_kelvin(fahrenheit):
    """
    Converts a temperature from Fahrenheit to Kelvin.

    Parameters:
    Inputs:
        fahrenheit: Temperature in degrees Fahrenheit.

    Returns:
        Temperature in Kelvin.
    """
    return (fahrenheit - 32) * 5/9 + 273.15

# Reading the Temperatures
def read_temp(fpath):
    """
    A function that reads out the temperature of a single file

    Parameters:
    Inputs:
        fpath (string): The path of the file to read the temperature from
    Outputs:
        temp (number): The temperature read from the file in Fahrenheits
    """
    with open(fpath, 'r', encoding='utf-8') as file:
        content = file.readlines()
        temp = float(content[0].split(":")[1].strip())
    return temp

# Listing Files According to an extension and a filter
def list_files(directory, extension, fltr):
    """
    Lists files in a directory that match a specific extension and contain a filter string.

    Parameters:
        directory (str): The directory to search for files.
        extension (str): The file extension to filter (e.g., ".md").
        fltr (str): The string to filter filenames (case-sensitive).

    Returns:
        list: A list of filenames (relative paths) that match the filter and extension.
    """
    try:
        # List all files in the directory
        files = [
            file for file in os.listdir(directory)
            if os.path.isfile(os.path.join(directory, file)) and file.endswith(extension) and fltr in file
        ]
        return files
    except FileNotFoundError as e:
        raise FileNotFoundError(f"The directory '{directory}' does not exist.") from e
    except Exception as e:
        raise RuntimeError(f"An unexpected error occurred: {e}") from e

# Listing Folders According to a filter
def list_folders(directory, fltr):
    """
    Lists folder names in a directory that contain a specific filter string.

    Parameters:
        directory (str): The directory to search for folders.
        fltr (str): The string to filter folder names (case-sensitive).

    Returns:
        list: A list of folder names that match the filter.
    """
    try:
        # List all folders in the directory
        folders = [
            folder for folder in os.listdir(directory)
            if os.path.isdir(os.path.join(directory, folder)) and fltr in folder
        ]
        return folders
    except FileNotFoundError as e:
        raise FileNotFoundError(f"The directory '{directory}' does not exist.") from e
    except Exception as e:
        raise RuntimeError(f"An unexpected error occurred: {e}") from e

# Non-linear Fitting
def model(x, a, b, c):
    """ Defines the model to be fitted """
    return a * math.sin(b * x + c)

def residuals(x_data, y_data, params):
    """ Calculates the residuals for the fit """
    a, b, c = params
    return [y - model(x, a, b, c) for x, y in zip(x_data, y_data)]

def fit_sinusoidal(x_data, y_data, steps, initial_guess):
    """
    Perform non-linear fitting for a sinusoidal function y = A * sin(B * x + C).

    Parameters:
        x_data (list): List of x values (independent variable).
        y_data (list): List of y values (dependent variable).
        steps (number): Step number (to be in the form of 2^n).
        initial_guess (tuple): Initial guesses for A, B, and C (amplitude, frequency, phase).

    Returns:
        tuple: Fitted parameters (A, B, C).
    """
    # Parameters and step size
    a, b, c = initial_guess
    step_size = 1 / (2 ** steps)
    for _ in range(100):  # Iterate for convergence
        res = residuals(x_data, y_data, (a, b, c))
        grad_a = -2 * sum(r * math.sin(b * x + c) for r, x in zip(res, x_data))
        grad_b = -2 * sum(r * a * x * math.cos(b * x + c) for r, x in zip(res, x_data))
        grad_c = -2 * sum(r * a * math.cos(b * x + c) for r, x in zip(res, x_data))
        # Update parameters using gradient descent
        a -= step_size * grad_a
        b -= step_size * grad_b
        c -= step_size * grad_c
        # Check for convergence
        if max(abs(grad_a), abs(grad_b), abs(grad_c)) < 1e-6:
            break
    return a, b, c

# FFT Wrapper
def check_equidistant(data, x_col):
    """
    Check if the x-coordinates in the DataFrame are equidistant.

    Parameters:
        data (pandas.DataFrame): DataFrame containing the data.
        x_col (str): Column name for x-coordinates.

    Returns:
        bool: True if equidistant, False otherwise.
    """
    x = data[x_col]
    differences = np.diff(x.values)
    return np.allclose(differences, differences[0])

def fft(data, x_col, y_col):
    """
    Compute the FFT of the data from a DataFrame, ensuring equidistant x-coordinates.

    Parameters:
        data (pandas.DataFrame): DataFrame containing the data.
        x_col (str): Column name for x-coordinates.
        y_col (str): Column name for y-values.

    Returns:
        numpy.ndarray: FFT of the y-values.
    """
    y = data[y_col]
    if not check_equidistant(data, x_col):
        raise ValueError(f"Column '{x_col}' contains non-equidistant data.")
    return pd.Series(np.fft.fft(y))

def inv_fft(data, x_col, y_fft):
    """
    Compute the inverse FFT of the data from a DataFrame, ensuring equidistant x-coordinates.

    Parameters:
        data (pandas.DataFrame): DataFrame containing the data.
        x_col (str): Column name for x-coordinates.
        y_fft (numpy.ndarray): FFT-transformed data.

    Returns:
        pandas.Series: Inverse FFT of the transformed data.
    """
    if not check_equidistant(data, x_col):
        raise ValueError(f"Column '{x_col}' contains non-equidistant data.")
    return pd.Series(np.fft.ifft(y_fft), index = data.index)

# Calculating the Frequency Axis
def calculate_frequency_axis(sample_rate, num_samples):
    """
    Calculates the frequency axis for a given sampling rate and number of samples.

    Parameters:
        sample_rate (number): The sampling rate in inverse unit (samples per unit).
        num_samples (number): The total number of samples.

    Returns:
        list: Frequency axis in inverse units.
    """
    frequencies = []
    for k in range(num_samples):
        freq = k * sample_rate / num_samples
        frequencies.append(freq)
    return pd.Series(frequencies)

# Reading out the latitudes and longitudes from a .csv file
def xy_coord(path):
    """
    Extracts the latitudes and longitudes data out of a .csv file
    and returns the data converted to xy coordinates

    Parameters:
        Inputs:
            path (str): the absolute path of the .csv file
        
        Outputs:
            a tuble of two lists of x and y positions respectively
    """
    with open(path, 'r') as file:
        content = file.readlines()
        latitudes = list([float(i.split(',')[1]) for i in content[1:]])
        longitudes = list([float(i.split(',')[2]) for i in content[1:]])
    return xy_on_earth(latitudes, longitudes)
