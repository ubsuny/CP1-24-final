"""
This module is to provide all the needed functionality for the final exam of CP1-24
"""

import numpy as np
import pandas as pd

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
