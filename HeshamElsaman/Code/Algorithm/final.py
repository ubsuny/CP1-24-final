"""
This module is to provide all the needed functionality for the final exam of CP1-24
"""

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