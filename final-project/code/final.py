import numpy as np
import pandas as pd
import matplotlib as plt
import os

def fahrenheit_to_kelvin(fahrenheit):
    """
    Convert Fahrenheit to Kelvin.

    Parameters:
        fahrenheit (float): Temperature in Fahrenheit.

    Returns:
        float: Temperature in Kelvin.
    """
    kelvin = (fahrenheit - 32) * 5/9 + 273.15
    return kelvin

def parse_temperature_from_markdown(filepath):
    """
    Parse the temperature value from a markdown file.

    Parameters:
        filepath (str): Path to the markdown file.

    Returns:
        float: Temperature in Fahrenheit (as a float).
        None: If no temperature is found in the file.
    """
    try:
        # Read the markdown file using pandas
        with open(filepath, 'r') as file:
            lines = file.readlines()
        
        # Look for a line containing "Temperature | XX°F"
        for line in lines:
            if "Temperature" in line and "°F" in line:
                # Extract the numeric value before °F
                temperature = float(line.split('|')[1].strip().replace('°F', ''))
                return temperature
        
        print("No temperature found in the markdown file.")
        return None
    except FileNotFoundError:
        print(f"File not found: {filepath}")
        return None
