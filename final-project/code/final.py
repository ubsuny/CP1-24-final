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

def list_markdown_files(directory, filename_filter):
    """
    List all markdown files in a directory that match a specific filename filter.

    Parameters:
        directory (str): Path to the directory containing files.
        filename_filter (str): A string filter to match filenames (e.g., "sinewalk").

    Returns:
        list: A list of markdown file paths that match the filter.
    """
    try:
        # Get all files in the directory
        all_files = os.listdir(directory)

        # Filter files by those ending with '.md' and containing the filename_filter
        markdown_files = [
            os.path.join(directory, file) for file in all_files
            if file.endswith('.md') and filename_filter in file
        ]
        return markdown_files

    except FileNotFoundError:
        print(f"Directory not found: {directory}")
        return []
    except Exception as e:
        print(f"An error occurred: {e}")
        return []
