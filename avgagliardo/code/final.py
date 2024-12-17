"""
final.py

This module implements the functionality needed to complete the algorithm tasks
for the PHY410 final project. This includes unit converters, file parsers, and
data handling methods.
"""
import re

def convert_f_to_k(f):
    """
    Convert a temperature from Fahrenheit to Kelvin.

    Parameters:
        fahrenheit (float): The temperature in degrees Fahrenheit.
    Returns:
        float: The temperature converted to Kelvin.
    """
    return (f - 32) * 5 / 9 + 273.15

def parse_markdown(file_path, field_name):
    """
    Parse a markdown file and extract the value of a specified field.

    Parameters:
        file_path (str): Path to the markdown file.
        field_name (str): The name of the field to extract (e.g., 'Temperature').

    Returns:
        float: The extracted value as a float, or None if the field is not found.
    """
    with open(file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()  # Read all lines into a list
        extracted_value = None
        for i, line in enumerate(lines):
            if line.strip() == f"##{field_name}":
                # The next line should contain the value in the format '33 F'
                if i + 1 < len(lines):
                    match = re.match(r"(-?\d+\.?\d*)\s*F", lines[i + 1].strip())
                    if match:
                        extracted_value = float(match.group(1))
    return extracted_value
