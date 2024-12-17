"""
final.py

This module implements the functionality needed to complete the algorithm tasks
for the PHY410 final project. This includes unit converters, file parsers, and
data handling methods.
"""
import os
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

def extract_number_from_filename(fname):
    """
    Obtain the trailing numeric value from a filename.

    Parameters:
        fname (str): The filename to process.

    Returns:
        int: The numeric value found before the file extension.
             Returns a high default value for invalid filenames.
    """
    try:
        # Split by underscore and get last part
        last_part = fname.rsplit('_', 1)[-1]
        # Remove file extension
        number_str = last_part.split('.', 1)[0]
        # Convert to integer
        return int(number_str)
    except (ValueError, TypeError):
        # Return a large default value for filenames without numeric suffixes
        return float('inf')

def sort_filenames(file_list):
    """
    Sorts a list of filenames based on the trailing numeric value before the file extension.

    Parameters
    ----------
    file_list : list of str
        The list of filenames to be sorted.

    Returns
    -------
    list of str
        The input filenames sorted based on the trailing numeric value before the extension.
    """

    return sorted(file_list, key=extract_number_from_filename)

def filter_markdown_files(directory_path, filter_string):
    """
    Sort and list all markdown filenames in a directory that contain the filter string.

    Parameters:
        directory_path (str): The path to the directory to scan.
        filter_string (str): The string to filter filenames.

    Returns:
        list: A list of sorted markdown filenames containing the filter string.
    """
    # check for the directoryj
    if not os.path.exists(directory_path):
        raise FileNotFoundError(f"The directory '{directory_path}' does not exist.")
    # complain if its missing
    if not os.path.isdir(directory_path):
        raise NotADirectoryError(f"'{directory_path}' is not a valid directory.")

    # List all files in the directory and filter markdown files with the filter string
    filtered_files = [
        filename for filename in os.listdir(directory_path)
        if filename.endswith(".md") and filter_string in filename
    ]
    # return the filenames to caller
    sorted_files = sort_filenames(filtered_files)


    return sorted_files
