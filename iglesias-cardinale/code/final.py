'''This module contains all of the functions necessary to complete 
the Final Exam for Computational Physics 1, 2024.'''

import os
import numpy as np

def fahrenheit_to_kelvin(fahrenheit):
    '''This function converts a given temperature in Degrees Fahrenheit 
    to the equivalent absolute temperature in Kelvin.
    
    Parameters:
        -fahrenheit: Degrees Fahrenheit, to be converted
        
    Returns:
        -kelvin: Equivalent temperature in Kelvin
    '''
    kelvin = (fahrenheit-32)*5/9 + 273.15

    return kelvin

def get_temp(path):
    '''Reads the temperature of a given experiment from that 
    trials metadata markdown file
    
    Parameters:
        -path: String. Path to markdown file containing metadata
    
    Returns:
        -temp: temperature from metadata file'''
    data = np.loadtxt(path, unpack=True)
    temp = data[0]

    return temp


def list_files_with_string(directory, search_string, file_extension):
    """
    Generates a list of file paths in a directory (and its subdirectories)
    where the filenames contain a specific string.

    Parameters:
        directory (str): The path to the directory to search.
        search_string (str): The string to look for in filenames.
        file_extension (str): The file extension we're searching for

    Returns:
        list: A list of file paths matching the search criteria.
    """
    matching_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if search_string in file and file.endswith(file_extension):
                matching_files.append(os.path.join(root, file))
    return matching_files
