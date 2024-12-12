"""
This module provides functions to read acceleration data from a CSV file 
and determine the direction of movement based on the acceleration values.

Functions:
- read_acceleration(file): Reads acceleration values from a specified CSV file.
- check_direction(y_values): Checks the direction of movement based on the acceleration values.
"""

import csv

def read_acceleration(file):
    """
    Reads acceleration values from a specified CSV file.

    file: Placeholder to the CSV file containing acceleration data.

    Returns:
        list: A list of acceleration values extracted from the third column 
        of the CSV file.
    """

    y_values = []
    with open(file, mode='r', encoding='utf-8') as csvfile:
        csv_reader = csv.reader(csvfile)
        next(csv_reader)  # Skip the header
        for row in csv_reader:
            value = float(row[2])  # Convert to float
            y_values.append(value)

    return y_values

def check_direction(y_values):
    """
    Checks the direction of movement based on the acceleration values.

    y_values (list): A list of acceleration values (floats).

    Returns:
        str: "up" if any acceleration value is greater than 1 first,
             "down" if any value is less than -1 first,
             "No significant trend" if all values are within the range [-1, 1].
    """

    for value in y_values:
        if value > 1:
            return "up"
        if value < -1:
            return "down"
    return "No significant trend"
