"""
This module calculates the direction of movement based on acceleration data 
read from a CSV file. It uses functions from the mtaccelfuncts module to 
process the acceleration values and determine the movement direction.

Functions:
- calculate_dir(accel_file): Calculates the direction of movement from acceleration data.
"""

import os
import sys
from mtaccelfuncts import read_acceleration, check_direction
sys.path.append('/workspaces/FORKCP1-24-final/p_pxmpo_midterm/midtermcode/accel')

def calculate_dir(accel_file):
    """
    Calculate the direction of movement based on acceleration data from a CSV file.


    accel_file: The path to the CSV file containing acceleration data.

    Returns:
        The direction of movement ("up", "down", or "No significant trend").
    """

    y_values = read_acceleration(accel_file)
    direction = check_direction(y_values)
    return direction

# Prompt the user for the file name
file_input = input("Please enter the CSV acceleration data file path: ")

# Check if the file exists
if not os.path.isfile(file_input):
    print(f"Error: The file '{file_input}' was not found.")
else:
    DIRECTION_OF_ACCELERATION = calculate_dir(file_input)
    print("The direction of acceleration is", DIRECTION_OF_ACCELERATION )
