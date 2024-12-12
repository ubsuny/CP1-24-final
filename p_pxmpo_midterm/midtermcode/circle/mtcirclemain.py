"""
This module prompts the user for a CSV file containing GPS coordinates, 
validates the file's existence, and calculates the distances between adjacent 
GPS points using functions from the `mtcirclefuncts` module.
"""
import sys
import os
sys.path.append('/workspaces/FORKCP1-24-final/p_pxmpo_midterm/midtermcode/circle')

import mtcirclefuncts

# Prompt the user for the CSV file name
file_input = input("Please enter the CSV circle data file path: ")

# Check if the file exists
if not os.path.isfile(file_input):
    print(f"Error: The file '{file_input}' was not found.")
else:
    # Load the GPS coordinates from the CSV file
    gps_coordinates = mtcirclefuncts.read_gps_from_csv(file_input)

    # Calculate distances between adjacent GPS points
    distances = mtcirclefuncts.calculate_distances(gps_coordinates)

    # Print the distances
    for i, distance in enumerate(distances):
        print(f"Distance between point {i} and {i + 1}: {distance:.9f} km")
