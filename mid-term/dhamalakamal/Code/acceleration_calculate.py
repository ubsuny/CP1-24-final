"""
This module provides functions to calculate the direction of motion from recorded acceleration data,
based on azimuth and elevation angles derived from accelerometer readings in the X, Y, and Z axes.

Functions:
    - calculate_direction(accel_x, accel_y, accel_z): Computes the azimuth (XY plane angle) and
      elevation (vertical angle) from the provided acceleration data.
    - calculate_directions_from_csv(file_path): Reads acceleration data from a CSV 
    file and calculates the direction of motion for each data point using the 
    calculate_direction function.

The calculated direction is represented by two angles:
    - Azimuth: The angle between the positive X-axis and the projection of the 
    vector onto the XY plane (in degrees).
    - Elevation: The angle between the vector and the XY plane, 
    i.e., how "upward" or "downward" the motion is (in degrees).

CSV File Structure:
    The CSV file should contain the following columns:
    - 'Acceleration_X': Acceleration data along the X-axis.
    - 'Acceleration_Y': Acceleration data along the Y-axis.
    - 'Acceleration_Z': Acceleration data along the Z-axis.
    Additional columns like 'Timestamp' may be present, but are not used by the function.

Example CSV Structure:
    Timestamp,Acceleration_X,Acceleration_Y,Acceleration_Z
    2024-10-24 00:00:00,0.1,0.2,9.8
    2024-10-24 00:00:01,0.2,0.1,9.7

Usage:
    To use this module, call the 'calculate_directions_from_csv' function, passing the path
    to the CSV file containing the acceleration data. This function will return a list of columns,
    where each columns contains the azimuth and elevation angles for each recorded data point.

Example:
    csv_file = 'data/acceleration_data.csv'
    directions = calculate_directions_from_csv(csv_file)
    for idx, (azimuth, elevation) in enumerate(directions, 1):
        print(f"Direction for point {idx}: Azimuth = {azimuth:.2f}°, Elevation = {elevation:.2f}°")
"""

import pandas as pd
import numpy as np

def calculate_direction(acc_x, acc_y, acc_z):
    """
    Calculate the direction of motion based on acceleration data.
    Args:
        accel_x (float): Acceleration in the X direction.
        accel_y (float): Acceleration in the Y direction.
        accel_z (float): Acceleration in the Z direction.
    Returns:
        tuple: A tuple containing the angles (in degrees) representing the direction
               of motion in the XY plane (azimuth) and elevation.
    """
    # Calculate azimuth angle in the XY plane
    direction_azimuth = np.arctan2(acc_y, acc_x) * (180 / np.pi)
    # Calculate elevation angle
    hypotenuse = np.sqrt(acc_x**2 + acc_y**2)
    direction_elevation = np.arctan2(acc_z, hypotenuse) * (180 / np.pi)
    return direction_azimuth, direction_elevation
def calculate_directions_from_csv(file_path):
    """
    Read acceleration data from a CSV file and calculate the direction of motion
    for each recorded data point.
    Args:
        file_path (str): Path to the CSV file containing acceleration data.
    Returns:
        list: A list of tuples containing the azimuth and elevation angles
              for each recorded acceleration data point.
    """
    # Load the acceleration data into a DataFrame
    data = pd.read_csv(file_path)
    directions = []
    for row in data.iterrows():
        acc_x = row['Linear Acceleration x (m/s^2)']
        acc_y = row['Linear Acceleration y (m/s^2)']
        acc_z = row['Linear Acceleration z (m/s^2)']
        direction = calculate_direction(acc_x, acc_y, acc_z)
        directions.append(direction)
    return directions
