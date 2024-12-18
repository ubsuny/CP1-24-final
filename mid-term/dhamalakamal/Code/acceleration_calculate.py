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
"""

import pandas as pd
import numpy as np

COLS = [
    "Linear Acceleration x (m/s^2)",
    "Linear Acceleration y (m/s^2)",
    "Linear Acceleration z (m/s^2)"
    ]

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
    data = pd.read_csv(file_path, header=0)
    directions = []
    for row in data.iterrows():
        acc_x = row[1]['Linear Acceleration x (m/s^2)']
        acc_y = row[1]['Linear Acceleration y (m/s^2)']
        acc_z = row[1]['Linear Acceleration z (m/s^2)']
        direction = calculate_direction(acc_x, acc_y, acc_z)
        directions.append(direction)
    return directions
