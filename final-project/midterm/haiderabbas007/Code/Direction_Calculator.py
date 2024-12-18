"""
This module calculates the direction of motion from acceleration data.

It reads CSV files containing raw linear acceleration data in (x, y, z), calculates the
magnitude of acceleration vectors, and normalizes these vectors to obtain
unit vectors (direction vectors) that represent the direction of motion. The results are
provided as a dictionary with the file paths as keys and lists of unit vectors as values.

Functions:
    calculate_direction_of_motion(paths): Reads raw acceleration data and calculates unit vectors
    for the direction of motion.
"""


import pandas as pd
import numpy as np

# List of paths of raw acceleration data files
paths_list = ["haiderabbas007/Data/Raw Data (.csv)/Acceleration_Up_to_Down.csv",
              "haiderabbas007/Data/Raw Data (.csv)/Acceleration_Down_to_Up.csv"]

def calculate_direction_of_motion(paths):
    """
    Calculates the direction of motion as a unit vector based on of x, y, z components of acceleration.

    Parameters:
        paths (list of str): List of paths to the CSV files containing acceleration data.

    Returns:
        dict: A dictionary where each key is the file path and each value is a list of tuples of
        components of unit vectors.
    """

    all_unit_vectors = {}

    for path in paths_list:
        # Load the data
        data = pd.read_csv(path)

        # Pull out acceleration components
        ax = data['Linear Acceleration x (m/s^2)'].values
        ay = data['Linear Acceleration y (m/s^2)'].values
        az = data['Linear Acceleration z (m/s^2)'].values

        # Calculate unit vectors and their magnitudes
        magnitudes = np.sqrt(ax ** 2 + ay ** 2 + az ** 2)
        unit_vectors = [(ax[i] / magnitudes[i], ay[i] / magnitudes[i], az[i] / magnitudes[i]) if magnitudes[i] != 0 else (0, 0, 0)
            for i in range(len(magnitudes))]
        # Assign unit vectors to the parent file path
        all_unit_vectors[path] = unit_vectors

    return all_unit_vectors
