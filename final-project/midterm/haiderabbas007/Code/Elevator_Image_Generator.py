"""
This module plots the direction vectors (unit vectors) and acceleration magnitudes
over time for the Elevator Runs.

It reads CSV files containing raw linear acceleration data in (x, y, z), extracts the
magnitudes and direction vectors, and plots them over time.

It also provides functions to plot and visualize the direction vectors in 3D plots.

Functions:
    plot_acceleration_vs_time(paths): Plots the magnitude of acceleration over time for each dataset.
    plot_direction_vectors_3d(paths, direction_of_motion): Plots direction vectors as 3D quivers for each dataset.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from Direction_Calculator import calculate_direction_of_motion

# Paths of raw CVS data files
Acceleration_Down_to_Up = "haiderabbas007/Data/Raw Data (.csv)/Acceleration_Down_to_Up.csv"
Acceleration_Up_to_Down = "haiderabbas007/Data/Raw Data (.csv)/Acceleration_Up_to_Down.csv"

paths_list = [Acceleration_Up_to_Down, Acceleration_Down_to_Up]

def plot_acceleration_vs_time(paths):
    """
    Plots the magnitude of acceleration over time for each dataset.

    Parameters:
        paths (list of str): List of paths to the CSV files containing acceleration data.
    """
    for path in paths:
        # Load the data
        data = pd.read_csv(path)

        # Extract time and acceleration components
        time = data['Time (s)'].values
        ax = data['Linear Acceleration x (m/s^2)'].values
        ay = data['Linear Acceleration y (m/s^2)'].values
        az = data['Linear Acceleration z (m/s^2)'].values

        # Calculate the magnitude of the acceleration
        magnitude = np.sqrt(ax ** 2 + ay ** 2 + az ** 2)

        # Plotting the magnitude of acceleration over time
        plt.figure(figsize=(10, 6))
        plt.plot(time, magnitude, label='Acceleration Magnitude', color='blue')
        plt.xlabel('Time (s)')
        plt.ylabel('Acceleration Magnitude (m/s^2)')
        plt.title(f'Acceleration Magnitude vs Time\n{path}')
        plt.legend()
        plt.grid(True)
        plt.show()

def plot_direction_vectors_3d(paths, direction_of_motion):
    """
    Plots the direction vectors as 3D quivers for each dataset.

    Parameters:
        paths (list of str): List of paths to the CSV files containing acceleration data.
        direction_of_motion (dict): A dictionary containing unit vectors for each dataset.
    """
    for path in paths:
        # Extract unit vectors
        unit_vectors = np.array(direction_of_motion[path])
        ax = unit_vectors[:, 0]
        ay = unit_vectors[:, 1]
        az = unit_vectors[:, 2]

        # Reduce the number of vectors for better visualization
        step = max(1, len(ax) // 100)
        ax = ax[::step]
        ay = ay[::step]
        az = az[::step]

        # Create a 3D plot of the direction vectors
        fig = plt.figure(figsize=(10, 6))
        ax_3d = fig.add_subplot(111, projection='3d')
        ax_3d.quiver(0, 0, 0, ax, ay, az, length=0.1, normalize=True, color='blue')
        ax_3d.set_xlabel('X Component')
        ax_3d.set_ylabel('Y Component')
        ax_3d.set_zlabel('Z Component')
        ax_3d.set_title(f'Direction Vectors (Unit Vectors)\n{path}')
        plt.show()

# Import the calculate_direction_of_motion function from Direction_Calculator.py
# Calculate direction of motion
direction_of_motion = calculate_direction_of_motion(paths_list)

# Plot acceleration vs time
plot_acceleration_vs_time(paths_list)

# Plot direction vectors in 3D
plot_direction_vectors_3d(paths_list, direction_of_motion)
