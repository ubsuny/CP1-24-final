"""This module reads acceleration data from a CSV file and plots the motion path 
in x, y, and z directions over time.
"""

import numpy as np
import matplotlib.pyplot as plt

def calculate_direction(x, y, _z=0):
    """Calculate the direction in the XY plane based on x and y acceleration components."""
    return np.degrees(np.arctan2(y, x))

def read_acceleration_data(csv_file):
    """Read acceleration data from a CSV file."""
    try:
        data = np.genfromtxt(csv_file, delimiter=',', skip_header=1)
        if data.shape[1] != 4:
            raise ValueError("CSV file should have exactly four columns: time, x, y, z")
        return data
    except IOError:
        print(f"Error: Could not read file {csv_file}")
        return None
    except ValueError as e:
        print(f"Error: {e}")
        return None

def calculate_position(data):
    """Calculate position using numerical integration of acceleration data."""
    time = data[:, 0]
    x_acc, y_acc, z_acc = data[:, 1], data[:, 2], data[:, 3]

    dt = np.diff(time, prepend=0)
    x_velocity = np.cumsum(x_acc * dt)
    y_velocity = np.cumsum(y_acc * dt)
    z_velocity = np.cumsum(z_acc * dt)
    x_position = np.cumsum(x_velocity * dt)
    y_position = np.cumsum(y_velocity * dt)
    z_position = np.cumsum(z_velocity * dt)

    return x_position, y_position, z_position

def plot_motion(x_position, y_position, z_position, gen_pic):
    """Plot motion in x, y, and z directions over time and save as an image."""
    plt.figure(figsize=(15, 10))
    positions = [(x_position, 'r', "X Position"), (y_position, 'g', "Y Position"),
                 (z_position, 'b', "Z Position")]
    for i, (position, color, title) in enumerate(positions, start=1):
        plt.subplot(3, 1, i)
        plt.plot(position, color=color, linewidth=2)
        plt.title(f"{title} over Time")
        plt.xlabel("Time (s)")
        plt.ylabel(f"{title} (m)")
        plt.grid()
    plt.tight_layout()
    plt.savefig(gen_pic)
    plt.close()
    print(f"Motion plot saved as {gen_pic}")
