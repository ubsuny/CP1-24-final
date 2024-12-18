"""This Module imports data from a .csv file and calculates and plots the direction of motion."""

import numpy as np
import matplotlib.pyplot as plt

def read_acceleration_data(csv_file):

    """Read acceleration data from a CSV file."""

    data = np.genfromtxt(csv_file, delimiter=',', skip_header=1)

    # This will return a 2D array with time, x, y, z values

    return data

def calculate_position(data):

    """Calculate the position based on acceleration data using numerical integration."""

    time = data[:, 0]
    x_acc = data[:, 1]
    y_acc = data[:, 2]
    z_acc = data[:, 3]

    # Calculate the time intervals

    dt = np.diff(time, prepend=0)  # Time intervals
    x_velocity = np.cumsum(x_acc * dt)  # Velocity = integral of acceleration
    y_velocity = np.cumsum(y_acc * dt)  # Velocity = integral of acceleration
    z_velocity = np.cumsum(z_acc * dt)  # Velocity = integral of acceleration

    # Calculate cumulative position

    x_position = np.cumsum(x_velocity * dt)  # Position = integral of velocity
    y_position = np.cumsum(y_velocity * dt)  # Position = integral of velocity
    z_position = np.cumsum(z_velocity * dt)  # Position = integral of velocity

    return x_position, y_position, z_position

def plot_motion(x_position, y_position, z_position, gen_pic):

    """Plot the motion path for x, y, and z positions and save it as a PNG image."""

    plt.figure(figsize=(15, 10))

    # Plot X Position

    plt.subplot(3, 1, 1)
    plt.plot(x_position, color='r', linewidth=2)
    plt.title("X Position over Time")
    plt.xlabel("Time (s)")
    plt.ylabel("X Position (m)")
    plt.grid()

    # Plot Y Position

    plt.subplot(3, 1, 2)
    plt.plot(y_position, color='g', linewidth=2)
    plt.title("Y Position over Time")
    plt.xlabel("Time (s)")
    plt.ylabel("Y Position (m)")
    plt.grid()

    # Plot Z Position

    plt.subplot(3, 1, 3)
    plt.plot(z_position, color='b', linewidth=2)
    plt.title("Z Position over Time")
    plt.xlabel("Time (s)")
    plt.ylabel("Z Position (m)")
    plt.grid()

    plt.tight_layout()
    plt.savefig(gen_pic)
    plt.close()

def process_acceleration_data(csv_file, gen_pic):

    """Main function to read acceleration data, calculate positions, and plot the motion."""

    # Step 1: Read the acceleration data

    data = read_acceleration_data(csv_file)

    # Step 2: Calculate the position based on acceleration

    x_position, y_position, z_position = calculate_position(data)

    # Step 3: Plot the motion and save as PNG

    plot_motion(x_position, y_position, z_position, gen_pic)

    return x_position, y_position, z_position
