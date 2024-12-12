"""
This module reads acceleration data from a CSV file and plots acceleration (m/s²) versus time (s).

The CSV file should contain columns for 'time' and 'acceleration'.
The plot is generated using matplotlib.
"""

import csv
import matplotlib.pyplot as plt

def plot_acceleration(csv_file_path):
    """
    Plots acceleration (m/s²) vs time (s) from a CSV file.

    Args:
        csv_file_path (str): The path to the CSV file containing the data.
                         The CSV should have columns 'time' and 'acceleration'.
    """
    # Initialize lists for time and acceleration
    time = []
    acceleration = []

    # Read the CSV file using with open
    with open(csv_file_path, 'r', encoding='utf-8') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter = ',')
        next(csv_reader)

        # Extract data
        for row in csv_reader:
            time.append(float(row[0]))
            acceleration.append(float(row[2]))

    # Plotting
    plt.figure(figsize=(10, 5))
    plt.plot(time, acceleration, marker='o', linestyle='-', color='b')
    plt.title('Acceleration vs Time')
    plt.xlabel('Time (s)')
    plt.ylabel('Acceleration (m/s²)')
    plt.grid()
    plt.axhline(0, color='black', lw=0.8, ls='--')  # Add a horizontal line at y=0
    plt.show()

# Usage
csv_file_input = input("Enter the path to the CSV file: ")
try:
    plot_acceleration(csv_file_input)
except FileNotFoundError:
    print(f"Error: The file '{csv_file_input}' was not found.")
except ValueError:
    print("Error: There was an issue with the data format. "
          "Please ensure the CSV file contains valid numeric values.")
except IOError:
    print(f"Error: There was an issue opening the file '{csv_file_input}'.")
