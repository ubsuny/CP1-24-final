"""
This module loads, reads, and visualizes GPS location data for the Circle and Triangle runs.
It creates a scatter plot to represent the geographical locations of the two paths.
Circle data is plotted in blue and the Triangle data is plotted in red.

Libraries:
- pandas: Used to load and handle the CSV data files.
- matplotlib: Used for visualizing the data points on a scatter plot.
"""

import pandas as pd
import matplotlib.pyplot as plt

# Load the GPS data files
circle_data = pd.read_csv('haiderabbas007/Data/Raw Data (.csv)/Circle.csv') # Circle Run raw data file
triangle_data = pd.read_csv('haiderabbas007/Data/Raw Data (.csv)/Triangle.csv') # Triangle Run raw data file

# Draw canvas
plt.figure(figsize=(10, 6))

# Plot Circle dataset points
plt.scatter(circle_data['Longitude (°)'], circle_data['Latitude (°)'], color='blue', label='Circle Path', alpha=0.5)

# Plot Triangle dataset points
plt.scatter(triangle_data['Longitude (°)'], triangle_data['Latitude (°)'], color='red', label='Triangle Path', alpha=0.5)

# Label the plot
plt.xlabel('Longitude (°)')
plt.ylabel('Latitude (°)')
plt.title('GPS Points: Circle and Triangle Paths')
plt.legend()
plt.grid(True)  # Turn on the grid!

# Display the beauty!
plt.show()
