# gps_motion_plot.py

import pandas as pd
import matplotlib.pyplot as plt

# Load sample GPS data (replace with actual data file path)
data = pd.read_csv("Data/circle_path.csv")

plt.figure(figsize=(8, 8))
plt.plot(data['longitude'], data['latitude'], marker="o")
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.title("Circle Path GPS Motion")
plt.savefig("documentation/circle_path_trajectory.png")
plt.show()