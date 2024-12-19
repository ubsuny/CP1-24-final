# elevator_direction_plot.py

import matplotlib.pyplot as plt
import pandas as pd
from Tolani4.Code.direction_calculator import calculate_direction

# Load sample acceleration data (replace with actual data file path)
data = pd.read_csv("Data/elevator_up.csv")
directions = calculate_direction(data)

plt.plot(directions)
plt.xlabel("Time")
plt.ylabel("Direction (degrees)")
plt.title("Elevator Direction Over Time")
plt.savefig("documentation/elevator_direction.png")
plt.show()