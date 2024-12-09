# direction_calculator.py
import numpy as np

def calculate_direction(acceleration_data):
    """
    Calculates direction of motion from x, y, z acceleration data.
    Returns a list of direction angles in degrees.
    """
    directions = []
    for _, row in acceleration_data.iterrows():
        x, y, z = row['x'], row['y'], row['z']
        direction = np.degrees(np.arctan2(y, x))
        directions.append(direction)
    return directions