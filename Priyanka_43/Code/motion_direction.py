"""
This module calculate the direction of motion from acceleration data.
"""
# motion_direction.py
import numpy as np

def calculate_direction(acceleration_data):
    """
    Calculates direction of motion from x, y acceleration data.
    Returns a list of direction angles in degrees.
    """
    directions = []
    for _, row in acceleration_data.iterrows():
        x, y = row['x'], row['y']
        direction = np.degrees(np.arctan2(y, x))
        directions.append(direction)
    return directions
