"""
This module calculates the direction of motion based on acceleration data. 
It includes a function to normalize acceleration vectors and an example usage.
"""
import numpy as np

def calculate_direction(acceleration_data):
    """
    Calculate the direction of motion based on acceleration data.

    Parameters:
    acceleration_data (list of lists): A list of [ax, ay, az] values representing 
    the acceleration in the x, y, and z axes.

    Returns:
    list of tuples: A list of normalized direction vectors in the form (dx, dy, dz).
    """
    directions = []
    for ax, ay, az in acceleration_data:
        magnitude = np.sqrt(ax**2 + ay**2 + az**2)
        if magnitude != 0:
            direction = (ax / magnitude, ay / magnitude, az / magnitude)
        else:
            direction = (0, 0, 0)
        directions.append(direction)
    return directions

def example_usage():
    """
    Example usage of the calculate_direction function using sample acceleration data.
    """
    example_acceleration_data = [
        [1.027989984E-1, 9.331600368E-2, -6.961200237E-1],  # ax, ay, az for time t1
        [3.441699967E-2, -8.970600367E-2, -1.392800063E-1],  # ax, ay, az for time t2
    ]
    directions = calculate_direction(example_acceleration_data)
    print(directions)  # This will give the direction vectors over time

# Call the example usage function
if __name__ == "__main__":
    example_usage()
