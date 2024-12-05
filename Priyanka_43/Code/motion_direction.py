"""
This module calculate the direction of motion from acceleration data.
"""
# motion_direction.py

import math

def calculate_direction(x, y, z):
   
    """
    Calculate the direction of motion from acceleration data.
    
    Parameters:
    acceleration (tuple): A tuple containing the acceleration data (ax, ay, az)
                          in the x, y, and z directions.
                          
    Returns the polar and azimuthal angles (in degrees).
    """
    # Calculate the magnitude of the vector
    magnitude = math.sqrt(x**2 + y**2 + z**2)

    # Calculate the polar angle θ (angle from z-axis)
    theta = math.acos(z / magnitude)
   
    # Calculate the azimuthal angle φ (angle in x-y plane)
    phi = math.atan2(y, x)
    # Convert radians to degrees
    theta_deg = math.degrees(theta)
    phi_deg = math.degrees(phi)

    return "theta_deg, phi_deg"
 
