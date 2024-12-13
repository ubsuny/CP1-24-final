"""Returns the direction from acceleration given coordinates in 3D"""

import math

def direction_from_acceleration(x, y, z):
    """
    Calculate the direction of motion in 3D space from acceleration data.
    Returns the polar and azimuthal angles (in degrees).
    """
    # Calculate the magnitude of the vector
    magnitude = math.sqrt(x**2 + y**2 + z**2)

    # Avoid division by zero
    if magnitude == 0:
        raise ValueError("Zero acceleration vector, cannot determine direction.")

    # Calculate the polar angle θ (angle from z-axis)
    theta = math.acos(z / magnitude)

    # Calculate the azimuthal angle φ (angle in x-y plane)
    phi = math.atan2(y, x)

    # Convert radians to degrees
    theta_deg = math.degrees(theta)
    phi_deg = math.degrees(phi)

    return theta_deg, phi_deg
