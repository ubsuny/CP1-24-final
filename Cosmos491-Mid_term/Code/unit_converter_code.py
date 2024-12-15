"""
This module contains functions for converting distances between feet, yards, and meters.
"""
def feet_to_meters(feet):
    """
    Converts a distance from feet to meters.

    Parameters:
    feet (float): Distance in feet.

    Returns:
    float: Distance in meters.
    """
    return feet * 0.3048  # Conversion factor for feet to meters
def yards_to_meters(yards):
    """
    Converts a distance from yards to meters.

    Parameters:
    yards (float): Distance in yards.

    Returns:
    float: Distance in meters.
    """
    return yards * 0.9144  # Conversion factor for yards to meters
