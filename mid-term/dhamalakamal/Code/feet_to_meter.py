"""
This module provides simple conversion functions to convert distances from 
feet and yards to meters.

Constants:
    - FEET_TO_METERS: Conversion factor from feet to meters (1 foot = 0.3048 meters).
    - YARDS_TO_METERS: Conversion factor from yards to meters (1 yard = 0.9144 meters).

Functions:
    - feet_to_meters(feet): Converts a given distance in feet to meters.
    - yards_to_meters(yards): Converts a given distance in yards to meters.
"""
# Conversion factors
FEET_TO_METERS = 0.3048
YARDS_TO_METERS = 0.9144

def feet_to_meters(feet):
    """
    Convert feet to meters.
    Parameters:
    feet (float): The distance in feet.
    Returns:
    float: The distance in meters.
    """
    return feet * FEET_TO_METERS

def yards_to_meters(yards):
    """
    Convert yards to meters.
    Parameters:
    yards (float): The distance in yards.
    Returns:
    float: The distance in meters.
    """
    return yards * YARDS_TO_METERS
