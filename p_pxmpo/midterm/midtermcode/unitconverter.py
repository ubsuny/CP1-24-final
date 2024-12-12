"""
This module provides conversion functions for length measurements.

The following conversions are included:
- Yards to meters
- Feet to meters
"""
def yards_to_meters(yards):
    """
    Convert yards to meters.

    Args:
        yards (float): The length in yards.

    Returns:
        float: The length in meters.
    """
    return yards * 0.9144


def feet_to_meters(feet):
    """
    Convert feet to meters.

    Args:
        feet (float): The length in feet.

    Returns:
        float: The length in meters.
    """
    return feet * 0.3048
