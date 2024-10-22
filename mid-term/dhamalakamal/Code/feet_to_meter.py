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
