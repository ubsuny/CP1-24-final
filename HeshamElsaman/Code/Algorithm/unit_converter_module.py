"""
This module is to convert between the units
"""

from libraries_module import np

def foot_to_meters(foot):
    """
    A function that returns a length in (meters)
    that's equivalent to some length in (foot)
    
    Parameters:
    foot (number): the input length in (foot)

    Returns:
    (number): the equivalent length in (meters)
    """
    return 0.3048 * foot

def foot_to_kilometers(foot):
    """
    A function that returns a length in (kilometers)
    that's equivalent to some length in (foot)
    
    Parameters:
    foot (number): the input length in (foot)

    Returns:
    (number): the equivalent length in (kilometers)
    """
    return 0.0003048 * foot

def yards_to_meters(yards):
    """
    A function that returns a length in (meters)
    that's equivalent to some length in (yards)
    
    Parameters:
    yards (number): the input length in (yards)

    Returns:
    (number): the equivalent length in (meters)
    """
    return 0.9144 * yards

def yards_to_kilometers(yards):
    """
    A function that returns a length in (kilometers)
    that's equivalent to some length in (yards)
    
    Parameters:
    yards (number): the input length in (yards)

    Returns:
    (number): the equivalent length in (kilometers)
    """
    return 0.0009144 * yards

def meters_to_foot(meters):
    """
    A function that returns a length in (foot)
    that's equivalent to some length in (meters)
    
    Parameters:
    meters (number): the input length in (meters)

    Returns:
    (number): the equivalent length in (foot)
    """
    return 3.28084 * meters

def kilometers_to_foot(kilometers):
    """
    A function that returns a length in (foot)
    that's equivalent to some length in (kilometers)
    
    Parameters:
    kilometers (number): the input length in (kilometers)

    Returns:
    (number): the equivalent length in (foot)
    """
    return 3280.84 * kilometers

def meters_to_yards(meters):
    """
    A function that returns a length in (yards)
    that's equivalent to some length in (meters)
    
    Parameters:
    meters (number): the input length in (meters)

    Returns:
    (number): the equivalent length in (yards)
    """
    return 1.09361 * meters

def kilometers_to_yards(kilometers):
    """
    A function that returns a length in (yards)
    that's equivalent to some length in (kilometers)
    
    Parameters:
    kilometers (number): the input length in (kilometers)

    Returns:
    (number): the equivalent length in (yards)
    """
    return 1093.61 * kilometers

def foot_to_yards(foot):
    """
    A function that returns a length in (yards)
    that's equivalent to some length in (foot)
    
    Parameters:
    foot (number): the input length in (foot)

    Returns:
    (number): the equivalent length in (yards)
    """
    return foot / 3

def yards_to_foot(yards):
    """
    A function that returns a length in (foot)
    that's equivalent to some length in (yards)
    
    Parameters:
    yards (number): the input length in (yards)

    Returns:
    (number): the equivalent length in (foot)
    """
    return 3 * yards

def meters_to_kilometers(meters):
    """
    A function that returns a length in (kilometers)
    that's equivalent to some length in (meters)
    
    Parameters:
    meters (number): the input length in (meters)

    Returns:
    (number): the equivalent length in (kilometers)
    """
    return meters / 1000

def kilometers_to_meters(kilometers):
    """
    A function that returns a length in (meters)
    that's equivalent to some length in (kilometers)
    
    Parameters:
    kilometers (number): the input length in (kilometers)

    Returns:
    (number): the equivalent length in (meters)
    """
    return kilometers * 1000

def deg_to_rad(degrees):
    """
    Returns the angle in radians

    Parameters:
    Inputs:
    degrees (array of numbers): the angle(s) in degrees
    Outputs:
    degs (array of numbers): the angle(s) in radians
    """
    degs = np.array(degrees)
    return np.radians(degs)

def rad_to_deg(radians):
    """
    Returns the angle in degrees

    Parameters:
    Inputs:
    radians (array of numbers): the angle(s) in radians
    Outputs:
    rads (array of numbers): the angle(s) in degrees
    """
    rads = np.array(radians)
    return np.degrees(rads)
