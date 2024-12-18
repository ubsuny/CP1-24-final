"""
This module is to define the functions to be used
to transform the angular GPS locations into planar coordinates
"""

from libraries_module import np

# Function to calculate the mercator projections
def latlon_to_xy_mercator(latitudes, longitudes, radius):
    """
    Converts latitude and longitude to x, y coordinates using the Mercator projection.

    Parameters:
    Inputs:
    latitudes (list or array): List of latitudes in degrees.
    longitudes (list or array): List of longitudes in degrees.
    radius (number): Radius of the sphere.

    Returns:
    x (array): X coordinates in the 2D plane.
    y (array): Y coordinates in the 2D plane.
    """
    latitudes_rad = list(np.radians(np.array(latitudes)))
    longitudes_rad = list(np.radians(np.array(longitudes)))

    x = [radius * i for i in longitudes_rad]
    y = [radius * np.log(np.tan(np.pi / 4 + i / 2)) for i in latitudes_rad]

    return x, y

EARTH_RADIUS = 6371 # in kilometers

# Function to calculate the planar positions on the earth's surface
def xy_on_earth(latitudes, longitudes):
    """
    Returns the X & Y coordinates on the Earth's surface

    Parameters:
    Inputs:
    latitudes (list or array), longitudes (list or array): angular positions on the Earth's surface
    Outputs:
    x (list): X coordinates in the 2D plane.
    y (list): Y coordinates in the 2D plane.
    """
    return latlon_to_xy_mercator(latitudes, longitudes, EARTH_RADIUS)
