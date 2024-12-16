"""
This module contains a function to calculate the great-circle distance 
between two geographic points on Earth using the Haversine formula.
"""
import numpy as np
def haversine(lat1, lon1, lat2, lon2):
    """
    Calculate the great-circle distance between two points on Earth using the Haversine formula.

    Parameters:
    lat1, lon1: Latitude and Longitude of point 1 (in decimal degrees).
    lat2, lon2: Latitude and Longitude of point 2 (in decimal degrees).

    Returns:
    float: Distance between the two points in meters.
    """
    # Convert latitude and longitude from degrees to radians
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    # Haversine formula
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    # Radius of the Earth in meters
    earth_radius = 6378137
    return earth_radius * c  # Distance in meters

def example_usage():
    """
    Example usage of the haversine function using sample coordinates.
    """
    # Example from VS003_gps_circle.csv
    lat1, lon1 = 4.299989630E1, -7.879087550E1
    lat2, lon2 = 4.300014220E1, -7.879150780E1

    distance = haversine(lat1, lon1, lat2, lon2)
    print(f"Distance: {distance} meters")

if __name__ == "__main__":
    example_usage()
