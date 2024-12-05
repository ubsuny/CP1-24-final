"""
This module Calculate the great-circle distance between two points on the Earth.
""" 
# gps_distance.py

import math

def haversine(lat1, lon1, lat2, lon2):
    """Calculate the great-circle distance between two points
    on the Earth (specified in decimal degrees)."""

    # Convert latitude and longitude from degrees to radians
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])

    # Haversine formula
    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = math.sin(dlat / 2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2)**2
    c = 2 * math.asin(math.sqrt(a))

    # Radius of Earth in kilometers. Use 3956 for miles
    r = 6371.0
    return c * r
