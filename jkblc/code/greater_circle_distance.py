"""Returns the greater circle distance between two points given a pair of coordinates"""

import math

def haversine(coord1, coord2):
    """
    Calculate the great-circle distance between two points on Earth 
    given their latitude and longitude in decimal degrees.
    """
    # Radius of the Earth in meters
    radius = 6371000

    lat1, lon1 = coord1
    lat2, lon2 = coord2

    # Convert decimal degrees to radians
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])

    # Haversine formula
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    # Distance in meters
    return radius * c
