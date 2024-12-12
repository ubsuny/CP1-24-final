"""
This module provides functions to calculate distances between GPS coordinates
using the Haversine formula. It includes a function to read GPS coordinates 
from a CSV file and compute distances between adjacent coordinates.

Functions:
- haversine(latitude1, longitude1, latitude2, longitude2): Calculates the great-circle distance 
  between two points on the Earth specified by their latitude and longitude.
- read_gps_from_csv(file): Reads GPS coordinates from a CSV file and returns them as a list 
  of tuples.
- calculate_distances(gps_coords): Calculates the distances between adjacent GPS coordinates.
"""

import csv
import math

def haversine(latitude1, longitude1, latitude2, longitude2):
    """
    Calculate the great-circle distance between two points 
    on the Earth specified by their latitude and longitude.

    Parameters:
    latitude1: Latitude of the first point in degrees.
    longitude1: Longitude of the first point in degrees.
    latitude2: Latitude of the second point in degrees.
    longitude2: Longitude of the second point in degrees.

    Returns:
    float: Distance between the two points in kilometers.
    """

    # Convert latitude and longitude from degrees to radians
    latitude1, longitude1, latitude2, longitude2 = map(
        math.radians, [latitude1, longitude1, latitude2, longitude2]
    )

     # Haversine formula
    dlatitude = latitude2 - latitude1
    dlongitude = longitude2 - longitude1
    a = (math.sin(dlatitude / 2)**2 +
         math.cos(latitude1) * math.cos(latitude2) *
         math.sin(dlongitude / 2)**2)
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    # Radius of Earth in kilometers (use 3956 for miles)
    radius_km = 6371
    distance = radius_km * c
    return distance

def read_gps_from_csv(file):
    """
    Read GPS coordinates (latitudes and longitudes) from a CSV file.

    Returns:
    list of tuple: A list of (latitude, longitude) tuples.
    """
    gps_coords = []
    with open(file, mode='r', encoding='utf-8') as csvfile:
        csvreader = csv.reader(csvfile)
        next(csvreader)  # Skip the header
        for row in csvreader:
            lat, lon = float(row[1]), float(row[2])
            gps_coords.append((lat, lon))
    return gps_coords

def calculate_distances(gps_coords):
    """
    Calculate distances between adjacent GPS coordinates.

    Parameters:
    gps_coords (list of tuple): List of (latitude, longitude) tuples.

    Returns:
    list of float: List of distances between adjacent points in kilometers.
    """
    distances = []
    for i in range(len(gps_coords) - 1):
        lat1, lon1 = gps_coords[i]
        lat2, lon2 = gps_coords[i + 1]
        distance = haversine(lat1, lon1, lat2, lon2)
        distances.append(distance)
    return distances
