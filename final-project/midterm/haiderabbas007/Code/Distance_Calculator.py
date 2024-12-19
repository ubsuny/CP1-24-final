"""
This module provides functions to calculate distances between GPS points (both arbitrary and adjacent).
Data used is latitude, longitude, and elevation data from CSV files.

Functions included:

1. calculate_average_latitude(circle_path, triangle_path):
   - Calculates the average latitude for combined Circle and Triangle datasets.

2. calculate_average_elevation(circle_path, triangle_path):
   - Calculates the average elevation for combined Circle and Triangle datasets.

3. calculate_earth_radius(latitude, elevation):
   - Calculates the radius of the Earth at a given latitude and elevation using the WGS84 reference ellipsoid formula.

4. calculate_distance(x1, y1, x2, y2, earth_radius):
   - Calculates the distance between two GPS points using the Haversine formula, considering the curvature of the Earth.

5. calculate_adjacent_distances(path):
   - Calculates the distances between adjacent GPS points in a dataset using the Haversine formula.
"""

import numpy as np
import pandas as pd

#  Reference: https://nssdc.gsfc.nasa.gov/planetary/factsheet/earthfact.html
RADIUS_EQUATOR = 6378137  # Equatorial radius of Earth in meters
RADIUS_POLE = 6356752  # Polar radius of Earth in meters

CIRCLE_PATH = "haiderabbas007/Data/Raw Data (.csv)/Circle.csv" # Path to Circle Data CSV file
TRIANGLE_PATH = "haiderabbas007/Data/Raw Data (.csv)/Triangle.csv" # Path to Triangle Data CSV file

def calculate_average_latitude(circle_path, triangle_path):
    """
    Calculates the average latitude for the combined Circle and Triangle dataset and returns the total average latitude.

    Parameters:
        circle_path (str): Path to the Circle CSV file.
        triangle_path (str): Path to the Triangle CSV file.

    Returns:
        float: The overall average latitude for both datasets.
    """

    # Load data for the Circle Run
    circle_data = pd.read_csv(circle_path)

    # Load data for the Triangle Run
    triangle_data = pd.read_csv(triangle_path)

    # Calculate the average latitude for Circle
    circle_data_latitude = circle_data['Latitude (°)']
    circle_latitude_average = circle_data_latitude.mean()

    # Calculate the average latitude for Triangle
    triangle_data_latitude = triangle_data['Latitude (°)']
    triangle_latitude_average = triangle_data_latitude.mean()

    # Calculate the overall average latitude
    total_average_latitude = (circle_latitude_average + triangle_latitude_average) / 2

    return total_average_latitude

def calculate_average_elevation(circle_path, triangle_path):
    """
    Calculates the average elevation for the combined Circle and Triangle dataset and returns the total average elevation.

    Parameters:
        circle_path (str): Path to the Circle CSV file.
        triangle_path (str): Path to the Triangle CSV file.

    Returns:
        float: The overall average elevation for both datasets.
    """

    # Load data for the Circle Run
    circle_data = pd.read_csv(circle_path)

    # Load data for the Triangle Run
    triangle_data = pd.read_csv(triangle_path)

    # Calculate the average elevation for Circle
    circle_data_elevation = circle_data['Altitude (m)']
    circle_elevation_average = circle_data_elevation.mean()

    # Calculate the average elevation for Triangle
    triangle_data_elevation = triangle_data['Altitude (m)']
    triangle_elevation_average = triangle_data_elevation.mean()

    # Calculate the overall average elevation
    total_average_elevation = (circle_elevation_average + triangle_elevation_average) / 2

    return total_average_elevation

def calculate_earth_radius(latitude, elevation):
    """
    Calculates the radius of the Earth at a given latitude and elevation.

    Parameters:
        latitude (float): Latitude in degrees.
        elevation (float): Elevation in meters.

    Returns:
        float: The radius of the Earth in meters.
    """
    # Convert latitude from degrees to radians
    phi = np.radians(latitude)

    # Calculate the radius of Earth at the given latitude using WGS84 reference ellipsoid formula
    # Reference: National Geospatial-Intelligence Agency. WGS 84: The World Geodetic System 1984.
    # Available at: https://earth-info.nga.mil/
    numerator = (RADIUS_EQUATOR**2 * np.cos(phi))**2 + (RADIUS_POLE**2 * np.sin(phi))**2
    denominator = (RADIUS_EQUATOR * np.cos(phi))**2 + (RADIUS_POLE * np.sin(phi))**2
    radius_at_latitude = np.sqrt(numerator / denominator)

    # Add elevation to the radius
    total_radius = radius_at_latitude + elevation

    return total_radius

def calculate_distance(x1, y1, x2, y2):
    """
    Calculates the distance between two GPS points.
    Takes into account the curvature of Earth by using the Haversine Formula.

    Parameters:
        x1 (float): Latitude of the first point.
        y1 (float): Longitude of the first point.
        x2 (float): Latitude of the second point.
        y2 (float): Longitude of the second point.
    Returns:
        float: Distance between the two GPS points in meters.
    """

    # Convert latitude in degrees to radians
    phi1 = np.radians(x1)
    phi2 = np.radians(x2)

    # Convert longitude in degrees to radians
    lambda1 = np.radians(y1)
    lambda2 = np.radians(y2)

    delta_phi = phi2 - phi1
    delta_lambda = lambda2 - lambda1

    # Calculate average latitude and radius of Earth at location of data collection
    total_average_latitude = calculate_average_latitude(CIRCLE_PATH, TRIANGLE_PATH)
    total_average_elevation = calculate_average_elevation(CIRCLE_PATH, TRIANGLE_PATH)
    earth_radius = calculate_earth_radius(total_average_latitude, total_average_elevation)

    # Calculate distance between GPS points using Haversine Formula
    # Reference: "Heavenly Mathematics", Glen Robert (2013), Princeton University Press
    a = np.sin(delta_phi / 2) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(delta_lambda / 2) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

    return earth_radius * c

def calculate_adjacent_distances(path):
    """
    Calculates the distance between adjacent GPS points in the dataset using the Haversine formula.

    Parameters:
        path (str): Path to the CSV file containing latitude and longitude data.

    Returns:
        list: A list of distances between adjacent points in meters.
    """
    # Load the data
    data = pd.read_csv(path)

    # Extract latitude and longitude values
    latitudes = data['Latitude (°)'].values
    longitudes = data['Longitude (°)'].values

    # Calculate distances between adjacent points
    adjacent_distances = []
    for i in range(1, len(latitudes)):
        x1, y1 = latitudes[i - 1], longitudes[i - 1]
        x2, y2 = latitudes[i], longitudes[i]

        # Convert latitude in degrees to radians
        phi1 = np.radians(x1)
        phi2 = np.radians(x2)

        # Convert longitude in degrees to radians
        lambda1 = np.radians(y1)
        lambda2 = np.radians(y2)

        delta_phi = phi2 - phi1
        delta_lambda = lambda2 - lambda1

        # Calculate the radius of the Earth at the average latitude and elevation
        average_latitude = (x1 + x2) / 2
        average_elevation = calculate_average_elevation(CIRCLE_PATH, TRIANGLE_PATH)
        earth_radius = calculate_earth_radius(average_latitude, average_elevation)

        # Calculate the distance between adjacent GPS points using Haversine formula
        a = np.sin(delta_phi / 2)**2 + np.cos(phi1) * np.cos(phi2) * np.sin(delta_lambda / 2)**2
        c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
        distance_adjacent = earth_radius * c

        adjacent_distances.append(distance_adjacent)

    return adjacent_distances
