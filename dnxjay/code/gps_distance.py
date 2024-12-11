"""This module calculates the distance between GPS coordinates using the Haversine formula."""

import numpy as np

# Earth radius constants in various units
EARTH_RADIUS_METERS = 6371e3
EARTH_RADIUS_KILOMETERS = 6371
EARTH_RADIUS_MILES = 3958.8

def haversine(lat1, lon1, lat2, lon2, unit="meters"):
    """Calculate the Haversine distance between two GPS coordinates."""
    if not (-90 <= lat1 <= 90 and -90 <= lat2 <= 90):
        raise ValueError("Latitude values must be between -90 and 90 degrees.")
    if not (-180 <= lon1 <= 180 and -180 <= lon2 <= 180):
        raise ValueError("Longitude values must be between -180 and 180 degrees.")

    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    delta_phi = np.radians(lat2 - lat1)
    delta_lambda = np.radians(lon2 - lon1)

    a = (np.sin(delta_phi / 2) ** 2 +
         np.cos(phi1) * np.cos(phi2) * np.sin(delta_lambda / 2) ** 2)
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

    if unit == "meters":
        radius = EARTH_RADIUS_METERS
    elif unit == "kilometers":
        radius = EARTH_RADIUS_KILOMETERS
    elif unit == "miles":
        radius = EARTH_RADIUS_MILES
    else:
        raise ValueError("Invalid unit. Choose 'meters', 'kilometers', or 'miles'.")

    return radius * c

def cumulative_distance(lat_vals, lon_vals, unit="meters"):
    """Calculate cumulative distance across latitude and longitude points."""
    if len(lat_vals) != len(lon_vals):
        raise ValueError("Latitude and longitude arrays must have the same length.")
    distances = [
        haversine(lat_vals[i], lon_vals[i], lat_vals[i+1], lon_vals[i+1], unit=unit)
        for i in range(len(lat_vals) - 1)
    ]
    return np.cumsum(distances)

def total_distance(lat_vals, lon_vals, unit="meters"):
    """Calculate the total distance across latitude and longitude points."""
    cumulative_distances = cumulative_distance(lat_vals, lon_vals, unit=unit)
    return cumulative_distances[-1] if len(cumulative_distances) > 0 else 0
