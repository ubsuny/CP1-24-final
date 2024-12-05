"""
This module contains unit tests for the `haversine` function, which
Calculate the distance between two points on the Earth.
"""
import pytest
import numpy as np
from gps_distance import haversine

def test_haversine():
    """
    Tests the haversine function to verify that it calculates:
    - The distance between two points
    """
    # Test with known distance between two points
    lat1, lon1 = 42.95158835, -78.83160962  # Point 1
    lat2, lon2 = 42.95158835, -78.83161934  # point 2
    expected_distance = 0.0007911  # Approx. 0.0007911 km 
    distance = haversine(lat1, lon1, lat2, lon2)
    assert np.isclose(distance, expected_distance, rtol=1e-2)
