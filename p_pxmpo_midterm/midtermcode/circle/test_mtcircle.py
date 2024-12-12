"""
Unit tests for functions in the `mtcirclefuncts` module.

This module includes tests for:
1. `haversine`: Calculates the great-circle distance between two points on the Earth
   given their latitude and longitude.
2. `calculate_distances`: Calculates the pairwise distances between multiple GPS
   coordinates using the haversine formula.

The tests use known GPS coordinates to validate the correctness of the distance
calculations.
"""
import sys
import math
import pytest
from mtcirclefuncts import haversine, calculate_distances
sys.path.append('/workspaces/FORKCP1-24-final/p_pxmpo_midterm/midtermcode/circle')

def test_haversine():
    """
    Test for the `haversine` function.

    Verifies that the function correctly calculates the great-circle distance between
    two known points on Earth.

    Coordinates:
        Warsaw (lat1, lon1): 42.99936406°N, -78.79109360°W
        Rome (lat2, lon2): 42.99936473°N, -78.79110473°W

    The expected distance between the two points is approximately 0.000908 km.
    """
    # Example from data
    lat1, lon1 = 4.299936406E1, -7.879109360E1
    lat2, lon2 = 4.299936473E1, -7.879110473E1
    expected_distance = 0.000908193

    distance = haversine(lat1, lon1, lat2, lon2)
    assert math.isclose(distance, expected_distance, abs_tol=0.1)

def test_calculate_distances():
    """
    Test for the `calculate_distances` function.

    Ensures that the function correctly calculates the pairwise distances between
    multiple GPS coordinates.
    
    Coordinates:
        - Point 1: 42.99936406°N, -78.79109360°W (Warsaw)
        - Point 2: 42.99936473°N, -78.79110473°W (Rome)

    The expected distance between the two points is approximately 0.000908 km.
    """
    gps_coords = [(4.299936406E1, -7.879109360E1), (4.299936473E1, -7.879110473E1)]
    expected_distances = [0.000908193]

    distances = calculate_distances(gps_coords)
    assert len(distances) == 1
    assert math.isclose(distances[0], expected_distances[0], abs_tol=0.1)

# Run the tests if this file is executed directly
if __name__ == "__main__":
    pytest.main()
