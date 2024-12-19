# test_modules.py
import sys
sys.path.append("/workspaces/CP1-24-midtermforked")
from Tolani4.Code.unit_converter import feet_to_meters, yards_to_meters
from Tolani4.Code.distance_calculator import haversine_distance
from Tolani4.Code.unix_time_converter import convert_to_unix
from Tolani4.Code.direction_calculator import calculate_direction
import pytest
import pandas as pd

def test_feet_to_meters():
    #print("Expected: " + 3.048 + "; result: "+ feet_to_meters(10) )
    print("anything")
    assert feet_to_meters(10) == pytest.approx(3.048, 0.001)

def test_yards_to_meters():
    assert yards_to_meters(5) == pytest.approx(4.572, 0.001)

def test_haversine_distance():
    # Example GPS points close to each other
    assert haversine_distance(40.748817, -73.985428, 40.748917, -73.985328) == pytest.approx(13.9, 0.1)

def test_convert_to_unix():
    
    print(convert_to_unix("2024-10-25", "14:30"))
    assert convert_to_unix("2024-10-25", "14:30") == pytest.approx(1729866600)

def test_calculate_direction():
    # Sample acceleration data as a DataFrame
    data = pd.DataFrame({"x": [1, 0], "y": [0, 1], "z": [0, 0]})
    assert calculate_direction(data) == [0, 90]  # Approximate expected angles