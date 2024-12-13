'''
unit testing module for unit_converter.py and motion_direction.py
'''
import pytest

from unit_converter import feet_to_meters, yards_to_meters

from motion_direction import calculate_direction

def test_feet_to_meters():
    '''
    function to test the foot to meter conversion
    '''
    assert feet_to_meters(1) == 0.3048   # checking if 1 foot = 0.3048 meter or not

def test_yards_to_meters():
    '''
    function to test the yards to meters conversion
    '''
    assert yards_to_meters(1) == 0.9144   # checking if 1 yard = 0.9144 meter or not

def test_calculate_direction():
    # Sample acceleration data as a DataFrame
    data = pd.DataFrame({"x": [1, 0], "y": [0, 1], "z": [0, 0]})
    assert calculate_direction(data) == [0, 90]  # Approximate expected angles
