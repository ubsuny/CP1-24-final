'''
This module tests various functions in unit_converter.py to ensure they are correctly converting 
units
'''

import pytest
from unit_converter import meters2feet, feet2meters

def test_meters_to_feet():
    '''Tests that one meter is converted to 3.28084 feet'''
    assert meters2feet(1) == pytest.approx(3.28084, rel=1e-6)

def test_feet_to_meters():
    '''Tests that one foot is converted to 0.3048 meters'''
    assert feet2meters(1) == pytest.approx(0.3048, rel=1e-6)
