'''
This module tests the function in final.py to make sure 
they give what we expect in edge cases
'''

import pytest
from final import (fahrenheit_to_kelvin, get_temp,
                   list_files, is_equidistant)

def test_fahrenheit_to_kelvin():
    """
    Test the fahrenheit_to_kelvin function abides by what is expected for the freezing point 
    of water, absolute zero, and the boiling point of water
    """
    assert fahrenheit_to_kelvin(32) == pytest.approx(273.15, rel=1e-6)
    assert fahrenheit_to_kelvin(-459.67) == pytest.approx(0, rel=1e-6)
    assert fahrenheit_to_kelvin(212) == pytest.approx(373.15, rel=1e-6)

def test_get_temp():
    ''' 
    Tests get_temp gets the correct temperature from the md files
    '''
    temp = get_temp('/workspaces/CP1-24-final/iglesias-cardinale/data/Final/ic001_sinewalk.md')
    assert temp == 38

def test_list_files():
    '''
    Tests that the number of output files is what we expect
    '''
    num = len(list_files('/workspaces/CP1-24-final/iglesias-cardinale/data/Final', 'ic0', '.md'))

    assert num == 10

def test_is_equidistant():
    '''
    Tests that equidistant data is shown to be equidistant
    '''

    assert is_equidistant([0,1,2]) == 'Your data is equidistant!'
    assert is_equidistant([0,1,3]) == 'Your data is not equidistant :('
