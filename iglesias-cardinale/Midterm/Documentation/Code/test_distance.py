'''This module tests if various distance formulas in distance.py 
return expected values in edge cases'''

import pytest
import numpy as np
from distance import gps_distance_flat_earth, r_earth, gps_wgs84, gps_distance_wgs84

def test_gps_distance_flat_earth_is_zero():
    '''
    This function tests that the distance between two equivalent points is calculated to be zero.
    '''
    p1 = np.array([1, 2, 3])
    p2 = np.array([1, 2, 3])

    assert gps_distance_flat_earth(p1,p2) == 0

def test_r_earth_at_equator():
    '''
    This function tests that r_earth = R_EQUATOR = 6378137 Meters when phi = 0
    '''

    assert r_earth(0) == 6378137

def test_gps_wgs84_at_equator():
    '''
    This function tests that the position given with altitude=latitude=longitude=0 
    returns x = R_Equator and y=z=0
    '''

    x, y, z = gps_wgs84(0, 0, 0)
    assert x == pytest.approx(6378137, rel=1e-6)
    assert y == pytest.approx(0, rel=1e-6)
    assert z == pytest.approx(0, rel=1e-6)

def test_gps_distance_wgs84_is_zero():
    '''
    This function tests that the distance between two equivalent points is calculated to be zero.
    '''

    p1 = np.array([1, 2, 3])
    p2 = np.array([1, 2, 3])

    assert gps_distance_wgs84(p1,p2) == 0
