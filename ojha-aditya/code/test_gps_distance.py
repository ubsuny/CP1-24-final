'''This is to test module give_distance'''

import numpy as np
from gps_distance import give_distance


def test_give_distance() -> None:
    '''
    Test that give_distance returns the correct result when given dummy parameters
    '''
    latitude_arr = np.arange(2)
    longitude_arr = np.arange(2)
    altitude_arr = np.arange(2)
    r = 10
    tmp = give_distance(latitude_arr, longitude_arr, altitude_arr, r)
    assert np.isclose(tmp[0], 1.0300099)
