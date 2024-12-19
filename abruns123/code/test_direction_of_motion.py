"""
This module tests the direction_of_motion module
"""
from direction_of_motion import get_direction, acc_reader

def test_direction():
    """
    test_direction verifies that the start and end behaviors
    of the two elevator experiments are as expected
    """
    acc,times=acc_reader("/abruns123/data/down_data/Raw Data.csv")
    directions=get_direction(times, acc)
    assert directions[0]==0
    assert directions[len(directions)-1]==0

    acc,times=acc_reader("/abruns123/data/up_data/Raw Data.csv")
    directions=get_direction(times, acc)
    assert directions[0]==0
    assert directions[len(directions)-1]==0