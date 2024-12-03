"""
This module tests the distance_calc module
"""
from distance_calc import diffm, reader

def test_reader():
    """
    test reader verifies that the various reader functions 
    can run without producing an error in the case that a 
    file without the data they're meant to read is the input.
    """
    reader("/abruns123/data/triangle _data/meta/time.csv")

def test_diffm():
    """
    test_diffm confirms that diffm
    returns an object with length 2.
    This is important as there need to 
    be two indices to represent x and y
    """
    lat1=43.008895100
    lat2=43.008135760
    lon1=-78.785005700
    lon2=-78.784864890
    assert len(diffm(lat1,lat2,lon1,lon2))==2
