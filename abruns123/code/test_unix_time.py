"""
This module tests the unix_time module
"""
from unix_time import years, time, time_reader,date_reader

def test_years():
    """
    test_years verifies that the 
    years function can accurately
    create lists of length corresponding
    to the number of leap years and 
    non-leap years between 1970 and the 
    input year. 
    """
    ny,ly=years(2130)
    assert len(ny)==121
    assert len(ly)==39

def test_time():
    """
    test_time uses a known unix time 
    taken from the unix timestamp
    website and verifies that
    the time function calculates the correct
    unix time.
    """
    #Test is based on comparison with test value used 
    #on unix timestamp website
    #citation: [1][1] Unix Time Stamp - Epoch Converter. (n.d.). 
    # https://www.unixtimestamp.com/
    date=[2024,10,24,17,45,53]
    assert time(date)==1729806353

def test_reader():
    """
    test reader verifies that the various reader functions 
    can run without producing an error in the case that a 
    file without the data they're meant to read is the input.
    """
    time_reader("/abruns123/data/triangle_data/meta/time.csv")
    date_reader("/abruns123/data/triangle_ata/Raw Data.csv")
    