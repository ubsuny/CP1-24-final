'''
This module tests unix_time.py to ensure that conversions are done appropriately
'''

import numpy as np
from unix_time import date2unix, gettime

def test_unix_time_0():
    '''Tests that January 1, 2024 reads 1704067200 in UNIX Time'''
    assert date2unix(1, 1, 0, 0, 0) == 1704067200

def test_gettime_gets_correct_month():
    '''Tests that the month recorded from the gps circle data metafile is [10,10]'''

    year, month, day, hour, minute, second = gettime(
        '/iglesias-cardinale/data/ic001_gps_circle_run001.md')

    date = [year, month, day, hour, minute, second]
    assert np.all(date[1] == 10)
