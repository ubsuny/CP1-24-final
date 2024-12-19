'''
Module to implement unit test for Unix time converter
'''

from unix_time_converter import unix_date, unix_time

def test_unix_date():
    '''
    Function to check Unix date
    '''
    assert unix_date(1970, 1, 1) == 0
    assert unix_date(1970, 1, 2) == 86400

def test_unix_time():
    '''
    Function to test unix time
    '''
    assert unix_time("1970-01-01", "00-00-00") == 0
    assert unix_time("2024-10-24", "23-07-09") == 1729811229
    # Unix timestamp for one of the experiments
