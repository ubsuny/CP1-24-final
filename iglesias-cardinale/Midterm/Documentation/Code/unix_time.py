"""
This module converts time given in year, month, day, hour, minute, second format into UNIX time. 
Unix time is a time measurement measuring the number of elapsed seconds since January 1, 1970
"""
import datetime
import numpy as np

def gettime(path):
    '''
    This function extracts date and time information from markdown metadata files.
    The metadata files must have the time data starting in row 10 and column 4 (beginning index 
    of 1). The data must be formatted in order of year, month, day, hour, minute, second and
    separated by commas.
    
    Parameters:
    - path: String type path to markdown file: '/path/to/markdown/file/meta_data_file.md'

    Returns:
    - year: The year.
    - month: The month (1-12).
    - day: The day of the month (1-31).
    - hour: The hour (0-23).
    - minute: The minute (0-59).
    - second: The second (0-59).

    '''
    year, month, day, hour, minute, second = np.loadtxt(
    path ,
    skiprows=9,
    usecols=(3,4,5,6,7,8),
    delimiter = ',',
    unpack=True)

    return year, month, day, hour, minute, second

def date2unix(month, day, hour, minute, second):
    """
    Converts a date and time into UNIX time (seconds since January 1, 1970). The year of the 
    calculated date is always 2024. This is to stop pylint from complaining about too many 
    positional arguments.
    
    Parameters:.
    - month: Integer. The month (1-12).
    - day: Integer. The day of the month (1-31).
    - hour: Integer. The hour (0-23).
    - minute: integer. The minute (0-59).
    - second: Integer. The second (0-59).
    
    Returns:
    - The number of seconds since January 1, 1970 (UNIX time).
    """

    dt = datetime.datetime(2024, month, day, hour, minute, second)
    unix_time = int((dt - datetime.datetime(1970, 1, 1)).total_seconds())
    return unix_time
