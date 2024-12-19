"""
This module is to define a function that converts a given date and time to Unix time
"""

from time_conversion_preliminaries import years_to_days, days_past, seconds_past

# The epoch of the Unix time
YEAR_0 = 1970

# Function to convert time from UTC to Unix
def to_unix_time(month, day, year, hr, minute, sec):
    """
    Returns the time in Unix format

    Parameters:
    Inputs:
    month (number), day (number), year (number), hr (number), minute (number), sec (number):
    the date and time desired to be converted
    Outputs:
    seconds (number): the date and time in Unix format 
    """
    days = years_to_days(YEAR_0, year - 1) + days_past(year, month, day)
    seconds = 86400 * days + seconds_past(hr, minute, sec)
    return seconds
