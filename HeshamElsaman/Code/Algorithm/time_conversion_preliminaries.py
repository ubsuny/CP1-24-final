"""
This module is to define the preliminary functions needed to convert to Unix time
"""

# The number of days in the (n + 1) month
monthdays = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]

# Function to check leap years
def check_leap(year):
    """
    Returns True (False) if the year is (not) a leap year.

    Parameters:
    Inputs:
    year (number): the year needed to be checked
    Outputs:
    Boolean
    """
    if year % 4 == 0:
        if year % 100 == 0:
            if year % 400 == 0:
                return True
            return False
        else:
            return True
    return False

# Function that calculates the total number of days given two years
def years_to_days(year_1, year_2):
    """
    Returns the total number of days between two years (including the two years)

    Parameters:
    Inputs:
    year_1 (number), year_2 (number): the start and end year respectively
    Outputs:
    days (number): the number of days between those two years (including them)
    """
    year_1, year_2 = min([year_1, year_2]), max([year_1, year_2])
    yeardif = year_2 - year_1
    days = 0
    years = [ year_1 + i for i in range(yeardif + 1)]
    for j in years:
        if check_leap(j):
            days += 366
        else:
            days += 365
    return days

# Function to calculate the number of days past in a given year since its beginning
def days_past(year, month, day):
    """
    Returns the number of days past in a year since its beginning given the date

    Parameters:
    Inputs:
    year (number), month (number), day (number): the end date
    Outputs:
    days (number): the desired number of days
    """
    days = day - 1
    if (check_leap(year)) & (month > 2): days += 1
    for i in range(month - 1):
        days += monthdays[i]
    return days

# Function that calculates the passed seconds in a day since its beginning
def seconds_past(hr, min, sec):
    """
    Returns the number of seconds past in a day given a specific time

    Parameters:
    Inputs:
    hr (number), min (number), sec (number): the hour, minute, and second of the given time
    Outputs:
    seconds (number): the desired number of seconds passed
    """
    seconds = sec + 60 * min + 3600 * hr
    return seconds
