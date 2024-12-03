"""This module takes the date and time from a metafile and converts it into Unixtime."""

def read_text_metafile(filename):

    """Read metadata from a text file and return it as a dictionary."""

    metadata = {}
    with open(filename, 'r', encoding='utf-8') as file:
        for line in file:
            key, value = line.strip().split(": ", 1)
            metadata[key] = value
    return metadata


def is_leap_year(year):

    """Check if a given year is a leap year."""

    return year % 4 == 0 and (year % 100 != 0 or year % 400 == 0)


def days_in_month(month, year):

    """Get the number of days in a given month of a given year."""

    days_in_each_month = [31, 28 + is_leap_year(year), 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    return days_in_each_month[month - 1]


def convert_to_unix_time(date_str, time_str=None):

    """Convert a date (and optional time) string to Unix time."""

    # Split the date and time strings

    year, month, day = map(int, date_str.split('-'))
    if time_str:
        hours, minutes, seconds = map(int, time_str.split(':'))
    else:
        hours, minutes, seconds = 0, 0, 0

    # Calculate total days from 1970 to the given year
    days_since_epoch = 0
    for y in range(1970, year):
        days_since_epoch += 365 + is_leap_year(y)

    # Add days from the current year
    for m in range(1, month):
        days_since_epoch += days_in_month(m, year)

    days_since_epoch += (day - 1)

    # Convert everything to seconds
    total_seconds = (
        days_since_epoch * 86400 +  # Convert days to seconds
        hours * 3600 +               # Convert hours to seconds
        minutes * 60 +               # Convert minutes to seconds
        seconds                      # Add seconds
    )

    return total_seconds


def get_unix_time_from_metafile(metafile):

    """Read the date and time from a metafile and convert it to Unix time."""

    metadata = read_text_metafile(metafile)

    # Check if the required fields are in the metadata
    date = metadata.get('Date')
    time_day = metadata.get('Time')

    if not date:
        raise ValueError("Date not found in metafile.")

    # Convert date and time to Unix time
    unix_time = convert_to_unix_time(date, time_day)
    return unix_time
