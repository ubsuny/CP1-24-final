'''
This module reads date and time from meta file and converts to Unix time
'''

def read_date_time(meta_file):
    '''
    Function to read date and time values from the meta file
    '''
    with open(meta_file, 'r', encoding='utf-8') as m_f:
        metadata = m_f.readlines()

    date_time = {}
    for line in metadata:
        if line.startswith("Date:"):
            date_time['date'] = line.split(":")[1].strip()
        elif line.startswith("Time:"):
            date_time['time'] = line.split(":")[1].strip()

    return date_time

def check_leap_year(year):
    '''
    Function to check if a year is a leap year, returns true or false
    '''
    return (year % 4 == 0 and year % 100 != 0) or (year % 400 == 0)

def no_of_days(month, year):
    '''
    Function to return the number of days in a month
    '''
    days = [31, 28, 31, 30, 31, 30,
                     31, 31, 30, 31, 30, 31]    # define the array of all months
    if month == 2 and check_leap_year(year):
        return 29   # return 29 days for February of a leap year
    return days[month - 1]  # return days from the month array

def unix_date(year, month, day):
    '''
    Function to convert a date to Unix time (seconds since 1970-01-01)
    '''
    total_days = 0  # variable to count number of days

    for y in range(1970, year):
        total_days += 365 + (1 if check_leap_year(y) else 0)
        # add no of days till current year

    for m in range(1, month):
        total_days += no_of_days(m, year)
        # add no of days of the current year

    total_days += day - 1
    # add days of current month beginning from 1st day of the month

    return total_days * 86400  # Convert days to seconds

def seconds_since_midnight(hour, minute, second):
    '''
    Function to convert time to seconds since midnight
    '''
    return hour * 3600 + minute * 60 + second

def unix_time(date, time):
    '''
    Function to convert date and time strings to Unix time
    '''
    year, month, day = map(int, date.split('-'))
    hour, minute, second = map(int, time.split('-'))

    date_seconds = unix_date(year, month, day)
    time_seconds = seconds_since_midnight(hour, minute, second)

    return date_seconds + time_seconds
