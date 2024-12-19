"""
This module contains functions revolving around
converting time to unix time. 
"""

import pandas as pd

def years(y):
    """
    years takes in a year input y and
    returns 2 lists defining the number 
    of days within the years and leap
    years.
    """

    the_years=list(range(1970,y))
    #nyears is a list containing 365 for each non-leapyear
    nyears=[365 for y in the_years if y%4!=0 or (y%100==0 and y%400!=0)]
    #lyears is a list containing 366 for each leapyear
    lyears=[366 for y in the_years if (y%100==0 and y%400==0) or (y%100!=0 and y%4==0)]

    #These two lists together contain the information needed to determine how many
    #days have passed from the beginning of 1970 to the input year.

    return nyears,lyears

def months_2024():
    """
    months_2024 makes a list of the number of 
    days in each month in 2024. 
    """
    jan=31
    feb=29
    mar=31
    apr=30
    may=31
    jun=30
    jul=31
    aug=31
    sep=30
    octo=31
    nov=30
    dec=31
    return [jan,feb,mar,apr,may,jun,jul,aug,sep,octo,nov,dec]

def time(date):
    """
    the time function takes in a year, month,
    day, hour, minute, and second to calculate
    unix time of that specific date.
    """

    #y1_s and y2_s are equivalent to nyears, lyears
    #multiplied by 86400, the number of seconds in
    #a day

    y1_s=(year*86400 for year in years(date[0])[0])
    y2_s=(year*86400 for year in years(date[0])[1])

    #year_sec will contain the years between
    #1970 and y in terms of seconds
    year_sec=0

    #The total seconds from the years between
    #1970 and y are added to year_sec
    for year in y1_s:
        year_sec+=year
    for year in y2_s:
        year_sec+=year

    #months is a list of the days in a month
    #per month of 2024
    months=months_2024()

    count=0

    #month_sec will contain the number of
    #whole months between the start of the year
    #and the current date in terms of seconds
    month_sec=0

    #The number of seconds per day is multiplied
    #by the days in each month and added to month_sec
    while date[1]-1>count:
        month_sec+=months[count]*86400
        count+=1

    #day_sec is the number of whole days
    #between the start of the month and
    #the current date in terms of seconds
    day_sec=(date[2]-1)*86400

    #hour_sec is the number of whole hours
    #that have passed since the start of the day
    #in units of seconds. Accounts for utc time difference
    hour_sec=(date[3]+4)*3600

    #min_sec is the number of whole minutes
    #that have passed since the start of the hour
    #in units of seconds
    min_sec=(date[4])*60

    #sec is the number of seconds that have passed
    #since the start of the minute.
    sec=date[5]

    #The sum of all the seconds between the start of 1970
    #and the current date is returned
    return year_sec+ month_sec+day_sec+hour_sec+min_sec+sec

def date_reader(path):
    """
    date_reader is meant to read time.csv files
    such that the initial date information may be 
    collected
    """
    #The file is read, the column containing the
    #desired information is isolated, and the
    #split function is used to isolate individual
    #date elements such as years, months, days, etc
    condition=False
    file=pd.read_csv(path)
    for c in file:
        if c=="system time text":
            condition=True
    if condition is True:

        date=file["system time text"][1].split(" ")
        day_o_y=date[0].split("-")
        t=date[1].split(":")

        #A new list is made for the purpose
        #of recombining the split lists
        time_info=day_o_y

        #A set of loops recombine the two lists
        #after the splitting and makes sure each
        #element is of the correct data type.
        for i in t:
            time_info.append(i)

        final_t_info=[]

        for i in time_info:
            if i!=time_info[len(time_info)-1]:
                i=int(i)
                final_t_info.append(i)
            else:
                i=float(i)
                final_t_info.append(i)

        return final_t_info
    return "Bad file. No system time text"


def time_reader(path):
    """
    time reader takes in a file path
    and gets the Time (s) column
    """

    file=pd.read_csv(path)
    condition=False
    for c in file:
        if c=="Time (s)":
            condition=True
    if condition is True:
        times=file["Time (s)"]

        return times

    return "Bad file. No time (s) data"

def unix_time_getter(path1, path2):
    """
    unix_time_getter takes in two files containing
    date and time information to return a list
    of unix time values for each time datapoint.
    """
    times=time_reader(path1)
    date=date_reader(path2)
    unix_times=[time(date)+s for s in times]
    return unix_times
