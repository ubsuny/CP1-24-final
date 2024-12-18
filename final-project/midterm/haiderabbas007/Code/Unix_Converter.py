"""
This module extracts date and time from metadata files and converts them into Unix time.

It reads a list of metadata files, with each file containing start and end dates and times (among other stuff).
It extracts the date and time values and converts them to Unix time using Pandas.
The results are returned as a dictionary with the file paths as keys and Unix times for 'Start'
and 'End' as values.

Functions:
    unix_time_converter(paths): Extracts date and time from each metadata file and converts it to Unix time.
"""

import pandas as pd

# List of paths of metadata files
metadata_paths = [
    "haiderabbas007/Data/Meta Data (.md)/HA007_Elevator_Down.md",
    "haiderabbas007/Data/Meta Data (.md)/HA007_Elevator_Up.md",
    "haiderabbas007/Data/Meta Data (.md)/HA007_Triangle.md",
    "haiderabbas007/Data/Meta Data (.md)/HA007_Circle.md"
]

def unix_time_converter(paths):
    """
    Extracts date and time from each metadata file and converts it into Unix time.

    Parameters:
        paths (list of str): List of paths to the metadata files.

    Returns:
        dict: A dictionary where each key is the file path and each value is another dictionary
              with 'Start' and 'End' as keys and the respective Unix times as values.
    """
    unix_times = {}

    for path in paths:
        with open(path, 'r') as file:   # Open file in read mode
            lines_list = file.readlines()   # Creates a list of strings of lines in the file
            times = {}
            for line in lines_list:
                if line.startswith("| Start"):
                    parts = line.split("|")
                    date = parts[2].strip()  # Date string from file
                    time = parts[3].strip().split(" ")[0]  # Time string with timezone removed
                    date_and_time = f"{date} {time}" # f-string used in lieu of concatenation :)

                    # Convert the date and time string to Pandas Timestamp object
                    datetime_pd = pd.to_datetime(date_and_time, format="%Y-%m-%d %H:%M:%S.%f")
                    # Convert this object to Unix time
                    unix_time = int(datetime_pd.timestamp())

                    # Store it in the dictionary
                    times['Start'] = unix_time

                elif line.startswith("| End"):
                    parts = line.split("|")
                    date = parts[2].strip()  # Date string from file
                    time = parts[3].strip().split(" ")[0]  # Time string with timezone removed
                    date_and_time = f"{date} {time}" # f-string used in lieu of concatenation :)

                    # Convert the date and time string to Pandas Timestamp object
                    datetime_pd = pd.to_datetime(date_and_time, format="%Y-%m-%d %H:%M:%S.%f")
                    # Convert this object to Unix time
                    unix_time = int(datetime_pd.timestamp())

                    # Store it in the dictionary
                    times['End'] = unix_time

            unix_times[path] = times

    return unix_times
