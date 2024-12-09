"""Reads out date and time form each metafile and converts to unix time"""

from datetime import datetime

def convert_to_unix_time(date_string, date_format="%Y-%m-%d %H:%M:%S"):
    """
    Convert a date string with a given format to Unix time.
    """
    dt = datetime.strptime(date_string, date_format)
    unix_time = int(dt.timestamp())
    return unix_time

def read_metafile_and_extract_unix(file_path):
    """
    Read a metafile, extract date and time, and convert it to Unix time.
    Assumes each line contains a timestamp in the format: YYYY-MM-DD HH:MM:SS.
    """
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            date_string = line.strip()
            yield convert_to_unix_time(date_string)
