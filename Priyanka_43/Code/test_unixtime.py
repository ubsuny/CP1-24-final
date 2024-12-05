"""
This module contains functions for processing CSV files and converting timestamps to Unix time.
"""
import pandas as pd
from metafile_to_unix import convert_to_unix

def test_convert_to_unix(file_path):
    """
    Read the date and time from a metafile and convert it to Unix time.

    Parameters:
    file_path (str): The path to the metafile.

    Returns:
    int: Unix time (seconds since epoch).
    """
    try:
        # Read the first line of the file
        date_time_str = pd.read_csv(file_path, header=None, nrows=1).iloc[0, 0]
    
        # Convert the datetime string to Unix time
        date_time_obj = pd.to_datetime(date_time_str, format='%Y-%m-%d %H:%M:%S')
        unix_time = int(date_time_obj.timestamp())
   
    return unix_time

    except FileNotFoundError:
        raise FileNotFoundError(f"The file {file_path} does not exist.")
    except ValueError:
        raise ValueError("Date format should be 'YYYY-MM-DD HH:MM:SS'.")
