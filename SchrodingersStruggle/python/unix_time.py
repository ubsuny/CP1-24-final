"""
This module is responsible for reading the metadata files which have the time in them,
and converting that time to unix form.
"""

import pandas as pd
import numpy as np

def read_metafile(csv_file):
    """
    Reads the metafile CSV and returns the first entry of 'system time text' column.

    Parameters:
    csv_file (str): Path to the CSV metafile.

    Returns:
    system_time_text (Series): Pandas Series containing date-time strings.
    """
    if not isinstance(csv_file, str):
        raise TypeError("csv_file must be a string representing the file path.")
    try:
        open_file = pd.read_csv(csv_file)
        system_time = open_file['system time text'].iloc[0]
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"File '{csv_file}' was not found.") from exc
    except KeyError as exc:
        raise KeyError("CSV file must contain a 'system time text' column.") from exc
    return system_time

def convert_system_time_to_unix(system_time):
    """
    Converts date-time string in 'system time text' to unix time.

    Starts by getting rid of UTC and whitespace, then
    uses default pandas features to get unix time.

    Parameters:
    system_time (str): Pandas Series containing date-time strings.

    Returns:
    unix_time (int64): system time converted to unix time values.
    """
    system_time_chopped = system_time.str.replace('UTC', '').str.strip()
    unix_times = pd.to_datetime(system_time_chopped, utc=True, errors='coerce')
    if unix_times.isnull().any():
        failed_entries = system_time_chopped[unix_times.isnull()]
        raise ValueError(f"Failed to break down the following date-time entries:\n{failed_entries}")
    unix_times = unix_times.astype(np.int64) / 1e9
    return unix_times
