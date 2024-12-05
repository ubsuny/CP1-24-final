"""
This module convers metafile to unix time.
"""

# metafile_to_unix.py

def convert_to_unix(file_path):
    """
    Read the date and time from a metafile and convert it to Unix time.

    Parameters:
    file_path (str): The path to the metafile.

    Returns:
    int: Unix time (seconds since epoch).
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            # Assuming the datetime is on the first line in the format 'YYYY-MM-DD HH:MM:SS'
            date_time_str = file.readline().strip()

            # Parse the datetime string
            year, month, day, hour, minute, second = map(int, date_time_str.replace('-', ' ').replace(':', ' ').split())

            # Calculate Unix time (seconds since epoch)
            # Simple calculation assuming 1970-01-01 00:00:00 UTC as the epoch
            # Note: This does not account for leap years, daylight saving time, etc.
            unix_time = (
                (year - 1970) * 365 * 24 * 3600 +  # Years to seconds
                (month - 1) * 30 * 24 * 3600 +     # Months to seconds (approximation)
                (day - 1) * 24 * 3600 +             # Days to seconds
                hour * 3600 +                       # Hours to seconds
                minute * 60 +                       # Minutes to seconds
                second                               # Seconds
            )
            return unix_time
    except FileNotFoundError:
        raise FileNotFoundError(f"The file {file_path} does not exist.")
    except ValueError:
        raise ValueError("Date format should be 'YYYY-MM-DD HH:MM:SS'.")
