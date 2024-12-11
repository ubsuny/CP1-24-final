"""Module to convert date and time strings into Unix timestamps and validate date formats."""

from datetime import datetime

def to_unix_time(date_str, time_str, date_format="%B %d, %Y",
                 time_format="%I:%M %p", adjustment_seconds=0):
    """Convert date and time strings to a Unix timestamp."""
    datetime_str = f"{date_str} {time_str}"
    full_format = f"{date_format} {time_format}"

    try:
        dt_obj = datetime.strptime(datetime_str, full_format)
    except ValueError as exc:
        raise ValueError(
            f"Error: Provided date or time does not match the format: '{full_format}'"
        ) from exc

    unix_time = int(dt_obj.timestamp()) + adjustment_seconds
    return unix_time

def from_unix_time(unix_timestamp, output_format="%B %d, %Y %I:%M %p"):
    """Convert a Unix timestamp to a formatted date and time string."""
    dt_obj = datetime.fromtimestamp(unix_timestamp)
    return dt_obj.strftime(output_format)

def validate_datetime_format(date_str, time_str, date_format="%B %d, %Y", time_format="%I:%M %p"):
    """Validate if the date and time strings match the specified formats."""
    datetime_str = f"{date_str} {time_str}"
    full_format = f"{date_format} {time_format}"

    try:
        datetime.strptime(datetime_str, full_format)
        return True
    except ValueError:
        return False
