"""
This module provides functions to extract date and time information from a Markdown file 
and convert it to Unix time.

Functions:
    - extract_datetime_from_md(md_content, date_pattern, time_format): 
        Extracts a date and time string from the Markdown content based on a regular
          expression pattern and converts it to Unix time.
    - read_md_file_and_convert_to_unix(file_path, date_pattern): 
        Reads a Markdown file, extracts a date and time string, and converts it to Unix time.
"""

import re
from datetime import datetime

def extract_datetime_from_md(
        md_content,date_pattern=r'\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}',
        time_format='%Y-%m-%d %H:%M:%S'
        ):
    """
    Extracts a date and time string from Markdown content based on a specified pattern 
    and converts it to Unix time.

    Args:
        md_content (str): The content of the markdown file.
        date_pattern (str): A regular expression pattern to match the date and time string. 
        Default is 'YYYY-MM-DD HH:MM:SS'.
        time_format (str): The format in which the date string is provided. 
        Default is '%Y-%m-%d %H:%M:%S'.

    Returns:
        int: Unix time (number of seconds since the Unix epoch), or None if no date is found.
    """
    # Search for the date pattern in the Markdown content
    match = re.search(date_pattern, md_content)
    if match:
        date_str = match.group()
        # Convert the matched date string to Unix time
        dt = datetime.strptime(date_str, time_format)
        unix_converted_time = int(dt.timestamp())
        return unix_converted_time
    return None

def read_md_file_and_convert_to_unix(
        file_path, date_pattern=r'\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}'
        ):
    """
    Reads a markdown file, extracts date and time, and converts it to Unix time.

    Args:
        file_path (str): Path to the markdown (.md) file.
        date_pattern (str): A regular expression pattern to match 
        the date and time string. Default is 'YYYY-MM-DD HH:MM:SS'.

    Returns:
        int: Unix time (number of seconds since the Unix epoch), or None if no date is found.
    """
    # Read the markdown file content
    with open(file_path, 'r', encoding='utf-8') as file:  # pylint: disable=unused-variable
        md_content = file.read()
    # Extract and convert the date and time to Unix time
    return extract_datetime_from_md(md_content, date_pattern)
