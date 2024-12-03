"""Test for unixtime_converter.py"""
from unixtime_converter import (
    read_text_metafile,
    is_leap_year,
    days_in_month,
    convert_to_unix_time,
    get_unix_time_from_metafile
)

def test_is_leap_year():
    """Test if the year function is working"""
    assert is_leap_year(2020) is True  # Leap year
    assert is_leap_year(2019) is False  # Not a leap year
    assert is_leap_year(2000) is True  # Divisible by 400
    assert is_leap_year(1900) is False  # Divisible by 100, but not 400

def test_days_in_month():
    "Test if the function returns the correct days of each month"
    assert days_in_month(2, 2020) == 29  # February in leap year
    assert days_in_month(2, 2019) == 28  # February in non-leap year
    assert days_in_month(1, 2020) == 31  # January
    assert days_in_month(4, 2020) == 30  # April

def test_convert_to_unix_time():
    """Test the unix time convertor function"""
    assert convert_to_unix_time('1970-01-01', '00:00:00') == 0  # Epoch time
    assert convert_to_unix_time('1970-01-02', '00:00:00') == 86400  # One day after epoch
    assert convert_to_unix_time('2024-10-21', '14:30:00') == 1729521000  # Updated correct timestamp

def test_get_unix_time_from_metafile(tmpdir):
    """Create a temporary metafile for testing"""
    metafile = tmpdir.join("test_metadata.txt")
    metafile.write("Date: 2024-10-21\nTime: 14:30:00")

    unix_time = get_unix_time_from_metafile(str(metafile))
    assert unix_time == 1729521000  # Updated correct timestamp

def test_read_text_metafile(tmpdir):
    """Test reading the metafile and parsing it correctly"""
    # Create a temporary file using pytest's tmpdir fixture
    metafile = tmpdir.join("test_metadata.txt")
    metafile.write("Date: 2024-10-21\nTime: 14:30:00")

    expected_metadata = {
        'Date': '2024-10-21',
        'Time': '14:30:00'
    }
    result = read_text_metafile(str(metafile))
    assert result == expected_metadata
