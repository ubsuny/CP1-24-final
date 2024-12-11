"""Unit tests for unixtime_converter module, testing time conversion functions."""

from unixtime_converter import to_unix_time, from_unix_time, validate_datetime_format

def test_to_unix_time():
    """Test the to_unix_time function with known values and edge cases."""
    assert to_unix_time("January 1, 1970", "12:00 AM") == 0
    assert to_unix_time("February 29, 2020", "12:00 AM") == 1582934400  # Leap year
    assert to_unix_time("January 1, 1970", "01:00 AM", adjustment_seconds=3600) == 7200

def test_from_unix_time():
    """Test the from_unix_time function for known conversions."""
    assert from_unix_time(0) == "January 01, 1970 12:00 AM"
    assert from_unix_time(1483228799) == "December 31, 2016 11:59 PM"

def test_validate_datetime_format():
    """Test datetime format validation."""
    assert validate_datetime_format("January 1, 1970", "12:00 AM") is True
    assert validate_datetime_format("February 29, 2020", "11:59 PM") is True
    assert validate_datetime_format("February 30, 2021", "12:00 PM") is False
