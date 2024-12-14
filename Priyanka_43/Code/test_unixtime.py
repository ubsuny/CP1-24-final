"""
This module contains functions for processing CSV files and converting timestamps to Unix time.
"""
import pytest
from convert_to_unix import convert_to_unix

def test_convert_to_unix():
    """
    functions for converting timestamps to Unix time.
    """
    print(convert_to_unix("2024-10-25", "14:30"))
    assert convert_to_unix("2024-10-25", "14:30") == pytest.approx(1729866600)
