"""
This module provides unit testing for the metafile processing functions.
"""

import pytest
import pandas as pd
import numpy as np
import unix_time as mp

class TestReadMetafile:
    """
    Testing class for the read_metafile function.
    """

    def test_read_metafile_valid_input(self):
        """
        Test read_metafile with a valid CSV input.
        Creates valid placeholder csv, then clears it.
        """
        data = {
            'system time text': ['1999-12-31 01:00:00 UTC']
        }
        test_csv = 'test_placeholder.csv'
        pd.DataFrame(data).to_csv(test_csv, index=False)
        system_time = mp.read_metafile(test_csv)
        expected_time = '1999-12-31 01:00:00 UTC'
        assert system_time == expected_time
        pd.DataFrame().to_csv(test_csv, index=False) #Clears placeholder csv.

    def test_read_metafile_file_not_found(self):
        """
        Test that the function raises FileNotFoundError when CSV file is missing.
        """
        with pytest.raises(FileNotFoundError):
            mp.read_metafile('phony_file.csv')
    def test_read_metafile_missing_column(self):
        """
        Test that the function raises KeyError when required
        'system time text' column is missing.
        """
        data = {
            'other column': ['some data']
        }
        test_csv = 'test_placeholder.csv'
        pd.DataFrame(data).to_csv(test_csv, index=False)
        with pytest.raises(KeyError):
            mp.read_metafile(test_csv)
            pd.DataFrame().to_csv(test_csv, index=False)

    def test_read_metafile_invalid_input_type(self):
        """
        Test that the function raises TypeError when input type is incorrect.
        """
        with pytest.raises(TypeError):
            mp.read_metafile(12345)

class TestConvertSystemTimeToUnix:
    """
    Testing class for the convert_system_time_to_unix function.
    """
    def test_convert_valid_system_time(self):
        """
        Test conversion of a valid system time string to Unix time.
        """
        system_time = pd.Series(['2011-11-11 11:11:11 UTC'])
        expected_unix_adj = pd.to_datetime(system_time.str.strip().str.replace('UTC', ''), utc=True)
        expected_unix_time = expected_unix_adj.astype(np.int64) / 1e9
        unix_time = mp.convert_system_time_to_unix(system_time)
        np.testing.assert_array_almost_equal(unix_time, expected_unix_time)

    def test_convert_empty_system_time(self):
        """
        Test the function with an empty input string.
        """
        system_time = pd.Series([''])
        with pytest.raises(ValueError):
            mp.convert_system_time_to_unix(system_time)
