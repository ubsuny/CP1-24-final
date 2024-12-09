"""Test unix conversions"""

import tempfile
from unix_convert import convert_to_unix_time, read_metafile_and_extract_unix

def test_convert_to_unix_time():
    """Test conversion to unix time"""
    unix_time = convert_to_unix_time("2024-10-20 12:00:00", "%Y-%m-%d %H:%M:%S")
    assert unix_time == 1729425600  # Expected Unix timestamp in UTC

def test_read_metafile_and_extract_unix():
    """Tests reading metafile and extracting unix"""
    with tempfile.NamedTemporaryFile(mode='w+', delete=False) as temp_file:
        temp_file.write("2024-10-20 12:00:00\n2024-10-21 13:30:00\n")
        temp_file_path = temp_file.name

    unix_times = list(read_metafile_and_extract_unix(temp_file_path))
    assert unix_times == [1729425600, 1729517400]  # Adjusted Unix times in UTC
