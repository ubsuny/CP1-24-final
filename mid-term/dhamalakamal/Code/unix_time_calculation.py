"""
Unix Time Calculation Module

"""
from unix_time import read_md_file_and_convert_to_unix

MD_FILE = r'D:\Projects\CP1-24-midterm\dhamalakamal\Documentation\KD_acceleration.md'
unix_time = read_md_file_and_convert_to_unix(MD_FILE)
if unix_time is not None:
    print(f"Unix time: {unix_time}")
else:
    print("No valid date found in the markdown file.")
