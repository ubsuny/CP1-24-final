"""
This module contains functions for processing CSV files and converting timestamps to Unix time.
"""
import datetime
from datetime import timedelta
import pandas as pd

def add_timestamp_to_data(file_path, start_time_str):
    """
    Reads the CSV file with time and acceleration data, and adds full timestamps
    by calculating them based on the start time.
    
    Parameters:
    file_path (str): Path to the CSV file.
    start_time_str (str): The start date and time in the format "YYYY-MM-DD HH:MM:SS".
    
    Returns:
    DataFrame: A DataFrame with the full timestamps and Unix times.
    """
    # Read the CSV file
    df = pd.read_csv(file_path)
    # Convert the start time string to a datetime object
    start_time = datetime.datetime.strptime(start_time_str, "%Y-%m-%d %H:%M:%S")
    # Add a new column to the DataFrame for the full timestamp
    df['timestamp'] = df['Time (s)'].apply(lambda t: start_time + timedelta(seconds=t))
    # Convert the timestamp to Unix time
    df['unix_time'] = df['timestamp'].apply(lambda x: int(x.timestamp()))
    return df

def process_multiple_files(csv_file_paths, csv_start_times):
    """
    Process multiple CSV files and convert their time data to full date-time and Unix time.
    
    Parameters:
    csv_file_paths (list of str): List of file paths for the CSV files.
    csv_start_times (list of str): List of start date-time strings for each corresponding file.
    
    Returns:
    dict: A dictionary where keys are file paths and values are DataFrames with timestamp data.
    """
    processed_data_frames = {}
    # Loop through each file and its corresponding start time
    for file_path, start_time_str in zip(csv_file_paths, csv_start_times):
        print(f"Processing {file_path} with start time {start_time_str}")
        # Process each file and store the result in the dictionary
        df_with_timestamps = add_timestamp_to_data(file_path, start_time_str)
        processed_data_frames[file_path] = df_with_timestamps
        # Optionally, save the DataFrame with timestamps to a new CSV file
        output_file = file_path.replace('.csv', '_with_timestamps.csv')
        df_with_timestamps.to_csv(output_file, index=False)
        print(f"Saved {output_file}")
    return processed_data_frames

# Example usage
# List of CSV file paths (replace with your actual file paths)
file_paths = [
    'VS001_acceleration.csv',
    'VS002_acceleration_return.csv',
    'VS003_gps_circle.csv',
    'VS004_gps_triangle.csv'
]
# List of start date and time strings corresponding to each file
start_times = [
    "2024-10-23 17:16:00",  # Start time for the first file
    "2024-10-23 17:12:00",  # Start time for the second file
    "2024-10-24 01:50:00",  # Start time for the third file
    "2024-10-24 02:30:00"   # Start time for the fourth file
]
# Process the files and convert their time data
data_frames = process_multiple_files(file_paths, start_times)
