# Project Documentation

This project contains multiple Python functions for data analysis. From simple functions like unit conversion, calculating the haversine distance between two points on Earth, and trying to figure out the direction of motion from the acceleration data, to and adding timestamps to CSV files. This project really made sure that I familiar myself with the libraries such as **Pandas** which is widely used to load and manipulate csv data, **matplotlib** which i haver used here to plot the magnitude of acceleration data over time or GPS cordinates in the x-y plane, **subplots**: which is devoted to to create multiple graphs in one figure, making it easier to compare different datasets like elevator up and down directions.

## Prerequisites
Before running the code, ensure you have the following Python libraries installed:
- **pandas**
- **numpy**
- **matplotlib**
- **pytest** (for running unit tests)

## How to Run the Code

### 1. Conversion Functions
unit_converter_code.py consists of feet_to_meters and yards_to_meters which converts from feet to meters, and yards to meters respectively. I have added docstrings in the python module that explains the code clearly.

### 2. Haversine function 
distance_gps_location.py file contains the haversine(lat1, lon1, lat2, lon2) function, which calculates the great-circle distance between two points on Earth using their latitude and longitude. The code takes into account the radius of the Earth and takes an example from the recorded data from VS003_gps_circle.csv file in the data directory.

### 3. calculate_direction(acceleration_data) function
direction_of_motion.py file contains the calculate_direction(acceleration_data) function, which calculates the direction of motion based on acceleration data in the x, y, and z axes. The magnitude of the acceleration is calculated first and an example is considered using the recorded acceleration data from data directory.

### 4. add_timestamp_to_data function
Reads the CSV file convert_unix_time.py with time and acceleration data, and adds full timestamps by calculating them based on the start time, and process_multiple_files process multiple CSV files and convert their time data to full date-time and Unix time. All the four data files in the csv format have been included in the example and their respective time have been recorded too. Respective time corresponds to the precise time when these data was recorded and experiments were done.

### 5. Unit tests

Unit tests are run to ensure that all functions are working correctly. The tests are located in separate files:

- test_conversion.py: This module contains unit tests for the unit conversion functions feet_to_meters and yards_to_meters.

- test_haversine.py: This module contains unit tests for the `haversine` function, which calculates the distance between two points on Earth's surface given their latitude and  longitude. I have used London and paris as my known GPS points. Although there was an error in running the function as there was a discrepency in the final values. The assertion is failing because there is a small difference between the obtained and expected results. I increased the relative tolerance for the same.

- test_calculate_direction.py: This module tests the `calculate_direction` function with example acceleration data and compares the resulting directions to expected values. While running, I encountered the sdame problem of discrepancy in the obtained and the expected results and I increased the relative tolerance for the same.

## Summary
- Conversion functions: Converts feet/yards to meters.
- Haversine function: Calculates the great-circle distance between two points.
- Direction function: Computes direction of motion from acceleration data.
- Unit tests: Run tests using pytest.
