# About
This project consists of multiple Python modules to process and analyze physical motion paths. Data is collected from Phyphox using GPS location and acceleration sensors.

The project includes functionality for converting units, calculating distances between GPS points, determining motion directions, and converting date and time strings into Unix timestamps.

## Modules
The code is organized into individual modules, each with a specific functionality.

### unit_converter.py Module
This module provides functions to convert distances from feet and yards into meters.

- **Functions**:
  - `feet_to_meters`: Converts feet to meters.
  - `yards_to_meters`: Converts yards to meters.

### gps_distance.py Module
This module provides functions to calculate distances between GPS coordinates using the Haversine formula. It supports calculating:
- **Functions**:
  - `haversine`: Calculates the distance between two GPS coordinates, accounting for Earth’s curvature.
  - `cumulative_distance`: Calculates cumulative distances across a series of GPS points.
  - `total_distance`: Calculates the total distance traveled across multiple GPS points.

### motion_direction.py Module
This module calculates the direction of motion based on x, y, and z acceleration data.

- **Function**:
  - `calculate_direction`: Calculates the direction in degrees using acceleration components in the x and y axes.

### unixtime_converter.py Module
This module converts date and time strings into Unix timestamps and vice versa.

- **Functions**:
  - `to_unix_time`: Converts a date and time string to Unix time.
  - `from_unix_time`: Converts Unix time back to a readable date and time string.
  - `validate_datetime_format`: Validates if a date and time string matches a given format.

## Testing Modules
Unit tests for each module are provided in separate files with names starting with `test_`, designed to run with `pytest`.

- **Testing Framework**: `pytest` is used for running the tests, while the Python `unittest` module is used for mocking data fixtures.

## How to Run

### Environment Setup
1. **Navigate to Project Directory**:
   ```bash
   cd /workspaces/CP1-24-midterm/dnxjay
