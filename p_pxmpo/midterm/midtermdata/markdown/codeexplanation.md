# Direction of Movement Calculation from Acceleration Data

This module calculates the direction of movement based on acceleration data read from a CSV file. It utilizes functions from the `mtaccelfuncts` module to process the acceleration values and determine the movement direction.

## Functions

- **calculate_dir(accel_file)**: Calculates the direction of movement from acceleration data.

## How to run the Code

1. **Prepare Your CSV File**: Ensure your CSV file contains the acceleration data formatted correctly. The code expects the acceleration data to be in the third column of the csv file which will only run the acceleration in the y-direction

2. **Save the Code**: Copy the provided Python code into a file.

3. **Open a Terminal**: Navigate to the directory where the file is saved.

4. **Run the Script**: Open the script and run.

# Distance Calculation Between GPS Coordinates

This module prompts the user for a CSV file containing GPS coordinates, validates the file's existence, and calculates the distances between adjacent GPS points using functions from the `mtcirclefuncts` module.

## Running the Code

1. **Prepare Your CSV File**: Ensure your CSV file contains the GPS coordinates structured correctly. The code expects the latitude to be in the second column and the longitude to be in the third column so that it can be readable by the `read_gps_from_csv` function from the `mtcirclefuncts` module.

2. **Save the Code**: Copy the provided Python code into a file.

3. **Open a Terminal**: Navigate to the directory where the file is saved.

4. **Run the Script**: Open the script and run the code.