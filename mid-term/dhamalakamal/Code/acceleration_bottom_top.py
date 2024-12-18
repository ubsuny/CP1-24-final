"""
This module demonstrates the use of the `calculate_directions_from_csv` 
function from the `acceleration_calculate` module to calculate azimuth 
and elevation directions between GPS points provided in a CSV file.

The CSV file is expected to contain data that allows for the calculation 
of directional angles (azimuth and elevation) for each point in relation to others.

Functions used:
    - calculate_directions_from_csv(file_path): 
        Calculates the azimuth and elevation directions from the data provided in the CSV file.
"""

from acceleration_calculate import calculate_directions_from_csv

CSV_FILE = r'dhamalakamal/Data/Accl_bottom_top.csv'
calculated_directions = calculate_directions_from_csv(CSV_FILE)

print("Direction for points:")
for index, (azimuth, elevation) in enumerate(calculated_directions):
    print(f"{index + 1}: Azimuth = {azimuth:.2f}°, Elevation = {elevation:.2f}°")
