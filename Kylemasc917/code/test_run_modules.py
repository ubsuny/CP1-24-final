"""This module contains test runs and examples on how to run modules"""

"""Test run of unix time converter module"""

import unixtime_converter
from direction_of_motion_elevator import process_acceleration_data
from distance_between_two_points import process_gps_data

# Specify the path to the metafile
metafile = 'Kylemasc917/data/Kylemascmetafile0111.txt'

# Get the Unix time from the metafile and print it

try:
    unix_time = unixtime_converter.get_unix_time_from_metafile(metafile)
    print(f"Unix time: {unix_time}")
except Exception as e:
    print(f"Error: {e}")

"""Import direction of motion function"""


# Specify the input CSV file and the output image file
csv_file = "Kylemasc917/data/Kylemasc011AElevator.csv"
# Replace with the actual path to your CSV file
gen_pic = "Kylemasc917/data/motion_plot_elevator1.png" 
# Replace with the desired output image file

# Process the data and plot the motion

x_position, y_position, z_position = process_acceleration_data(csv_file, gen_pic)

# Print the calculated positions (for example purposes)

print("X Position:", x_position)
print("Y Position:", y_position)
print("Z Position:", z_position)

"""Test run of direction of motion of elevator module"""

# Specify the input CSV file and the output image file
csv_file = "Kylemasc917/data/Kylemasc011CElevator2.csv"
# Replace with the actual path to your CSV file
gen_pic = "Kylemasc917/data/motion_plot_elevator2.png"
# Replace with the desired output image file

# Process the data and plot the motion

x_position, y_position, z_position = process_acceleration_data(csv_file, gen_pic)

# Print the calculated positions (for example purposes)

print("X Position:", x_position)
print("Y Position:", y_position)
print("Z Position:", z_position)

"""Test run of Distance between two GPS points module"""

# Specify the input CSV file and the output image file
csv_file = "Kylemasc917/data/Kylemasc011DWalkingCircle.csv"
# Replace with the actual path to your CSV file
output_image = "Kylemasc917/data/circle_graph.png"
# Replace with the desired output image file

# Process the data and plot the motion

distances = process_gps_data(csv_file, output_image)

# Print the calculated distances between adjacent GPS points (for example purposes)

print("Distances between adjacent GPS points:", distances)

# Specify the input CSV file and the output image file
csv_file = "Kylemasc917/data/Kylemasc011BWalkingTriangle.csv"
# Replace with the actual path to your CSV file
output_image = "Kylemasc917/data/triangle_graph.png"
# Replace with the desired output image file

# Process the data and plot the motion
distances = process_gps_data(csv_file, output_image)

# Print the calculated distances between adjacent GPS points (for example purposes)
print("Distances between adjacent GPS points:", distances)
