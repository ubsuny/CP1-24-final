"""
This module is to calculate the overall direction of motion
for a given set of accelerations
"""

from direction_of_motion_preliminaries import integrated_data, unit_vector, average

# The four runs for the downward motion experiment
file_ad_1_relative_path = "/workspaces/CP1-24-midterm/HeshamElsaman/Data/AccelerationDown/ad001_acceleration_down/Acceleration without g 2024-10-20 20-09-36/raw_data.csv"
file_ad_2_relative_path = "/workspaces/CP1-24-midterm/HeshamElsaman/Data/AccelerationDown/ad002_acceleration_down/Acceleration without g 2024-10-20 20-12-52/raw_data.csv"
file_ad_3_relative_path = "/workspaces/CP1-24-midterm/HeshamElsaman/Data/AccelerationDown/ad003_acceleration_down/Acceleration without g 2024-10-20 20-14-10/raw_data.csv"
file_ad_4_relative_path = "/workspaces/CP1-24-midterm/HeshamElsaman/Data/AccelerationDown/ad004_acceleration_down/Acceleration without g 2024-10-20 20-18-58/raw_data.csv"

# Reading the data of the first run
with open(file_ad_1_relative_path, 'r') as file_ad_1:
    content_ad_1 = file_ad_1.readlines()
    times_ad_1 = [float(i.split(',')[0]) for i in content_ad_1[1:]]
    acceleration_x_ad_1 = [float(i.split(',')[1]) for i in content_ad_1[1:]]
    acceleration_y_ad_1 = [float(i.split(',')[2]) for i in content_ad_1[1:]]
    acceleration_z_ad_1 = [float(i.split(',')[3]) for i in content_ad_1[1:]]

# The velocity components arrays for the first run
velocity_x_ad_1 = list(integrated_data(times_ad_1, acceleration_x_ad_1))
velocity_y_ad_1 = list(integrated_data(times_ad_1, acceleration_y_ad_1))
velocity_z_ad_1 = list(integrated_data(times_ad_1, acceleration_z_ad_1))

# The velocity vectors list for the first run
velocity_ad_1 = [
    [velocity_x_ad_1[i], velocity_y_ad_1[i], velocity_z_ad_1[i]]
    for i in range(len(times_ad_1))
    ]

# The unit vectors list for each velocity for the first run
velocity_unit_vectors_ad_1 = [list(unit_vector(i)) for i in velocity_ad_1]

# The overall direction of motion for the first run
overall_x_direction_ad_1 = average(velocity_unit_vectors_ad_1[:][0])
overall_y_direction_ad_1 = average(velocity_unit_vectors_ad_1[:][1])
overall_z_direction_ad_1 = average(velocity_unit_vectors_ad_1[:][2])
overall_direction_ad_1 = list([overall_x_direction_ad_1,
                          overall_y_direction_ad_1,
                          overall_z_direction_ad_1])

# Reading the data of the second run
with open(file_ad_2_relative_path, 'r') as file_ad_2:
    content_ad_2 = file_ad_2.readlines()
    times_ad_2 = [float(i.split(',')[0]) for i in content_ad_2[1:]]
    acceleration_x_ad_2 = [float(i.split(',')[1]) for i in content_ad_2[1:]]
    acceleration_y_ad_2 = [float(i.split(',')[2]) for i in content_ad_2[1:]]
    acceleration_z_ad_2 = [float(i.split(',')[3]) for i in content_ad_2[1:]]

# The velocity components arrays for the second run
velocity_x_ad_2 = list(integrated_data(times_ad_2, acceleration_x_ad_2))
velocity_y_ad_2 = list(integrated_data(times_ad_2, acceleration_y_ad_2))
velocity_z_ad_2 = list(integrated_data(times_ad_2, acceleration_z_ad_2))

# The velocity vectors list for the second run
velocity_ad_2 = [
    [velocity_x_ad_2[i], velocity_y_ad_2[i], velocity_z_ad_2[i]]
    for i in range(len(times_ad_2))
    ]

# The unit vectors list for each velocity for the second run
velocity_unit_vectors_ad_2 = [list(unit_vector(i)) for i in velocity_ad_2]

# The overall direction of motion for the second run
overall_x_direction_ad_2 = average(velocity_unit_vectors_ad_2[:][0])
overall_y_direction_ad_2 = average(velocity_unit_vectors_ad_2[:][1])
overall_z_direction_ad_2 = average(velocity_unit_vectors_ad_2[:][2])
overall_direction_ad_2 = list([overall_x_direction_ad_2,
                          overall_y_direction_ad_2,
                          overall_z_direction_ad_2])

# Reading the data of the third run
with open(file_ad_3_relative_path, 'r') as file_ad_3:
    content_ad_3 = file_ad_3.readlines()
    times_ad_3 = [float(i.split(',')[0]) for i in content_ad_3[1:]]
    acceleration_x_ad_3 = [float(i.split(',')[1]) for i in content_ad_3[1:]]
    acceleration_y_ad_3 = [float(i.split(',')[2]) for i in content_ad_3[1:]]
    acceleration_z_ad_3 = [float(i.split(',')[3]) for i in content_ad_3[1:]]

# The velocity components arrays for the third run
velocity_x_ad_3 = list(integrated_data(times_ad_3, acceleration_x_ad_3))
velocity_y_ad_3 = list(integrated_data(times_ad_3, acceleration_y_ad_3))
velocity_z_ad_3 = list(integrated_data(times_ad_3, acceleration_z_ad_3))

# The velocity vectors list for the third run
velocity_ad_3 = [
    [velocity_x_ad_3[i], velocity_y_ad_3[i], velocity_z_ad_3[i]]
    for i in range(len(times_ad_3))
    ]

# The unit vectors list for each velocity for the third run
velocity_unit_vectors_ad_3 = [list(unit_vector(i)) for i in velocity_ad_3]

# The overall direction of motion for the third run
overall_x_direction_ad_3 = average(velocity_unit_vectors_ad_3[:][0])
overall_y_direction_ad_3 = average(velocity_unit_vectors_ad_3[:][1])
overall_z_direction_ad_3 = average(velocity_unit_vectors_ad_3[:][2])
overall_direction_ad_3 = list([overall_x_direction_ad_3,
                          overall_y_direction_ad_3,
                          overall_z_direction_ad_3])

# Reading the data of the fourth run
with open(file_ad_4_relative_path, 'r') as file_ad_4:
    content_ad_4 = file_ad_4.readlines()
    times_ad_4 = [float(i.split(',')[0]) for i in content_ad_4[1:]]
    acceleration_x_ad_4 = [float(i.split(',')[1]) for i in content_ad_4[1:]]
    acceleration_y_ad_4 = [float(i.split(',')[2]) for i in content_ad_4[1:]]
    acceleration_z_ad_4 = [float(i.split(',')[3]) for i in content_ad_4[1:]]

# The velocity components arrays for the fourth run
velocity_x_ad_4 = list(integrated_data(times_ad_4, acceleration_x_ad_4))
velocity_y_ad_4 = list(integrated_data(times_ad_4, acceleration_y_ad_4))
velocity_z_ad_4 = list(integrated_data(times_ad_4, acceleration_z_ad_4))

# The velocity vectors list for the fourth run
velocity_ad_4 = [
    [velocity_x_ad_4[i], velocity_y_ad_4[i], velocity_z_ad_4[i]]
    for i in range(len(times_ad_4))
    ]

# The unit vectors list for each velocity for the fourth run
velocity_unit_vectors_ad_4 = [list(unit_vector(i)) for i in velocity_ad_4]

# The overall direction of motion for the fourth run
overall_x_direction_ad_4 = average(velocity_unit_vectors_ad_4[:][0])
overall_y_direction_ad_4 = average(velocity_unit_vectors_ad_4[:][1])
overall_z_direction_ad_4 = average(velocity_unit_vectors_ad_4[:][2])
overall_direction_ad_4 = list([overall_x_direction_ad_4,
                          overall_y_direction_ad_4,
                          overall_z_direction_ad_4])
###################################################################

# The four runs for the upward motion experiment
file_au_1_relative_path = "/workspaces/CP1-24-midterm/HeshamElsaman/Data/AccelerationUp/au001_acceleration_up/Acceleration without g 2024-10-20 20-07-51/raw_data.csv"
file_au_2_relative_path = "/workspaces/CP1-24-midterm/HeshamElsaman/Data/AccelerationUp/au002_acceleration_up/Acceleration without g 2024-10-20 20-10-58/raw_data.csv"
file_au_3_relative_path = "/workspaces/CP1-24-midterm/HeshamElsaman/Data/AccelerationUp/au002_acceleration_up/Acceleration without g 2024-10-20 20-10-58/raw_data.csv"
file_au_4_relative_path = "/workspaces/CP1-24-midterm/HeshamElsaman/Data/AccelerationUp/au004_acceleration_up/Acceleration without g 2024-10-20 20-17-14/raw_data.csv"

# Reading the data of the first run
with open(file_au_1_relative_path, 'r') as file_au_1:
    content_au_1 = file_au_1.readlines()
    times_au_1 = [float(i.split(',')[0]) for i in content_au_1[1:]]
    acceleration_x_au_1 = [float(i.split(',')[1]) for i in content_au_1[1:]]
    acceleration_y_au_1 = [float(i.split(',')[2]) for i in content_au_1[1:]]
    acceleration_z_au_1 = [float(i.split(',')[3]) for i in content_au_1[1:]]

# The velocity components arrays for the first run
velocity_x_au_1 = list(integrated_data(times_au_1, acceleration_x_au_1))
velocity_y_au_1 = list(integrated_data(times_au_1, acceleration_y_au_1))
velocity_z_au_1 = list(integrated_data(times_au_1, acceleration_z_au_1))

# The velocity vectors list for the first run
velocity_au_1 = [
    [velocity_x_au_1[i], velocity_y_au_1[i], velocity_z_au_1[i]]
    for i in range(len(times_au_1))
    ]

# The unit vectors list for each velocity for the first run
velocity_unit_vectors_au_1 = [list(unit_vector(i)) for i in velocity_au_1]

# The overall direction of motion for the first run
overall_x_direction_au_1 = average(velocity_unit_vectors_au_1[:][0])
overall_y_direction_au_1 = average(velocity_unit_vectors_au_1[:][1])
overall_z_direction_au_1 = average(velocity_unit_vectors_au_1[:][2])
overall_direction_au_1 = list([overall_x_direction_au_1,
                          overall_y_direction_au_1,
                          overall_z_direction_au_1])

# Reading the data of the second run
with open(file_au_2_relative_path, 'r') as file_au_2:
    content_au_2 = file_au_2.readlines()
    times_au_2 = [float(i.split(',')[0]) for i in content_au_2[1:]]
    acceleration_x_au_2 = [float(i.split(',')[1]) for i in content_au_2[1:]]
    acceleration_y_au_2 = [float(i.split(',')[2]) for i in content_au_2[1:]]
    acceleration_z_au_2 = [float(i.split(',')[3]) for i in content_au_2[1:]]

# The velocity components arrays for the second run
velocity_x_au_2 = list(integrated_data(times_au_2, acceleration_x_au_2))
velocity_y_au_2 = list(integrated_data(times_au_2, acceleration_y_au_2))
velocity_z_au_2 = list(integrated_data(times_au_2, acceleration_z_au_2))

# The velocity vectors list for the second run
velocity_au_2 = [
    [velocity_x_au_2[i], velocity_y_au_2[i], velocity_z_au_2[i]]
    for i in range(len(times_au_2))
    ]

# The unit vectors list for each velocity for the second run
velocity_unit_vectors_au_2 = [list(unit_vector(i)) for i in velocity_au_2]

# The overall direction of motion for the second run
overall_x_direction_au_2 = average(velocity_unit_vectors_au_2[:][0])
overall_y_direction_au_2 = average(velocity_unit_vectors_au_2[:][1])
overall_z_direction_au_2 = average(velocity_unit_vectors_au_2[:][2])
overall_direction_au_2 = list([overall_x_direction_au_2,
                          overall_y_direction_au_2,
                          overall_z_direction_au_2])

# Reading the data of the third run
with open(file_au_3_relative_path, 'r') as file_au_3:
    content_au_3 = file_au_3.readlines()
    times_au_3 = [float(i.split(',')[0]) for i in content_au_3[1:]]
    acceleration_x_au_3 = [float(i.split(',')[1]) for i in content_au_3[1:]]
    acceleration_y_au_3 = [float(i.split(',')[2]) for i in content_au_3[1:]]
    acceleration_z_au_3 = [float(i.split(',')[3]) for i in content_au_3[1:]]

# The velocity components arrays for the third run
velocity_x_au_3 = list(integrated_data(times_au_3, acceleration_x_au_3))
velocity_y_au_3 = list(integrated_data(times_au_3, acceleration_y_au_3))
velocity_z_au_3 = list(integrated_data(times_au_3, acceleration_z_au_3))

# The velocity vectors list for the third run
velocity_au_3 = [
    [velocity_x_au_3[i], velocity_y_au_3[i], velocity_z_au_3[i]]
    for i in range(len(times_au_3))
    ]

# The unit vectors list for each velocity for the third run
velocity_unit_vectors_au_3 = [list(unit_vector(i)) for i in velocity_au_3]

# The overall direction of motion for the third run
overall_x_direction_au_3 = average(velocity_unit_vectors_au_3[:][0])
overall_y_direction_au_3 = average(velocity_unit_vectors_au_3[:][1])
overall_z_direction_au_3 = average(velocity_unit_vectors_au_3[:][2])
overall_direction_au_3 = list([overall_x_direction_au_3,
                          overall_y_direction_au_3,
                          overall_z_direction_au_3])

# Reading the data of the fourth run
with open(file_au_4_relative_path, 'r') as file_au_4:
    content_au_4 = file_au_4.readlines()
    times_au_4 = [float(i.split(',')[0]) for i in content_au_4[1:]]
    acceleration_x_au_4 = [float(i.split(',')[1]) for i in content_au_4[1:]]
    acceleration_y_au_4 = [float(i.split(',')[2]) for i in content_au_4[1:]]
    acceleration_z_au_4 = [float(i.split(',')[3]) for i in content_au_4[1:]]

# The velocity components arrays for the fourth run
velocity_x_au_4 = list(integrated_data(times_au_4, acceleration_x_au_4))
velocity_y_au_4 = list(integrated_data(times_au_4, acceleration_y_au_4))
velocity_z_au_4 = list(integrated_data(times_au_4, acceleration_z_au_4))

# The velocity vectors list for the fourth run
velocity_au_4 = [
    [velocity_x_au_4[i], velocity_y_au_4[i], velocity_z_au_4[i]]
    for i in range(len(times_au_4))
    ]

# The unit vectors list for each velocity for the fourth run
velocity_unit_vectors_au_4 = [list(unit_vector(i)) for i in velocity_au_4]

# The overall direction of motion for the fourth run
overall_x_direction_au_4 = average(velocity_unit_vectors_au_4[:][0])
overall_y_direction_au_4 = average(velocity_unit_vectors_au_4[:][1])
overall_z_direction_au_4 = average(velocity_unit_vectors_au_4[:][2])
overall_direction_au_4 = list([overall_x_direction_au_4,
                          overall_y_direction_au_4,
                          overall_z_direction_au_4])
