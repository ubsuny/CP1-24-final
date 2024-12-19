"""
This module is to calculate the distance between two adjacent
GPS locations in a given set of data
"""

from unit_converter_module import deg_to_rad
from distance_functions import earth_dist

# The paths for the files to read
# The triangle runs
file_gt_1_relative_path = "/workspaces/CP1-24-midterm/HeshamElsaman/Data/GPSTriangle/gt001_gps_triangle/Location GPS 2024-10-20 19-25-53/raw_data.csv"
file_gt_2_relative_path = "/workspaces/CP1-24-midterm/HeshamElsaman/Data/GPSTriangle/gt002_gps_triangle/Location GPS 2024-10-20 19-29-23/raw_data.csv"
file_gt_3_relative_path = "/workspaces/CP1-24-midterm/HeshamElsaman/Data/GPSTriangle/gt003_gps_triangle/Location GPS 2024-10-20 19-33-08/raw_data.csv"
file_gt_4_relative_path = "/workspaces/CP1-24-midterm/HeshamElsaman/Data/GPSTriangle/gt004_gps_triangle/Location GPS 2024-10-20 19-36-53/raw_data.csv"
# The circle runs
file_gc_1_relative_path = "/workspaces/CP1-24-midterm/HeshamElsaman/Data/GPSCircle/gc001_gps_circle/Location GPS 2024-10-20 17-56-35/raw_data.csv"
file_gc_2_relative_path = "/workspaces/CP1-24-midterm/HeshamElsaman/Data/GPSCircle/gc002_gps_circle/Location GPS 2024-10-20 18-09-57/raw_data.csv"
file_gc_3_relative_path = "/workspaces/CP1-24-midterm/HeshamElsaman/Data/GPSCircle/gc003_gps_circle/Location GPS 2024-10-20 18-13-07/raw_data.csv"
file_gc_4_relative_path = "/workspaces/CP1-24-midterm/HeshamElsaman/Data/GPSCircle/gc004_gps_circle/Location GPS 2024-10-20 18-16-03/raw_data.csv"

# Reading the data of the first run for the triangle
with open(file_gt_1_relative_path, 'r') as file_gt_1:
    content_gt_1 = file_gt_1.readlines()
    times_gt_1 = [float(i.split(',')[0]) for i in content_gt_1[1:]]
    latitudes_gt_1 = list(deg_to_rad([float(i.split(',')[1]) for i in content_gt_1[1:]]))
    longitudes_gt_1 = list(deg_to_rad([float(i.split(',')[2]) for i in content_gt_1[1:]]))

# Calculating the distance between each adjacent pair of points for the first run for the triangle
distances_gt_1 = []
for i in range(1, len(times_gt_1)):
    distances_gt_1.append(earth_dist(longitudes_gt_1[i], longitudes_gt_1[i - 1],
                                     latitudes_gt_1[i], latitudes_gt_1[i - 1]))
    
# Reading the data of the second run for the triangle
with open(file_gt_2_relative_path, 'r') as file_gt_2:
    content_gt_2 = file_gt_2.readlines()
    times_gt_2 = [float(i.split(',')[0]) for i in content_gt_2[1:]]
    latitudes_gt_2 = list(deg_to_rad([float(i.split(',')[1]) for i in content_gt_2[1:]]))
    longitudes_gt_2 = list(deg_to_rad([float(i.split(',')[2]) for i in content_gt_2[1:]]))

# Calculating the distance between each adjacent pair of points for the second run for the triangle
distances_gt_2 = []
for i in range(1, len(times_gt_2)):
    distances_gt_2.append(earth_dist(longitudes_gt_2[i], longitudes_gt_2[i - 1],
                                     latitudes_gt_2[i], latitudes_gt_2[i - 1]))

# Reading the data of the third run for the triangle
with open(file_gt_3_relative_path, 'r') as file_gt_3:
    content_gt_3 = file_gt_3.readlines()
    times_gt_3 = [float(i.split(',')[0]) for i in content_gt_3[1:]]
    latitudes_gt_3 = list(deg_to_rad([float(i.split(',')[1]) for i in content_gt_3[1:]]))
    longitudes_gt_3 = list(deg_to_rad([float(i.split(',')[2]) for i in content_gt_3[1:]]))

# Calculating the distance between each adjacent pair of points for the third run for the triangle
distances_gt_3 = []
for i in range(1, len(times_gt_3)):
    distances_gt_3.append(earth_dist(longitudes_gt_3[i], longitudes_gt_3[i - 1],
                                     latitudes_gt_3[i], latitudes_gt_3[i - 1]))

# Reading the data of the fourth run for the triangle
with open(file_gt_4_relative_path, 'r') as file_gt_4:
    content_gt_4 = file_gt_4.readlines()
    times_gt_4 = [float(i.split(',')[0]) for i in content_gt_4[1:]]
    latitudes_gt_4 = list(deg_to_rad([float(i.split(',')[1]) for i in content_gt_4[1:]]))
    longitudes_gt_4 = list(deg_to_rad([float(i.split(',')[2]) for i in content_gt_4[1:]]))

# Calculating the distance between each adjacent pair of points for the fourth run for the triangle
distances_gt_4 = []
for i in range(1, len(times_gt_4)):
    distances_gt_4.append(earth_dist(longitudes_gt_4[i], longitudes_gt_4[i - 1],
                                     latitudes_gt_4[i], latitudes_gt_4[i - 1]))

# Reading the data of the first run for the circle
with open(file_gc_1_relative_path, 'r') as file_gc_1:
    content_gc_1 = file_gc_1.readlines()
    times_gc_1 = [float(i.split(',')[0]) for i in content_gc_1[1:]]
    latitudes_gc_1 = list(deg_to_rad([float(i.split(',')[1]) for i in content_gc_1[1:]]))
    longitudes_gc_1 = list(deg_to_rad([float(i.split(',')[2]) for i in content_gc_1[1:]]))

# Calculating the distance between each adjacent pair of points for the first run for the circle
distances_gc_1 = []
for i in range(1, len(times_gc_1)):
    distances_gc_1.append(earth_dist(longitudes_gc_1[i], longitudes_gc_1[i - 1],
                                     latitudes_gc_1[i], latitudes_gc_1[i - 1]))
    
# Reading the data of the second run for the circle
with open(file_gc_2_relative_path, 'r') as file_gc_2:
    content_gc_2 = file_gc_2.readlines()
    times_gc_2 = [float(i.split(',')[0]) for i in content_gc_2[1:]]
    latitudes_gc_2 = list(deg_to_rad([float(i.split(',')[1]) for i in content_gc_2[1:]]))
    longitudes_gc_2 = list(deg_to_rad([float(i.split(',')[2]) for i in content_gc_2[1:]]))

# Calculating the distance between each adjacent pair of points for the second run for the circle
distances_gc_2 = []
for i in range(1, len(times_gc_2)):
    distances_gc_2.append(earth_dist(longitudes_gc_2[i], longitudes_gc_2[i - 1],
                                     latitudes_gc_2[i], latitudes_gc_2[i - 1]))

# Reading the data of the third run for the circle
with open(file_gc_3_relative_path, 'r') as file_gc_3:
    content_gc_3 = file_gc_3.readlines()
    times_gc_3 = [float(i.split(',')[0]) for i in content_gc_3[1:]]
    latitudes_gc_3 = list(deg_to_rad([float(i.split(',')[1]) for i in content_gc_3[1:]]))
    longitudes_gc_3 = list(deg_to_rad([float(i.split(',')[2]) for i in content_gc_3[1:]]))

# Calculating the distance between each adjacent pair of points for the third run for the circle
distances_gc_3 = []
for i in range(1, len(times_gc_3)):
    distances_gc_3.append(earth_dist(longitudes_gc_3[i], longitudes_gc_3[i - 1],
                                     latitudes_gc_3[i], latitudes_gc_3[i - 1]))

# Reading the data of the fourth run for the circle
with open(file_gc_4_relative_path, 'r') as file_gc_4:
    content_gc_4 = file_gc_4.readlines()
    times_gc_4 = [float(i.split(',')[0]) for i in content_gc_4[1:]]
    latitudes_gc_4 = list(deg_to_rad([float(i.split(',')[1]) for i in content_gc_4[1:]]))
    longitudes_gc_4 = list(deg_to_rad([float(i.split(',')[2]) for i in content_gc_4[1:]]))

# Calculating the distance between each adjacent pair of points for the fourth run for the circle
distances_gc_4 = []
for i in range(1, len(times_gc_4)):
    distances_gc_4.append(earth_dist(longitudes_gc_4[i], longitudes_gc_4[i - 1],
                                     latitudes_gc_4[i], latitudes_gc_4[i - 1]))
