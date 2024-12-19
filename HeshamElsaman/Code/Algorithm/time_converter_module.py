"""
This module is to read the start and end times of each
experiment and convert them into unic time
"""

from time_conversion_function import to_unix_time

# Importing the files to read
# The downward acceleration experiment runs
file_ad_1_relative_path = "/workspaces/CP1-24-midterm/HeshamElsaman/Data/AccelerationDown/ad001_acceleration_down/Acceleration without g 2024-10-20 20-09-36/meta/time.csv"
file_ad_2_relative_path = "/workspaces/CP1-24-midterm/HeshamElsaman/Data/AccelerationDown/ad002_acceleration_down/Acceleration without g 2024-10-20 20-12-52/meta/time.csv"
file_ad_3_relative_path = "/workspaces/CP1-24-midterm/HeshamElsaman/Data/AccelerationDown/ad003_acceleration_down/Acceleration without g 2024-10-20 20-14-10/meta/time.csv"
file_ad_4_relative_path = "/workspaces/CP1-24-midterm/HeshamElsaman/Data/AccelerationDown/ad004_acceleration_down/Acceleration without g 2024-10-20 20-18-58/meta/time.csv"
# The upward acceleration experiment runs
file_au_1_relative_path = "/workspaces/CP1-24-midterm/HeshamElsaman/Data/AccelerationUp/au001_acceleration_up/Acceleration without g 2024-10-20 20-07-51/meta/time.csv"
file_au_2_relative_path = "/workspaces/CP1-24-midterm/HeshamElsaman/Data/AccelerationUp/au002_acceleration_up/Acceleration without g 2024-10-20 20-10-58/meta/time.csv"
file_au_3_relative_path = "/workspaces/CP1-24-midterm/HeshamElsaman/Data/AccelerationUp/au002_acceleration_up/Acceleration without g 2024-10-20 20-10-58/meta/time.csv"
file_au_4_relative_path = "/workspaces/CP1-24-midterm/HeshamElsaman/Data/AccelerationUp/au004_acceleration_up/Acceleration without g 2024-10-20 20-17-14/meta/time.csv"
# The triangle runs
file_gt_1_relative_path = "/workspaces/CP1-24-midterm/HeshamElsaman/Data/GPSTriangle/gt001_gps_triangle/Location GPS 2024-10-20 19-25-53/meta/time.csv"
file_gt_2_relative_path = "/workspaces/CP1-24-midterm/HeshamElsaman/Data/GPSTriangle/gt002_gps_triangle/Location GPS 2024-10-20 19-29-23/meta/time.csv"
file_gt_3_relative_path = "/workspaces/CP1-24-midterm/HeshamElsaman/Data/GPSTriangle/gt003_gps_triangle/Location GPS 2024-10-20 19-33-08/meta/time.csv"
file_gt_4_relative_path = "/workspaces/CP1-24-midterm/HeshamElsaman/Data/GPSTriangle/gt004_gps_triangle/Location GPS 2024-10-20 19-36-53/meta/time.csv"
# The circle runs
file_gc_1_relative_path = "/workspaces/CP1-24-midterm/HeshamElsaman/Data/GPSCircle/gc001_gps_circle/Location GPS 2024-10-20 17-56-35/meta/time.csv"
file_gc_2_relative_path = "/workspaces/CP1-24-midterm/HeshamElsaman/Data/GPSCircle/gc002_gps_circle/Location GPS 2024-10-20 18-09-57/meta/time.csv"
file_gc_3_relative_path = "/workspaces/CP1-24-midterm/HeshamElsaman/Data/GPSCircle/gc003_gps_circle/Location GPS 2024-10-20 18-13-07/meta/time.csv"
file_gc_4_relative_path = "/workspaces/CP1-24-midterm/HeshamElsaman/Data/GPSCircle/gc004_gps_circle/Location GPS 2024-10-20 18-16-03/meta/time.csv"

# Reading the data of the first run for the upward acceleration
with open(file_au_1_relative_path, 'r') as file_au_1:
    content_au_1 = file_au_1.readlines()
    # Reading the start date and time
    start_time_and_date_au_1 = content_au_1[1].strip().split(',')[3].strip('"')
    start_time_and_date_au_1 = start_time_and_date_au_1.strip().split()
    start_time_and_date_au_1[0] = start_time_and_date_au_1[0].strip().split('-')
    start_time_and_date_au_1[1] = start_time_and_date_au_1[1].strip().split(':')
    start_time_and_date_au_1[0] = [float(i) for i in start_time_and_date_au_1[0]]
    start_time_and_date_au_1[1] = [float(i) for i in start_time_and_date_au_1[1]]
    start_date_au_1 = start_time_and_date_au_1[0]
    start_time_au_1 = start_time_and_date_au_1[1]
    start_year_au_1 = int(start_date_au_1[0])
    start_month_au_1 = int(start_date_au_1[1])
    start_day_au_1 = int(start_date_au_1[2])
    start_hour_au_1 = int(start_time_au_1[0])
    start_minute_au_1 = int(start_time_au_1[1])
    start_second_au_1 = start_time_au_1[2]

    # Reading the end date and time
    end_time_and_date_au_1 = str(content_au_1[2].strip().split(',')[3]).strip('"')
    end_time_and_date_au_1 = end_time_and_date_au_1.strip().split()
    end_time_and_date_au_1[0] = end_time_and_date_au_1[0].strip().split('-')
    end_time_and_date_au_1[1] = end_time_and_date_au_1[1].strip().split(':')
    end_time_and_date_au_1[0] = [float(i) for i in end_time_and_date_au_1[0]]
    end_time_and_date_au_1[1] = [float(i) for i in end_time_and_date_au_1[1]]
    end_date_au_1 = end_time_and_date_au_1[0]
    end_time_au_1 = end_time_and_date_au_1[1]
    end_year_au_1 = int(end_date_au_1[0])
    end_month_au_1 = int(end_date_au_1[1])
    end_day_au_1 = int(end_date_au_1[2])
    end_hour_au_1 = int(end_time_au_1[0])
    end_minute_au_1 = int(end_time_au_1[1])
    end_second_au_1 = end_time_au_1[2]
    
# Converting the dates and times to unix
unix_start_time_au_1 = to_unix_time(start_month_au_1,
                                    start_day_au_1,
                                    start_year_au_1,
                                    start_hour_au_1,
                                    start_minute_au_1,
                                    start_second_au_1)

unix_end_time_au_1 = to_unix_time(end_month_au_1,
                                  end_day_au_1,
                                  end_year_au_1,
                                  end_hour_au_1,
                                  end_minute_au_1,
                                  end_second_au_1)

print(unix_end_time_au_1 - unix_start_time_au_1, 54.175 - 30.382)

# Reading the data of the second run for the upward acceleration
with open(file_au_2_relative_path, 'r') as file_au_2:
    content_au_2 = file_au_2.readlines()
    # Reading the start date and time
    start_time_and_date_au_2 = content_au_2[1].strip().split(',')[3].strip('"')
    start_time_and_date_au_2 = start_time_and_date_au_2.strip().split()
    start_time_and_date_au_2[0] = start_time_and_date_au_2[0].strip().split('-')
    start_time_and_date_au_2[1] = start_time_and_date_au_2[1].strip().split(':')
    start_time_and_date_au_2[0] = [float(i) for i in start_time_and_date_au_2[0]]
    start_time_and_date_au_2[1] = [float(i) for i in start_time_and_date_au_2[1]]
    start_date_au_2 = start_time_and_date_au_2[0]
    start_time_au_2 = start_time_and_date_au_2[1]
    start_year_au_2 = int(start_date_au_2[0])
    start_month_au_2 = int(start_date_au_2[1])
    start_day_au_2 = int(start_date_au_2[2])
    start_hour_au_2 = int(start_time_au_2[0])
    start_minute_au_2 = int(start_time_au_2[1])
    start_second_au_2 = start_time_au_2[2]

    # Reading the end date and time
    end_time_and_date_au_2 = str(content_au_2[2].strip().split(',')[3]).strip('"')
    end_time_and_date_au_2 = end_time_and_date_au_2.strip().split()
    end_time_and_date_au_2[0] = end_time_and_date_au_2[0].strip().split('-')
    end_time_and_date_au_2[1] = end_time_and_date_au_2[1].strip().split(':')
    end_time_and_date_au_2[0] = [float(i) for i in end_time_and_date_au_2[0]]
    end_time_and_date_au_2[1] = [float(i) for i in end_time_and_date_au_2[1]]
    end_date_au_2 = end_time_and_date_au_2[0]
    end_time_au_2 = end_time_and_date_au_2[1]
    end_year_au_2 = int(end_date_au_2[0])
    end_month_au_2 = int(end_date_au_2[1])
    end_day_au_2 = int(end_date_au_2[2])
    end_hour_au_2 = int(end_time_au_2[0])
    end_minute_au_2 = int(end_time_au_2[1])
    end_second_au_2 = end_time_au_2[2]
    
# Converting the dates and times to unix
unix_start_time_au_2 = to_unix_time(start_month_au_2,
                                    start_day_au_2,
                                    start_year_au_2,
                                    start_hour_au_2,
                                    start_minute_au_2,
                                    start_second_au_2)

unix_end_time_au_2 = to_unix_time(end_month_au_2,
                                  end_day_au_2,
                                  end_year_au_2,
                                  end_hour_au_2,
                                  end_minute_au_2,
                                  end_second_au_2)

# Reading the data of the third run for the upward acceleration
with open(file_au_3_relative_path, 'r') as file_au_3:
    content_au_3 = file_au_3.readlines()
    # Reading the start date and time
    start_time_and_date_au_3 = content_au_3[1].strip().split(',')[3].strip('"')
    start_time_and_date_au_3 = start_time_and_date_au_3.strip().split()
    start_time_and_date_au_3[0] = start_time_and_date_au_3[0].strip().split('-')
    start_time_and_date_au_3[1] = start_time_and_date_au_3[1].strip().split(':')
    start_time_and_date_au_3[0] = [float(i) for i in start_time_and_date_au_3[0]]
    start_time_and_date_au_3[1] = [float(i) for i in start_time_and_date_au_3[1]]
    start_date_au_3 = start_time_and_date_au_3[0]
    start_time_au_3 = start_time_and_date_au_3[1]
    start_year_au_3 = int(start_date_au_3[0])
    start_month_au_3 = int(start_date_au_3[1])
    start_day_au_3 = int(start_date_au_3[2])
    start_hour_au_3 = int(start_time_au_3[0])
    start_minute_au_3 = int(start_time_au_3[1])
    start_second_au_3 = start_time_au_3[2]

    # Reading the end date and time
    end_time_and_date_au_3 = str(content_au_3[2].strip().split(',')[3]).strip('"')
    end_time_and_date_au_3 = end_time_and_date_au_3.strip().split()
    end_time_and_date_au_3[0] = end_time_and_date_au_3[0].strip().split('-')
    end_time_and_date_au_3[1] = end_time_and_date_au_3[1].strip().split(':')
    end_time_and_date_au_3[0] = [float(i) for i in end_time_and_date_au_3[0]]
    end_time_and_date_au_3[1] = [float(i) for i in end_time_and_date_au_3[1]]
    end_date_au_3 = end_time_and_date_au_3[0]
    end_time_au_3 = end_time_and_date_au_3[1]
    end_year_au_3 = int(end_date_au_3[0])
    end_month_au_3 = int(end_date_au_3[1])
    end_day_au_3 = int(end_date_au_3[2])
    end_hour_au_3 = int(end_time_au_3[0])
    end_minute_au_3 = int(end_time_au_3[1])
    end_second_au_3 = end_time_au_3[2]
    
# Converting the dates and times to unix
unix_start_time_au_3 = to_unix_time(start_month_au_3,
                                    start_day_au_3,
                                    start_year_au_3,
                                    start_hour_au_3,
                                    start_minute_au_3,
                                    start_second_au_3)

unix_end_time_au_3 = to_unix_time(end_month_au_3,
                                  end_day_au_3,
                                  end_year_au_3,
                                  end_hour_au_3,
                                  end_minute_au_3,
                                  end_second_au_3)

# Reading the data of the fourth run for the upward acceleration
with open(file_au_4_relative_path, 'r') as file_au_4:
    content_au_4 = file_au_4.readlines()
    # Reading the start date and time
    start_time_and_date_au_4 = content_au_4[1].strip().split(',')[3].strip('"')
    start_time_and_date_au_4 = start_time_and_date_au_4.strip().split()
    start_time_and_date_au_4[0] = start_time_and_date_au_4[0].strip().split('-')
    start_time_and_date_au_4[1] = start_time_and_date_au_4[1].strip().split(':')
    start_time_and_date_au_4[0] = [float(i) for i in start_time_and_date_au_4[0]]
    start_time_and_date_au_4[1] = [float(i) for i in start_time_and_date_au_4[1]]
    start_date_au_4 = start_time_and_date_au_4[0]
    start_time_au_4 = start_time_and_date_au_4[1]
    start_year_au_4 = int(start_date_au_4[0])
    start_month_au_4 = int(start_date_au_4[1])
    start_day_au_4 = int(start_date_au_4[2])
    start_hour_au_4 = int(start_time_au_4[0])
    start_minute_au_4 = int(start_time_au_4[1])
    start_second_au_4 = start_time_au_4[2]

    # Reading the end date and time
    end_time_and_date_au_4 = str(content_au_4[2].strip().split(',')[3]).strip('"')
    end_time_and_date_au_4 = end_time_and_date_au_4.strip().split()
    end_time_and_date_au_4[0] = end_time_and_date_au_4[0].strip().split('-')
    end_time_and_date_au_4[1] = end_time_and_date_au_4[1].strip().split(':')
    end_time_and_date_au_4[0] = [float(i) for i in end_time_and_date_au_4[0]]
    end_time_and_date_au_4[1] = [float(i) for i in end_time_and_date_au_4[1]]
    end_date_au_4 = end_time_and_date_au_4[0]
    end_time_au_4 = end_time_and_date_au_4[1]
    end_year_au_4 = int(end_date_au_4[0])
    end_month_au_4 = int(end_date_au_4[1])
    end_day_au_4 = int(end_date_au_4[2])
    end_hour_au_4 = int(end_time_au_4[0])
    end_minute_au_4 = int(end_time_au_4[1])
    end_second_au_4 = end_time_au_4[2]
    
# Converting the dates and times to unix
unix_start_time_au_4 = to_unix_time(start_month_au_4,
                                    start_day_au_4,
                                    start_year_au_4,
                                    start_hour_au_4,
                                    start_minute_au_4,
                                    start_second_au_4)

unix_end_time_au_4 = to_unix_time(end_month_au_4,
                                  end_day_au_4,
                                  end_year_au_4,
                                  end_hour_au_4,
                                  end_minute_au_4,
                                  end_second_au_4)

# Reading the data of the first run for the downward acceleration
with open(file_ad_1_relative_path, 'r') as file_ad_1:
    content_ad_1 = file_ad_1.readlines()
    # Reading the start date and time
    start_time_and_date_ad_1 = content_ad_1[1].strip().split(',')[3].strip('"')
    start_time_and_date_ad_1 = start_time_and_date_ad_1.strip().split()
    start_time_and_date_ad_1[0] = start_time_and_date_ad_1[0].strip().split('-')
    start_time_and_date_ad_1[1] = start_time_and_date_ad_1[1].strip().split(':')
    start_time_and_date_ad_1[0] = [float(i) for i in start_time_and_date_ad_1[0]]
    start_time_and_date_ad_1[1] = [float(i) for i in start_time_and_date_ad_1[1]]
    start_date_ad_1 = start_time_and_date_ad_1[0]
    start_time_ad_1 = start_time_and_date_ad_1[1]
    start_year_ad_1 = int(start_date_ad_1[0])
    start_month_ad_1 = int(start_date_ad_1[1])
    start_day_ad_1 = int(start_date_ad_1[2])
    start_hour_ad_1 = int(start_time_ad_1[0])
    start_minute_ad_1 = int(start_time_ad_1[1])
    start_second_ad_1 = start_time_ad_1[2]

    # Reading the end date and time
    end_time_and_date_ad_1 = str(content_ad_1[2].strip().split(',')[3]).strip('"')
    end_time_and_date_ad_1 = end_time_and_date_ad_1.strip().split()
    end_time_and_date_ad_1[0] = end_time_and_date_ad_1[0].strip().split('-')
    end_time_and_date_ad_1[1] = end_time_and_date_ad_1[1].strip().split(':')
    end_time_and_date_ad_1[0] = [float(i) for i in end_time_and_date_ad_1[0]]
    end_time_and_date_ad_1[1] = [float(i) for i in end_time_and_date_ad_1[1]]
    end_date_ad_1 = end_time_and_date_ad_1[0]
    end_time_ad_1 = end_time_and_date_ad_1[1]
    end_year_ad_1 = int(end_date_ad_1[0])
    end_month_ad_1 = int(end_date_ad_1[1])
    end_day_ad_1 = int(end_date_ad_1[2])
    end_hour_ad_1 = int(end_time_ad_1[0])
    end_minute_ad_1 = int(end_time_ad_1[1])
    end_second_ad_1 = end_time_ad_1[2]
    
# Converting the dates and times to unix
unix_start_time_ad_1 = to_unix_time(start_month_ad_1,
                                    start_day_ad_1,
                                    start_year_ad_1,
                                    start_hour_ad_1,
                                    start_minute_ad_1,
                                    start_second_ad_1)

unix_end_time_ad_1 = to_unix_time(end_month_ad_1,
                                  end_day_ad_1,
                                  end_year_ad_1,
                                  end_hour_ad_1,
                                  end_minute_ad_1,
                                  end_second_ad_1)

# Reading the data of the second run for the downward acceleration
with open(file_ad_2_relative_path, 'r') as file_ad_2:
    content_ad_2 = file_ad_2.readlines()
    # Reading the start date and time
    start_time_and_date_ad_2 = content_ad_2[1].strip().split(',')[3].strip('"')
    start_time_and_date_ad_2 = start_time_and_date_ad_2.strip().split()
    start_time_and_date_ad_2[0] = start_time_and_date_ad_2[0].strip().split('-')
    start_time_and_date_ad_2[1] = start_time_and_date_ad_2[1].strip().split(':')
    start_time_and_date_ad_2[0] = [float(i) for i in start_time_and_date_ad_2[0]]
    start_time_and_date_ad_2[1] = [float(i) for i in start_time_and_date_ad_2[1]]
    start_date_ad_2 = start_time_and_date_ad_2[0]
    start_time_ad_2 = start_time_and_date_ad_2[1]
    start_year_ad_2 = int(start_date_ad_2[0])
    start_month_ad_2 = int(start_date_ad_2[1])
    start_day_ad_2 = int(start_date_ad_2[2])
    start_hour_ad_2 = int(start_time_ad_2[0])
    start_minute_ad_2 = int(start_time_ad_2[1])
    start_second_ad_2 = start_time_ad_2[2]

    # Reading the end date and time
    end_time_and_date_ad_2 = str(content_ad_2[2].strip().split(',')[3]).strip('"')
    end_time_and_date_ad_2 = end_time_and_date_ad_2.strip().split()
    end_time_and_date_ad_2[0] = end_time_and_date_ad_2[0].strip().split('-')
    end_time_and_date_ad_2[1] = end_time_and_date_ad_2[1].strip().split(':')
    end_time_and_date_ad_2[0] = [float(i) for i in end_time_and_date_ad_2[0]]
    end_time_and_date_ad_2[1] = [float(i) for i in end_time_and_date_ad_2[1]]
    end_date_ad_2 = end_time_and_date_ad_2[0]
    end_time_ad_2 = end_time_and_date_ad_2[1]
    end_year_ad_2 = int(end_date_ad_2[0])
    end_month_ad_2 = int(end_date_ad_2[1])
    end_day_ad_2 = int(end_date_ad_2[2])
    end_hour_ad_2 = int(end_time_ad_2[0])
    end_minute_ad_2 = int(end_time_ad_2[1])
    end_second_ad_2 = end_time_ad_2[2]
    
# Converting the dates and times to unix
unix_start_time_ad_2 = to_unix_time(start_month_ad_2,
                                    start_day_ad_2,
                                    start_year_ad_2,
                                    start_hour_ad_2,
                                    start_minute_ad_2,
                                    start_second_ad_2)

unix_end_time_ad_2 = to_unix_time(end_month_ad_2,
                                  end_day_ad_2,
                                  end_year_ad_2,
                                  end_hour_ad_2,
                                  end_minute_ad_2,
                                  end_second_ad_2)

# Reading the data of the third run for the downward acceleration
with open(file_ad_3_relative_path, 'r') as file_ad_3:
    content_ad_3 = file_ad_3.readlines()
    # Reading the start date and time
    start_time_and_date_ad_3 = content_ad_3[1].strip().split(',')[3].strip('"')
    start_time_and_date_ad_3 = start_time_and_date_ad_3.strip().split()
    start_time_and_date_ad_3[0] = start_time_and_date_ad_3[0].strip().split('-')
    start_time_and_date_ad_3[1] = start_time_and_date_ad_3[1].strip().split(':')
    start_time_and_date_ad_3[0] = [float(i) for i in start_time_and_date_ad_3[0]]
    start_time_and_date_ad_3[1] = [float(i) for i in start_time_and_date_ad_3[1]]
    start_date_ad_3 = start_time_and_date_ad_3[0]
    start_time_ad_3 = start_time_and_date_ad_3[1]
    start_year_ad_3 = int(start_date_ad_3[0])
    start_month_ad_3 = int(start_date_ad_3[1])
    start_day_ad_3 = int(start_date_ad_3[2])
    start_hour_ad_3 = int(start_time_ad_3[0])
    start_minute_ad_3 = int(start_time_ad_3[1])
    start_second_ad_3 = start_time_ad_3[2]

    # Reading the end date and time
    end_time_and_date_ad_3 = str(content_ad_3[2].strip().split(',')[3]).strip('"')
    end_time_and_date_ad_3 = end_time_and_date_ad_3.strip().split()
    end_time_and_date_ad_3[0] = end_time_and_date_ad_3[0].strip().split('-')
    end_time_and_date_ad_3[1] = end_time_and_date_ad_3[1].strip().split(':')
    end_time_and_date_ad_3[0] = [float(i) for i in end_time_and_date_ad_3[0]]
    end_time_and_date_ad_3[1] = [float(i) for i in end_time_and_date_ad_3[1]]
    end_date_ad_3 = end_time_and_date_ad_3[0]
    end_time_ad_3 = end_time_and_date_ad_3[1]
    end_year_ad_3 = int(end_date_ad_3[0])
    end_month_ad_3 = int(end_date_ad_3[1])
    end_day_ad_3 = int(end_date_ad_3[2])
    end_hour_ad_3 = int(end_time_ad_3[0])
    end_minute_ad_3 = int(end_time_ad_3[1])
    end_second_ad_3 = end_time_ad_3[2]
    
# Converting the dates and times to unix
unix_start_time_ad_3 = to_unix_time(start_month_ad_3,
                                    start_day_ad_3,
                                    start_year_ad_3,
                                    start_hour_ad_3,
                                    start_minute_ad_3,
                                    start_second_ad_3)

unix_end_time_ad_3 = to_unix_time(end_month_ad_3,
                                  end_day_ad_3,
                                  end_year_ad_3,
                                  end_hour_ad_3,
                                  end_minute_ad_3,
                                  end_second_ad_3)

# Reading the data of the fourth run for the downward acceleration
with open(file_ad_4_relative_path, 'r') as file_ad_4:
    content_ad_4 = file_ad_4.readlines()
    # Reading the start date and time
    start_time_and_date_ad_4 = content_ad_4[1].strip().split(',')[3].strip('"')
    start_time_and_date_ad_4 = start_time_and_date_ad_4.strip().split()
    start_time_and_date_ad_4[0] = start_time_and_date_ad_4[0].strip().split('-')
    start_time_and_date_ad_4[1] = start_time_and_date_ad_4[1].strip().split(':')
    start_time_and_date_ad_4[0] = [float(i) for i in start_time_and_date_ad_4[0]]
    start_time_and_date_ad_4[1] = [float(i) for i in start_time_and_date_ad_4[1]]
    start_date_ad_4 = start_time_and_date_ad_4[0]
    start_time_ad_4 = start_time_and_date_ad_4[1]
    start_year_ad_4 = int(start_date_ad_4[0])
    start_month_ad_4 = int(start_date_ad_4[1])
    start_day_ad_4 = int(start_date_ad_4[2])
    start_hour_ad_4 = int(start_time_ad_4[0])
    start_minute_ad_4 = int(start_time_ad_4[1])
    start_second_ad_4 = start_time_ad_4[2]

    # Reading the end date and time
    end_time_and_date_ad_4 = str(content_ad_4[2].strip().split(',')[3]).strip('"')
    end_time_and_date_ad_4 = end_time_and_date_ad_4.strip().split()
    end_time_and_date_ad_4[0] = end_time_and_date_ad_4[0].strip().split('-')
    end_time_and_date_ad_4[1] = end_time_and_date_ad_4[1].strip().split(':')
    end_time_and_date_ad_4[0] = [float(i) for i in end_time_and_date_ad_4[0]]
    end_time_and_date_ad_4[1] = [float(i) for i in end_time_and_date_ad_4[1]]
    end_date_ad_4 = end_time_and_date_ad_4[0]
    end_time_ad_4 = end_time_and_date_ad_4[1]
    end_year_ad_4 = int(end_date_ad_4[0])
    end_month_ad_4 = int(end_date_ad_4[1])
    end_day_ad_4 = int(end_date_ad_4[2])
    end_hour_ad_4 = int(end_time_ad_4[0])
    end_minute_ad_4 = int(end_time_ad_4[1])
    end_second_ad_4 = end_time_ad_4[2]
    
# Converting the dates and times to unix
unix_start_time_ad_4 = to_unix_time(start_month_ad_4,
                                    start_day_ad_4,
                                    start_year_ad_4,
                                    start_hour_ad_4,
                                    start_minute_ad_4,
                                    start_second_ad_4)

unix_end_time_ad_4 = to_unix_time(end_month_ad_4,
                                  end_day_ad_4,
                                  end_year_ad_4,
                                  end_hour_ad_4,
                                  end_minute_ad_4,
                                  end_second_ad_4)

# Reading the data of the first run for the triangle
with open(file_gt_1_relative_path, 'r') as file_gt_1:
    content_gt_1 = file_gt_1.readlines()
    # Reading the start date and time
    start_time_and_date_gt_1 = content_gt_1[1].strip().split(',')[3].strip('"')
    start_time_and_date_gt_1 = start_time_and_date_gt_1.strip().split()
    start_time_and_date_gt_1[0] = start_time_and_date_gt_1[0].strip().split('-')
    start_time_and_date_gt_1[1] = start_time_and_date_gt_1[1].strip().split(':')
    start_time_and_date_gt_1[0] = [float(i) for i in start_time_and_date_gt_1[0]]
    start_time_and_date_gt_1[1] = [float(i) for i in start_time_and_date_gt_1[1]]
    start_date_gt_1 = start_time_and_date_gt_1[0]
    start_time_gt_1 = start_time_and_date_gt_1[1]
    start_year_gt_1 = int(start_date_gt_1[0])
    start_month_gt_1 = int(start_date_gt_1[1])
    start_day_gt_1 = int(start_date_gt_1[2])
    start_hour_gt_1 = int(start_time_gt_1[0])
    start_minute_gt_1 = int(start_time_gt_1[1])
    start_second_gt_1 = start_time_gt_1[2]

    # Reading the end date and time
    end_time_and_date_gt_1 = str(content_gt_1[2].strip().split(',')[3]).strip('"')
    end_time_and_date_gt_1 = end_time_and_date_gt_1.strip().split()
    end_time_and_date_gt_1[0] = end_time_and_date_gt_1[0].strip().split('-')
    end_time_and_date_gt_1[1] = end_time_and_date_gt_1[1].strip().split(':')
    end_time_and_date_gt_1[0] = [float(i) for i in end_time_and_date_gt_1[0]]
    end_time_and_date_gt_1[1] = [float(i) for i in end_time_and_date_gt_1[1]]
    end_date_gt_1 = end_time_and_date_gt_1[0]
    end_time_gt_1 = end_time_and_date_gt_1[1]
    end_year_gt_1 = int(end_date_gt_1[0])
    end_month_gt_1 = int(end_date_gt_1[1])
    end_day_gt_1 = int(end_date_gt_1[2])
    end_hour_gt_1 = int(end_time_gt_1[0])
    end_minute_gt_1 = int(end_time_gt_1[1])
    end_second_gt_1 = end_time_gt_1[2]
    
# Converting the dates and times to unix
unix_start_time_gt_1 = to_unix_time(start_month_gt_1,
                                    start_day_gt_1,
                                    start_year_gt_1,
                                    start_hour_gt_1,
                                    start_minute_gt_1,
                                    start_second_gt_1)

unix_end_time_gt_1 = to_unix_time(end_month_gt_1,
                                  end_day_gt_1,
                                  end_year_gt_1,
                                  end_hour_gt_1,
                                  end_minute_gt_1,
                                  end_second_gt_1)

# Reading the data of the second run for the triangle
with open(file_gt_2_relative_path, 'r') as file_gt_2:
    content_gt_2 = file_gt_2.readlines()
    # Reading the start date and time
    start_time_and_date_gt_2 = content_gt_2[1].strip().split(',')[3].strip('"')
    start_time_and_date_gt_2 = start_time_and_date_gt_2.strip().split()
    start_time_and_date_gt_2[0] = start_time_and_date_gt_2[0].strip().split('-')
    start_time_and_date_gt_2[1] = start_time_and_date_gt_2[1].strip().split(':')
    start_time_and_date_gt_2[0] = [float(i) for i in start_time_and_date_gt_2[0]]
    start_time_and_date_gt_2[1] = [float(i) for i in start_time_and_date_gt_2[1]]
    start_date_gt_2 = start_time_and_date_gt_2[0]
    start_time_gt_2 = start_time_and_date_gt_2[1]
    start_year_gt_2 = int(start_date_gt_2[0])
    start_month_gt_2 = int(start_date_gt_2[1])
    start_day_gt_2 = int(start_date_gt_2[2])
    start_hour_gt_2 = int(start_time_gt_2[0])
    start_minute_gt_2 = int(start_time_gt_2[1])
    start_second_gt_2 = start_time_gt_2[2]

    # Reading the end date and time
    end_time_and_date_gt_2 = str(content_gt_2[2].strip().split(',')[3]).strip('"')
    end_time_and_date_gt_2 = end_time_and_date_gt_2.strip().split()
    end_time_and_date_gt_2[0] = end_time_and_date_gt_2[0].strip().split('-')
    end_time_and_date_gt_2[1] = end_time_and_date_gt_2[1].strip().split(':')
    end_time_and_date_gt_2[0] = [float(i) for i in end_time_and_date_gt_2[0]]
    end_time_and_date_gt_2[1] = [float(i) for i in end_time_and_date_gt_2[1]]
    end_date_gt_2 = end_time_and_date_gt_2[0]
    end_time_gt_2 = end_time_and_date_gt_2[1]
    end_year_gt_2 = int(end_date_gt_2[0])
    end_month_gt_2 = int(end_date_gt_2[1])
    end_day_gt_2 = int(end_date_gt_2[2])
    end_hour_gt_2 = int(end_time_gt_2[0])
    end_minute_gt_2 = int(end_time_gt_2[1])
    end_second_gt_2 = end_time_gt_2[2]
    
# Converting the dates and times to unix
unix_start_time_gt_2 = to_unix_time(start_month_gt_2,
                                    start_day_gt_2,
                                    start_year_gt_2,
                                    start_hour_gt_2,
                                    start_minute_gt_2,
                                    start_second_gt_2)

unix_end_time_gt_2 = to_unix_time(end_month_gt_2,
                                  end_day_gt_2,
                                  end_year_gt_2,
                                  end_hour_gt_2,
                                  end_minute_gt_2,
                                  end_second_gt_2)

# Reading the data of the third run for the triangle
with open(file_gt_3_relative_path, 'r') as file_gt_3:
    content_gt_3 = file_gt_3.readlines()
    # Reading the start date and time
    start_time_and_date_gt_3 = content_gt_3[1].strip().split(',')[3].strip('"')
    start_time_and_date_gt_3 = start_time_and_date_gt_3.strip().split()
    start_time_and_date_gt_3[0] = start_time_and_date_gt_3[0].strip().split('-')
    start_time_and_date_gt_3[1] = start_time_and_date_gt_3[1].strip().split(':')
    start_time_and_date_gt_3[0] = [float(i) for i in start_time_and_date_gt_3[0]]
    start_time_and_date_gt_3[1] = [float(i) for i in start_time_and_date_gt_3[1]]
    start_date_gt_3 = start_time_and_date_gt_3[0]
    start_time_gt_3 = start_time_and_date_gt_3[1]
    start_year_gt_3 = int(start_date_gt_3[0])
    start_month_gt_3 = int(start_date_gt_3[1])
    start_day_gt_3 = int(start_date_gt_3[2])
    start_hour_gt_3 = int(start_time_gt_3[0])
    start_minute_gt_3 = int(start_time_gt_3[1])
    start_second_gt_3 = start_time_gt_3[2]

    # Reading the end date and time
    end_time_and_date_gt_3 = str(content_gt_3[2].strip().split(',')[3]).strip('"')
    end_time_and_date_gt_3 = end_time_and_date_gt_3.strip().split()
    end_time_and_date_gt_3[0] = end_time_and_date_gt_3[0].strip().split('-')
    end_time_and_date_gt_3[1] = end_time_and_date_gt_3[1].strip().split(':')
    end_time_and_date_gt_3[0] = [float(i) for i in end_time_and_date_gt_3[0]]
    end_time_and_date_gt_3[1] = [float(i) for i in end_time_and_date_gt_3[1]]
    end_date_gt_3 = end_time_and_date_gt_3[0]
    end_time_gt_3 = end_time_and_date_gt_3[1]
    end_year_gt_3 = int(end_date_gt_3[0])
    end_month_gt_3 = int(end_date_gt_3[1])
    end_day_gt_3 = int(end_date_gt_3[2])
    end_hour_gt_3 = int(end_time_gt_3[0])
    end_minute_gt_3 = int(end_time_gt_3[1])
    end_second_gt_3 = end_time_gt_3[2]
    
# Converting the dates and times to unix
unix_start_time_gt_3 = to_unix_time(start_month_gt_3,
                                    start_day_gt_3,
                                    start_year_gt_3,
                                    start_hour_gt_3,
                                    start_minute_gt_3,
                                    start_second_gt_3)

unix_end_time_gt_3 = to_unix_time(end_month_gt_3,
                                  end_day_gt_3,
                                  end_year_gt_3,
                                  end_hour_gt_3,
                                  end_minute_gt_3,
                                  end_second_gt_3)

# Reading the data of the fourth run for the triangle
with open(file_gt_4_relative_path, 'r') as file_gt_4:
    content_gt_4 = file_gt_4.readlines()
    # Reading the start date and time
    start_time_and_date_gt_4 = content_gt_4[1].strip().split(',')[3].strip('"')
    start_time_and_date_gt_4 = start_time_and_date_gt_4.strip().split()
    start_time_and_date_gt_4[0] = start_time_and_date_gt_4[0].strip().split('-')
    start_time_and_date_gt_4[1] = start_time_and_date_gt_4[1].strip().split(':')
    start_time_and_date_gt_4[0] = [float(i) for i in start_time_and_date_gt_4[0]]
    start_time_and_date_gt_4[1] = [float(i) for i in start_time_and_date_gt_4[1]]
    start_date_gt_4 = start_time_and_date_gt_4[0]
    start_time_gt_4 = start_time_and_date_gt_4[1]
    start_year_gt_4 = int(start_date_gt_4[0])
    start_month_gt_4 = int(start_date_gt_4[1])
    start_day_gt_4 = int(start_date_gt_4[2])
    start_hour_gt_4 = int(start_time_gt_4[0])
    start_minute_gt_4 = int(start_time_gt_4[1])
    start_second_gt_4 = start_time_gt_4[2]

    # Reading the end date and time
    end_time_and_date_gt_4 = str(content_gt_4[2].strip().split(',')[3]).strip('"')
    end_time_and_date_gt_4 = end_time_and_date_gt_4.strip().split()
    end_time_and_date_gt_4[0] = end_time_and_date_gt_4[0].strip().split('-')
    end_time_and_date_gt_4[1] = end_time_and_date_gt_4[1].strip().split(':')
    end_time_and_date_gt_4[0] = [float(i) for i in end_time_and_date_gt_4[0]]
    end_time_and_date_gt_4[1] = [float(i) for i in end_time_and_date_gt_4[1]]
    end_date_gt_4 = end_time_and_date_gt_4[0]
    end_time_gt_4 = end_time_and_date_gt_4[1]
    end_year_gt_4 = int(end_date_gt_4[0])
    end_month_gt_4 = int(end_date_gt_4[1])
    end_day_gt_4 = int(end_date_gt_4[2])
    end_hour_gt_4 = int(end_time_gt_4[0])
    end_minute_gt_4 = int(end_time_gt_4[1])
    end_second_gt_4 = end_time_gt_4[2]
    
# Converting the dates and times to unix
unix_start_time_gt_4 = to_unix_time(start_month_gt_4,
                                    start_day_gt_4,
                                    start_year_gt_4,
                                    start_hour_gt_4,
                                    start_minute_gt_4,
                                    start_second_gt_4)

unix_end_time_gt_4 = to_unix_time(end_month_gt_4,
                                  end_day_gt_4,
                                  end_year_gt_4,
                                  end_hour_gt_4,
                                  end_minute_gt_4,
                                  end_second_gt_4)

# Reading the data of the first run for the circle
with open(file_gc_1_relative_path, 'r') as file_gc_1:
    content_gc_1 = file_gc_1.readlines()
    # Reading the start date and time
    start_time_and_date_gc_1 = content_gc_1[1].strip().split(',')[3].strip('"')
    start_time_and_date_gc_1 = start_time_and_date_gc_1.strip().split()
    start_time_and_date_gc_1[0] = start_time_and_date_gc_1[0].strip().split('-')
    start_time_and_date_gc_1[1] = start_time_and_date_gc_1[1].strip().split(':')
    start_time_and_date_gc_1[0] = [float(i) for i in start_time_and_date_gc_1[0]]
    start_time_and_date_gc_1[1] = [float(i) for i in start_time_and_date_gc_1[1]]
    start_date_gc_1 = start_time_and_date_gc_1[0]
    start_time_gc_1 = start_time_and_date_gc_1[1]
    start_year_gc_1 = int(start_date_gc_1[0])
    start_month_gc_1 = int(start_date_gc_1[1])
    start_day_gc_1 = int(start_date_gc_1[2])
    start_hour_gc_1 = int(start_time_gc_1[0])
    start_minute_gc_1 = int(start_time_gc_1[1])
    start_second_gc_1 = start_time_gc_1[2]

    # Reading the end date and time
    end_time_and_date_gc_1 = str(content_gc_1[2].strip().split(',')[3]).strip('"')
    end_time_and_date_gc_1 = end_time_and_date_gc_1.strip().split()
    end_time_and_date_gc_1[0] = end_time_and_date_gc_1[0].strip().split('-')
    end_time_and_date_gc_1[1] = end_time_and_date_gc_1[1].strip().split(':')
    end_time_and_date_gc_1[0] = [float(i) for i in end_time_and_date_gc_1[0]]
    end_time_and_date_gc_1[1] = [float(i) for i in end_time_and_date_gc_1[1]]
    end_date_gc_1 = end_time_and_date_gc_1[0]
    end_time_gc_1 = end_time_and_date_gc_1[1]
    end_year_gc_1 = int(end_date_gc_1[0])
    end_month_gc_1 = int(end_date_gc_1[1])
    end_day_gc_1 = int(end_date_gc_1[2])
    end_hour_gc_1 = int(end_time_gc_1[0])
    end_minute_gc_1 = int(end_time_gc_1[1])
    end_second_gc_1 = end_time_gc_1[2]
    
# Converting the dates and times to unix
unix_start_time_gc_1 = to_unix_time(start_month_gc_1,
                                    start_day_gc_1,
                                    start_year_gc_1,
                                    start_hour_gc_1,
                                    start_minute_gc_1,
                                    start_second_gc_1)

unix_end_time_gc_1 = to_unix_time(end_month_gc_1,
                                  end_day_gc_1,
                                  end_year_gc_1,
                                  end_hour_gc_1,
                                  end_minute_gc_1,
                                  end_second_gc_1)

# Reading the data of the second run for the circle
with open(file_gc_2_relative_path, 'r') as file_gc_2:
    content_gc_2 = file_gc_2.readlines()
    # Reading the start date and time
    start_time_and_date_gc_2 = content_gc_2[1].strip().split(',')[3].strip('"')
    start_time_and_date_gc_2 = start_time_and_date_gc_2.strip().split()
    start_time_and_date_gc_2[0] = start_time_and_date_gc_2[0].strip().split('-')
    start_time_and_date_gc_2[1] = start_time_and_date_gc_2[1].strip().split(':')
    start_time_and_date_gc_2[0] = [float(i) for i in start_time_and_date_gc_2[0]]
    start_time_and_date_gc_2[1] = [float(i) for i in start_time_and_date_gc_2[1]]
    start_date_gc_2 = start_time_and_date_gc_2[0]
    start_time_gc_2 = start_time_and_date_gc_2[1]
    start_year_gc_2 = int(start_date_gc_2[0])
    start_month_gc_2 = int(start_date_gc_2[1])
    start_day_gc_2 = int(start_date_gc_2[2])
    start_hour_gc_2 = int(start_time_gc_2[0])
    start_minute_gc_2 = int(start_time_gc_2[1])
    start_second_gc_2 = start_time_gc_2[2]

    # Reading the end date and time
    end_time_and_date_gc_2 = str(content_gc_2[2].strip().split(',')[3]).strip('"')
    end_time_and_date_gc_2 = end_time_and_date_gc_2.strip().split()
    end_time_and_date_gc_2[0] = end_time_and_date_gc_2[0].strip().split('-')
    end_time_and_date_gc_2[1] = end_time_and_date_gc_2[1].strip().split(':')
    end_time_and_date_gc_2[0] = [float(i) for i in end_time_and_date_gc_2[0]]
    end_time_and_date_gc_2[1] = [float(i) for i in end_time_and_date_gc_2[1]]
    end_date_gc_2 = end_time_and_date_gc_2[0]
    end_time_gc_2 = end_time_and_date_gc_2[1]
    end_year_gc_2 = int(end_date_gc_2[0])
    end_month_gc_2 = int(end_date_gc_2[1])
    end_day_gc_2 = int(end_date_gc_2[2])
    end_hour_gc_2 = int(end_time_gc_2[0])
    end_minute_gc_2 = int(end_time_gc_2[1])
    end_second_gc_2 = end_time_gc_2[2]
    
# Converting the dates and times to unix
unix_start_time_gc_2 = to_unix_time(start_month_gc_2,
                                    start_day_gc_2,
                                    start_year_gc_2,
                                    start_hour_gc_2,
                                    start_minute_gc_2,
                                    start_second_gc_2)

unix_end_time_gc_2 = to_unix_time(end_month_gc_2,
                                  end_day_gc_2,
                                  end_year_gc_2,
                                  end_hour_gc_2,
                                  end_minute_gc_2,
                                  end_second_gc_2)

# Reading the data of the third run for the circle
with open(file_gc_3_relative_path, 'r') as file_gc_3:
    content_gc_3 = file_gc_3.readlines()
    # Reading the start date and time
    start_time_and_date_gc_3 = content_gc_3[1].strip().split(',')[3].strip('"')
    start_time_and_date_gc_3 = start_time_and_date_gc_3.strip().split()
    start_time_and_date_gc_3[0] = start_time_and_date_gc_3[0].strip().split('-')
    start_time_and_date_gc_3[1] = start_time_and_date_gc_3[1].strip().split(':')
    start_time_and_date_gc_3[0] = [float(i) for i in start_time_and_date_gc_3[0]]
    start_time_and_date_gc_3[1] = [float(i) for i in start_time_and_date_gc_3[1]]
    start_date_gc_3 = start_time_and_date_gc_3[0]
    start_time_gc_3 = start_time_and_date_gc_3[1]
    start_year_gc_3 = int(start_date_gc_3[0])
    start_month_gc_3 = int(start_date_gc_3[1])
    start_day_gc_3 = int(start_date_gc_3[2])
    start_hour_gc_3 = int(start_time_gc_3[0])
    start_minute_gc_3 = int(start_time_gc_3[1])
    start_second_gc_3 = start_time_gc_3[2]

    # Reading the end date and time
    end_time_and_date_gc_3 = str(content_gc_3[2].strip().split(',')[3]).strip('"')
    end_time_and_date_gc_3 = end_time_and_date_gc_3.strip().split()
    end_time_and_date_gc_3[0] = end_time_and_date_gc_3[0].strip().split('-')
    end_time_and_date_gc_3[1] = end_time_and_date_gc_3[1].strip().split(':')
    end_time_and_date_gc_3[0] = [float(i) for i in end_time_and_date_gc_3[0]]
    end_time_and_date_gc_3[1] = [float(i) for i in end_time_and_date_gc_3[1]]
    end_date_gc_3 = end_time_and_date_gc_3[0]
    end_time_gc_3 = end_time_and_date_gc_3[1]
    end_year_gc_3 = int(end_date_gc_3[0])
    end_month_gc_3 = int(end_date_gc_3[1])
    end_day_gc_3 = int(end_date_gc_3[2])
    end_hour_gc_3 = int(end_time_gc_3[0])
    end_minute_gc_3 = int(end_time_gc_3[1])
    end_second_gc_3 = end_time_gc_3[2]
    
# Converting the dates and times to unix
unix_start_time_gc_3 = to_unix_time(start_month_gc_3,
                                    start_day_gc_3,
                                    start_year_gc_3,
                                    start_hour_gc_3,
                                    start_minute_gc_3,
                                    start_second_gc_3)

unix_end_time_gc_3 = to_unix_time(end_month_gc_3,
                                  end_day_gc_3,
                                  end_year_gc_3,
                                  end_hour_gc_3,
                                  end_minute_gc_3,
                                  end_second_gc_3)

# Reading the data of the fourth run for the circle
with open(file_gc_4_relative_path, 'r') as file_gc_4:
    content_gc_4 = file_gc_4.readlines()
    # Reading the start date and time
    start_time_and_date_gc_4 = content_gc_4[1].strip().split(',')[3].strip('"')
    start_time_and_date_gc_4 = start_time_and_date_gc_4.strip().split()
    start_time_and_date_gc_4[0] = start_time_and_date_gc_4[0].strip().split('-')
    start_time_and_date_gc_4[1] = start_time_and_date_gc_4[1].strip().split(':')
    start_time_and_date_gc_4[0] = [float(i) for i in start_time_and_date_gc_4[0]]
    start_time_and_date_gc_4[1] = [float(i) for i in start_time_and_date_gc_4[1]]
    start_date_gc_4 = start_time_and_date_gc_4[0]
    start_time_gc_4 = start_time_and_date_gc_4[1]
    start_year_gc_4 = int(start_date_gc_4[0])
    start_month_gc_4 = int(start_date_gc_4[1])
    start_day_gc_4 = int(start_date_gc_4[2])
    start_hour_gc_4 = int(start_time_gc_4[0])
    start_minute_gc_4 = int(start_time_gc_4[1])
    start_second_gc_4 = start_time_gc_4[2]

    # Reading the end date and time
    end_time_and_date_gc_4 = str(content_gc_4[2].strip().split(',')[3]).strip('"')
    end_time_and_date_gc_4 = end_time_and_date_gc_4.strip().split()
    end_time_and_date_gc_4[0] = end_time_and_date_gc_4[0].strip().split('-')
    end_time_and_date_gc_4[1] = end_time_and_date_gc_4[1].strip().split(':')
    end_time_and_date_gc_4[0] = [float(i) for i in end_time_and_date_gc_4[0]]
    end_time_and_date_gc_4[1] = [float(i) for i in end_time_and_date_gc_4[1]]
    end_date_gc_4 = end_time_and_date_gc_4[0]
    end_time_gc_4 = end_time_and_date_gc_4[1]
    end_year_gc_4 = int(end_date_gc_4[0])
    end_month_gc_4 = int(end_date_gc_4[1])
    end_day_gc_4 = int(end_date_gc_4[2])
    end_hour_gc_4 = int(end_time_gc_4[0])
    end_minute_gc_4 = int(end_time_gc_4[1])
    end_second_gc_4 = end_time_gc_4[2]
    
# Converting the dates and times to unix
unix_start_time_gc_4 = to_unix_time(start_month_gc_4,
                                    start_day_gc_4,
                                    start_year_gc_4,
                                    start_hour_gc_4,
                                    start_minute_gc_4,
                                    start_second_gc_4)

unix_end_time_gc_4 = to_unix_time(end_month_gc_4,
                                  end_day_gc_4,
                                  end_year_gc_4,
                                  end_hour_gc_4,
                                  end_minute_gc_4,
                                  end_second_gc_4)
