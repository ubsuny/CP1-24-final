"""
This module is to check the distance calculator module by passing the gps data file path
"""

from distance_calculator import load_gps_data, calculate_distances


# Replace with your actual file path
GPS_DATA_FILE_PATH = "dhamalakamal/Data/Circular_path_GPS.csv"
gps_data = load_gps_data(GPS_DATA_FILE_PATH)

if gps_data is not None:
    distances = calculate_distances(gps_data)
    print("Distances between adjacent GPS positions (in meters):")
    for idx, distance in enumerate(distances, start=1):
        print(f"Distance {idx}: {distance:.2f} meters")
