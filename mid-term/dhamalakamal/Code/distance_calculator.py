import pandas as pd
import math

# Initialize global variables
# Radius of the Earth in meters
R = 6371000
# Required columns list
columns_needed = ["Latitude (°)","Longitude (°)"]


def load_gps_data(file_path):
    """
    Load GPS positions from the CSV file containing GPS coordinates.

    :param file_path: Path to the CSV file containing GPS positions.
    :return: DataFrame containing latitude and longitude.
    """
    try:
        # Read csv file with first row as header
        gps_data = pd.read_csv(file_path, header=0)

        # Extract only the latitude and longitude columns
        gps_lat_lon_data = gps_data[columns_needed]
        print("GPS data loaded successfully.")
        return gps_lat_lon_data
    except Exception as e:
        print(f"Error loading GPS data: {e}")
        return None

def haversine(lat1, lon1, lat2, lon2):
    """
    Calculate the great-circle distance between two points on the Earth's surface using the Haversine formula.
    
    :param lat1: Latitude of the first point.
    :param lon1: Longitude of the first point.
    :param lat2: Latitude of the second point.
    :param lon2: Longitude of the second point.
    :return: Distance in meters between the two points.
    """
    # Convert latitude and longitude from degrees to radians
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])

    # Haversine formula
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    c = 2 * math.asin(math.sqrt(a))
    
    # Distance in meters
    distance = R * c
    return distance

def calculate_distances(gps_data):
    """
    Calculate distances between adjacent GPS positions.
    
    :param gps_data: DataFrame containing GPS coordinates (latitude, longitude).
    :return: List of distances between adjacent GPS positions.
    """
    distances = []
    # Looping through each row of gps data to calculate distance among two adjacent coordinates
    for i in range(len(gps_data) - 1):
        lat1, lon1 = gps_data.iloc[i]
        lat2, lon2 = gps_data.iloc[i + 1]
        distance = haversine(lat1, lon1, lat2, lon2)
        distances.append(distance)
    return distances

