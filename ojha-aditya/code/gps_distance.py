'''
This module calculates the distance between two adjacent GPS positions using the Haversine distance
'''
import numpy as np
import pandas as pd

def read_data(fname):
    '''
    Function to read the latitude, longitude and altitude readings from data files
    '''
    data_gps = pd.read_csv(fname)
    latitude_arr = np.array(data_gps['Latitude (°)'])
    longitude_arr = np.array(data_gps['Longitude (°)'])
    altitude_arr = np.array(data_gps['Altitude WGS84 (m)'])
    return latitude_arr, longitude_arr, altitude_arr

def give_distance(
    latitude_arr,
    longitude_arr,
    altitude_arr,
    r=6_378_137,
):
    '''
    Function to return the array with distances between two adjacent GPS positions in meters
 
            Parameters:
                    latitude_arr: Array containing latitudes in degrees.
                    longitude_arr: Array containing longitudes in degrees.
                    altitude_arr: Array containing altitudes in meters.
                    r: Radius of Earth in meters. Default R = 6,378,137 m.

            Returns:
                    distance: Distance between two adjacent GPS positions in meters.
    '''

    phi1 = np.radians(latitude_arr[:-1])
    phi2 = np.radians(latitude_arr[1:])
    lambda1 = np.radians(longitude_arr[:-1])
    lambda2 = np.radians(longitude_arr[1:])
    h1 = altitude_arr[:-1]
    h2 = altitude_arr[1:]
    delta_h = h2 - h1

    delta_phi = phi2 - phi1
    delta_lambda = lambda2 - lambda1

    distance_lateral = (
        2
        * r
        * np.arcsin(
            np.sqrt(
                0.5
                * (
                    1
                    - np.cos(delta_phi)
                    + np.cos(phi1) * np.cos(phi2) * (1 - np.cos(delta_lambda))
                )
            )
        )
    )
    distance_vertical = delta_h
    return np.sqrt(distance_lateral**2 + distance_vertical**2)
