"""
This module contains functions to find distances between two GPS positions
"""

import numpy as np

def gps_distance_flat_earth(x1,x2):
    '''
    Calculates the absolute distance between two adjacent gps points assuming a flat Earth

    Parameters:
    - x1: List of values containing the x, y and z values of the first position. 
          Should have the same units as x2.
    - x2: List of values containing the x, y and z values of the second position.
          Should have the same units as x1.

    Returns:
    - Absolute distance between x1 and x2 in the same units as x1 and x2
    '''

    return (sum((x2[i]-x1[i])**2 for i in range(len(x1))))**0.5

#Useful constants

#Radii of Earth using the WGS84 ellipsoid (as used in phyphox), according to
#https://gscommunitycodes.usf.edu/geoscicommunitycodes/public/geophysics/
R_EQUATOR = 6378137 #Meters
R_POLE = 6356752 #Meters

#Flattening
FLAT = (R_EQUATOR - R_POLE)/R_EQUATOR

def r_earth(phi):
    '''
    Calculates the radius of the earth in the WGS84 ellipsoid at a givel latitude
    
    Parameters: 
    - phi: Angle of latitude at which we are finding the radius
    
    Returns:
    - Radius of Earth at given latitude
    '''
    return R_EQUATOR*(1-FLAT*np.sin(phi)**2)

def gps_wgs84(lat,lon,alt):
    '''
    Converts GPS position given by latitude, longitude and altiude using the WGS84 
    ellipsoid into x, y and z coordinates

    parameters:
    - lat: latitude in radians
    - lon: longitude in radians
    - alt: altitude in meters in the WGS84 ellipsoid

    returns:
    - x, y, z: position along the x y and z axes

    '''

    r = r_earth(lat) + alt

    x = r * np.cos(lat) * np.cos(lon)
    y = r * np.cos(lat) * np.sin(lon)
    z = r * np.sin(lat)

    return x, y, z

def gps_distance_wgs84(p1,p2):
    '''
    Calculates the distance between two points given by latitude, longitude and altitude 
    using the WGS84 ellipsoid.

    parameters:
    - p1: List or array of the form [lat, lon, alt] for the first point
    - p2: List or array of the form [lat, lon, alt] for the second point

    returns:
    - absolute distance between p1 and p2
    '''

    x1, y1, z1 = gps_wgs84(p1[0], p1[1], p1[2])
    x2, y2, z2 = gps_wgs84(p2[0], p2[1], p2[2])

    d = np.sqrt((x2-x1)**2+(y2-y1)**2+(z2-z1)**2)

    return d
