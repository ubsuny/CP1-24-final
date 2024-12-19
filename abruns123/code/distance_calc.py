"""
The distance module includes functions
revolving around calculating the distance 
between two sets of longitudes and latitudes
in units of meters. 
"""
import pandas as pd
import numpy as np
import unit_converter as unit_converter


def diff(lat1,lat2, lon1,lon2):
    """
    diff calculates the distance between a pair
    of latitudes and longitudes.
    """

    #the latitudes and longitudes are converted to feet
    lat1, lat2=lat1*364000,lat2*364000
    lon1, lon2=lon1*288200,lon2*288200

    #The difference between the lat and lons are returned
    #as an array that represents x,y coordinates
    return np.array([(lon2-lon1),(lat2-lat1)])

def diffm(lat1, lat2,lon1, lon2):
    """
    diffm calls diff such that the distance between the two 
    points is represented in x, y coordinates and converts 
    from feet to meters. 
    """
    return unit_converter.ft_to_m(diff(lat1,lat2, lon1, lon2))

def reader(path):
    """
    reader is meant to take in a Location file
    and isolates the columns containing latitude 
    and longitude information
    """
    file=pd.read_csv(path)

    #condition1 and 2 are used to verify that
    #a file with the correct data is input
    condition1=False
    condition2=False
    for c in file:
        if c=="Latitude (°)":
            condition1=True
        if c=="Longitude (°)":
            condition2=True
    if condition1 is True and condition2 is True:
        lats=file["Latitude (°)"]
        lons=file["Longitude (°)"]
        return lats, lons
    return "Bad File. Does not have both latitude and longitude data"

def get_coords(d,xy,i):
    """
    get_coords uses recursion to determine 
    xy coordinates of a position after a 
    series of displacements.
    """

    if i==0:
        return xy
    return xy+get_coords(d, d[i-1], i-1)
