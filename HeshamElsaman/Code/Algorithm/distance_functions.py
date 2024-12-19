"""
This file is for the definition of the function that calculates the distance
on the surface of a ellipsoid (spheroid) using the Vincenty's Inverse Formula
adapted from https://www.ngs.noaa.gov/PUBS_LIB/inverse.pdf

Parameters:
a,  b, major and mninor semiaxes of the ellipsoid.
f, flattening = (a-b)/a.
phi, geodetic latitude, positive north of the equator.
longdiff, difference in longitude, positive east.
s, length of the geodesic.
alpha1, alpha2, azimuths of the geodesic, clockwise from north; alpha2 is the direction P1.P2 produced.
alpha, azimuth of the geodesic at the equator.
u ** 2 = (cos(alpha)) ** 2 * (a ** 2 - b ** 2)/b ** 2.
redlat, reduced latitude, defined by tan(redlat)= (1-f) * tan(phi).
longdiff_sphere, difference in longilude on an auxiliary sphere.
sigma, angular distance P1.P2 on the sphere.
sigma_1, angular distance on the sphere from the equator to P1
sigma_m angular distance on the sphere from the equator to the midpoint of the line. 
"""

from libraries_module import np

def dist(long_1, long_2, lat_1, lat_2, a, b, f):
    """
    Returns the distance between two points on the surface of an ellipsoid
    
    Parameters:

    """
    longdiff = np.absolute(long_1 - long_2)
    redlat_1 = (1 - f) * np.tan(lat_1)
    redlat_2 = (1 - f) * np.tan(lat_2)
    sin_redlat_1 = np.sin(redlat_1)
    cos_redlat_1 = np.cos(redlat_1)
    sin_redlat_2 = np.sin(redlat_2)
    cos_redlat_2 = np.cos(redlat_2)
    
    # initial guess
    longdiff_sphere = longdiff
    
    while 1:
        longdiff_sphere_0 = longdiff_sphere
        sin_longdiff_sphere = np.sin(longdiff_sphere)
        cos_longdiff_sphere = np.cos(longdiff_sphere)
        sin_sigma = np.sqrt((cos_redlat_2 * sin_longdiff_sphere) ** 2 + (cos_redlat_1 * sin_redlat_2 - sin_redlat_1 * cos_redlat_2 * cos_longdiff_sphere) ** 2)
        cos_sigma = sin_redlat_1 * sin_redlat_2 + cos_redlat_1 * cos_redlat_2 * cos_longdiff_sphere
        # tan_sigma = sin_sigma / cos_sigma
        sigma = np.acos(cos_sigma)
        sin_alpha = cos_redlat_1 * cos_redlat_2 * sin_longdiff_sphere / sin_sigma
        cos_2sigma_m = cos_sigma - 2 * sin_redlat_1 * sin_redlat_2 / (1 - (sin_alpha ** 2))
        ccc = (f / 16) * (1 - (sin_alpha ** 2)) * (4 + f * (4 - 3 * (1 - (sin_alpha ** 2))))
        longdiff_sphere = longdiff + (1 - ccc) * f * sin_alpha * (sigma + ccc * sin_sigma * (cos_2sigma_m + ccc * cos_sigma * (-1 + 2 * (cos_2sigma_m ** 2))))
        if np.absolute(longdiff_sphere - longdiff_sphere_0) < 10 ** (-12):
            break

        u_squared = (1 - sin_alpha ** 2) * (a ** 2 - b ** 2) / b ** 2
        aaa = 1 + (u_squared / 16384) * (4096 + u_squared * (-768 + u_squared * (320 - 175 * u_squared)))
        bbb = (u_squared / 1024) * (256 + u_squared * (-128 + u_squared * (74 - 47 * u_squared)))
        del_sigma = bbb * sin_sigma * (cos_2sigma_m + (bbb / 4) * cos_sigma * (-1 + 2 * cos_2sigma_m ** 2) - (bbb / 6) * cos_2sigma_m * (-3 + 4 * sin_sigma ** 2) * (-3 + 4 * cos_2sigma_m ** 2))
        
        # finally, the distance:
        s = b * aaa * (sigma - del_sigma)

        return s

# Earth's parameters
# a
SEMI_MAJOR_AXIS = 6378137.0
# b
SEMI_MINOR_AXIS = 6356752.314245
# 1/f
INVERSE_FLATTENING = 298.257223563

# Function that calculates the distance on Earth
def earth_dist(longitude_1, longitude_2, latitude_1, latitude_2):
    """
    Returns the distance between two angular positions (in radians) on the Earth's surface

    Parameters:
    Inputs:
    longitude_1 (number), latitude_1 (number), longitude_2 (number), latitude_2 (number):
    the angular positions of the first and second points respectively
    Outputs:
    distance (number): the desired distance in (km)
    """
    distance = dist(longitude_1, longitude_2, latitude_1, latitude_2,
                    SEMI_MAJOR_AXIS, SEMI_MINOR_AXIS, INVERSE_FLATTENING)
    return distance
