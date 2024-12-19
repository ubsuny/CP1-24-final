"""
This module contains functions to convert between various units. The conversions 
currently supported are:
    - Feet --> Yards
    - Yards --> Feet
    - Feet --> Meters
    - Meters --> Feet
    - Yards --> Meters
    - Meters --> Yards
    - Feet --> Kilometers
    - Kilometers --> Feet
    - Meters --> Kilometers
    - Kilometers --> Meters
"""

def feet2yards(ft):
    '''
    Calculates the number of yards equivalent to the input number of feet

    Parameters:
    - ft: The number of feet which will be converted to yards

    Returns:
    - Yards
    '''
    return ft/3

def yards2feet(yards):
    '''
    Calculates the number of feet equivalent to the input number of yards

    Parameters:
    - yards: The number of yards which will be converted to feet

    Returns:
    - feet
    '''
    return yards*3

def feet2meters(ft):
    '''
    Calculates the number of meters equivalent to the input number of feet

    Parameters:
    - ft: The number of feet which will be converted to meters

    Returns:
    - meters
    '''
    return 0.3048*ft

def meters2feet(meters):
    '''
    Calculates the number of feet equivalent to the input number of meters

    Parameters:
    - meters: The number of meters which will be converted to feet

    Returns:
    - feet
    '''
    return meters/0.3048

def yards2meters(yards):
    '''
    Calculates the number of meters equivalent to the input number of yards

    Parameters:
    - yards: The number of yards which will be converted to meters

    Returns:
    - meters
    '''
    return 0.9144*yards

def meters2yards(meters):
    '''
    Calculates the number of yards equivalent to the input number of meters

    Parameters:
    - meters: The number of meters which will be converted to yards

    Returns:
    - yards
    '''
    return meters/0.9144

def feet2kilometers(ft):
    '''
    Calculates the number of kilometers equivalent to the input number of feet

    Parameters:
    - ft: The number of feet which will be converted to kilometers

    Returns:
    - kilometers
    '''
    return 0.0003048*ft

def kilometers2feet(km):
    '''
    Calculates the number of feet equivalent to the input number of kilometers

    Parameters:
    - km: The number of kilometers which will be converted to feet

    Returns:
    - feet
    '''
    return km/0.0003048

def meters2kilometers(meters):
    '''
    Calculates the number of kilometers equivalent to the input number of meters

    Parameters:
    - meters: The number of meters which will be converted to kilometers

    Returns:
    - kilometers
    '''
    return meters*1000

def kilometers2meters(km):
    '''
    Calculates the number of meters equivalent to the input number of kilometers

    Parameters:
    - km: The number of kilometers which will be converted to meters

    Returns:
    - meters
    '''
    return km/1000
