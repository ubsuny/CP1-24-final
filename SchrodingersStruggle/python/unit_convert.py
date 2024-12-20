"""
This module creates functions capable of converting
yards and feet into their SI equivilant of meters and vice verca.
"""
#Conversion factor of feet to meters:
FEET_TO_METERS = 0.3048

def yrd_m(yards):
    """
    Converts input in yards into meters equivilant.
    """
    meters = FEET_TO_METERS * 3 * yards
    return meters

def ft_m(feet):
    """
    Converts input in feet into meters equivilant.
    """
    meters = FEET_TO_METERS * feet
    return meters

def m_ft(meters):
    """
    Converts input in meters to feet equivilant.
    """
    feet = 1/FEET_TO_METERS * meters
    return feet

def m_yrd(meters):
    """
    Converts input in meters to yards equivilant.
    """
    yards = 1/(3*FEET_TO_METERS) * meters
    return yards
