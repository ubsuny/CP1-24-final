"""
this function tests the unit_converter module
"""
import numpy as np
from unit_converter import ft_to_m, yd_to_m


def test_ft_to_m():
    """
    test_fit_to_m verifies that the 
    function successfully carries out 
    unit conversion from feet to meters
    """
    np.testing.assert_allclose(ft_to_m(1),(.3048))

def test_yd_to_m():
    """
    test_fit_to_m verifies that the 
    function successfully carries out 
    unit conversion from yards to meters
    """
    np.testing.assert_allclose(yd_to_m(1), .9144)