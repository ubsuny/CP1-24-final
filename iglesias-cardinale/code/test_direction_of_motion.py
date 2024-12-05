'''
This module tests the direction of motion velocity function
'''

import numpy as np
from direction_of_motion import velocity

def test_zero_a():
    '''Tests that the velocity is v_0 if the input acceleration is zero'''
    a = np.zeros(100)  # Zero acceleration
    t = np.linspace(0, 10, 100)  # Time range from 0 to 10 seconds
    v_0 = [0, 1, 10, 100, 1000]  # Initial velocities to test
    # Check that velocity remains constant (equal to v_0) for each initial velocity
    for v in v_0:
        result = velocity(t, a, v)
        expected = np.full(100, v)  # Expect velocity to stay constant
        np.testing.assert_array_almost_equal(result, expected, decimal=6)
