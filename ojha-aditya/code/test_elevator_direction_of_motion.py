'''
Module to test direction of motion generation for elevator
'''

import numpy as np
from elevator_direction_of_motion import calculate_velocity

def test_calculate_velocity() -> None:
    '''
    Function to test that calculate_velocity returns the correct result when given dummy parameters.
    '''
    vx,vy,vz = calculate_velocity(np.arange(1,6), np.ones(5), np.ones(5), np.ones(5))
    assert np.all(np.sqrt(vx**2+vy**2+vz**2)==1)
