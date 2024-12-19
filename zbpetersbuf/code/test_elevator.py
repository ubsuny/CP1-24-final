"""
test_elevator.py
this moduale is the unit test for elevator.py
"""
import pandas as pd
import numpy as np
from elevator import dimotion

dta = pd.DataFrame({'Time (s)': [1,2,3], 'Linear Acceleration x (m/s^2)': [3,1,1],
                    'Linear Acceleration y (m/s^2)': [3,2,2],
                    'Linear Acceleration z (m/s^2)': [3,2,3]})

def func():
    """this just outputs the expected values in the ame manner that the function imptomec does"""
    a = [[0,0,0],[3,3,3],[4,5,5]]
    b = [1,2,3]
    return np.array(a), b
v_and_t = func()

def test_imptomec():
    """This tests the imptomec function"""
    result = dimotion(dta)
    assert result[1] == v_and_t[1]

print((2024-1970)//4)
