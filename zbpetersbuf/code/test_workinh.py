"""
test_workinh.py
this moduale is the unit test for workinh.py
"""

import pandas as pd
import numpy as np
from workinh import walkeq, distbetp, gpsloc

dta_1 = [0,1,4,9,16,25]
matr = np.zeros(5)
i=0
while i < 5:
    matr[i]=dta_1[i+1]-dta_1[i]
    i+=1

def test_distbetp():
    """This tests the distbetp function"""
    assert int(distbetp(matr,2)) == (4-1)
    assert int(distbetp(matr,4)) == (16-9)

walk = pd.DataFrame({'Latitude (°)': [43,43.1,43.2],
                     'Longitude (°)': [-78.8,-78.9,-79],
                    'Altitude WGS84 (m)': [150,150.1,150.2]})

gpwl = gpsloc(walk)

matr_1 = np.zeros(3)
j=0
while j < 3-1:
    matr_1[j+1]=(((gpwl[0][j+1]-gpwl[0][j])**(2))+((gpwl[1][j+1]-gpwl[1][j])**2))**(1/2)
    j+=1

def test_walkeq():
    """This tests the distbetp function"""
    resl = walkeq(walk)
    assert resl[1] > matr_1[1] - 6
    assert resl[1] < matr_1[1] + 6
    assert resl[2] > matr_1[2] - 6
    assert resl[2] < matr_1[2] + 6
