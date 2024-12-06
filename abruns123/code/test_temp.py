"""
the test_temp module acts as a unit test for the 
f_k function from final.py which converts a fahrenheit 
temperature to kelvin
"""
import final
import numpy as np

def test_f_k():
    assert(final.f_k(32)==0)
    assert np.isclose(final.f_k(98), 309.817, rtol=1e-3)