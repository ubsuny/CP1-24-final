import final
import numpy as np
import matplotlib.pyplot as plt

def test_nl():
    x_val=np.array(np.linspace(.1,2*np.pi,101))
    y=np.array([5*np.sin(2.5*x+.1) for x in x_val])
    new_p=final.gauss_newton(x_val,y,[4,2.3,1], 10)[0]
    assert np.isclose(new_p[0], 5, rtol=1e-5)
