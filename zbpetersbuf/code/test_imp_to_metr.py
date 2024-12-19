"""
test_imp_to_metr.py
this modulae is the unit test for imp_to_metr.py
"""
from imp_to_metr import imptomec

def test_imptomec():
    """This tests the ohmscur function"""
    assert imptomec(2, 'yard')==1.83
    assert imptomec(3, 'foot')==0.915
