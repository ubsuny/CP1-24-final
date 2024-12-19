"""
imp_to_metr.py
this module contains the relevent cod for the feet and yard to meter convertion
"""
def imptomec(x,lenguni):
    """converts from feet or yard to meters"""
    if lenguni == 'yard':
        x = 3*x
    return 0.305*x
