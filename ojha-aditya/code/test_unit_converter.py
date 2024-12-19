'''
unit testing module for unit_converter.py
'''
from unit_converter import yard_to_metre, foot_to_metre

def test_yard_to_metre():
    '''
    function to test the yard to metre conversion
    '''
    assert yard_to_metre(1) == 0.9144   # checking if 1 yard = 0.9144 metre or not

def test_foot_to_metre():
    '''
    function to test the foot to metre conversion
    '''
    assert foot_to_metre(1) == 0.3048   # checking if 1 foot = 0.3048 metre or not
