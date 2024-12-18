import final as fin

def test_fahrenheit_to_kelvin():
    '''
    unit testing function for the fahrenheit to kelvin converter
    '''
    assert fin.fahrenheit_to_kelvin(32) == 273.15
