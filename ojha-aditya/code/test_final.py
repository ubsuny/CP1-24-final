'''
Unit test module for functions in final.py implemented using pytest using mock data files
'''
import numpy as np
#import pytest
#import unittest
import final as fin

def test_fahrenheit_to_kelvin():
    '''
    unit testing function for the fahrenheit to kelvin converter
    '''
    assert fin.fahrenheit_to_kelvin(32) == 273.15

def test_read_temperature():
    '''
    unit testing function for read_temperature
    '''
    assert fin.read_temperature('/workspaces/CP1-24-final/OA001_sinewalktrial.md') == 34

def test_filename_lister():
    '''
    unit testing function for filename_lister
    '''
    assert fin.filename_lister('/workspaces/CP1-24-final', 'sinewalktrial',
                                '.md') == ['OA001_sinewalktrial.md', 'OA002_sinewalktrial.md']

def simple_ansatz(x, a, b, c):
    '''
    ansatz to test non_linear_fit function
    '''
    return a * np.sin(b * x + c)

def test_non_linear_fit():
    '''
    unit testing function for non_linear_fit
    '''
    x = np.linspace(0, 10, 100)
    y = 2 * np.sin(x)  # Known function with a = 2, b = 1, c = 0
    initial_guess = {'a': 1.0, 'b': 1.0, 'c': 0.0}  # Initial guesses close to true values

    # Perform the fitting
    result = fin.non_linear_fit(simple_ansatz, x, y, initial_guess)

    # Check if the result is close to the true parameters
    assert np.isclose(result['a'], 2.0, atol=0.1)
    assert np.isclose(result['b'], 1.0, atol=0.1)
    assert np.isclose(result['c'], 0.0, atol=0.1)


# Sample data from the trial file
mock_csv_data = {
    'Latitude (°)': [4.300233480E1, 4.300225333E1, 4.300225333E1, 4.300225000E1, 4.300225000E1],
    'Longitude (°)': [-7.879122580E1, -7.879107500E1, -7.879107500E1, -7.879106833E1, 
                      -7.879107000E1],
}

# Expected x, y coordinates from the Mercator projection
expected_x = np.array([-8761185.94823218, -8761169.18003461, -8761169.18003461,
        -8761168.43836433, -8761168.62405989])
expected_y = np.array([5306383.6854801 , 5306371.29832689, 5306371.29832689,
        5306370.79201546, 5306370.79201546])

def test_get_coordinates():
    '''
    unit test for the get_coordinates function
    '''
    file_path = '/workspaces/CP1-24-final/OA001_sinewalktrial.csv'  # trial file path
    x, y = fin.get_coordinates(file_path)

    # Test if the x and y values are close to the expected ones
    assert np.allclose(x, expected_x)
    assert np.allclose(y, expected_y)

def test_shift_coordinates():
    '''
    unit test for shift_coordinates function
    '''
    file_path = '/workspaces/CP1-24-final/OA001_sinewalktrial.csv'  # trial file path
    x, y = fin.get_coordinates(file_path)

    x_shift, y_shift = fin.shift_coordinates(np.array(x), np.array(y))

    # checking the values with the expected values for trial file entries
    assert x_shift[0] == [ 0.]
    assert y_shift[0] == [ 0.]

def test_resample_data():
    '''
    unit test for resample_data function
    '''
    #testing the resampling function against data from trial file
    file_path = '/workspaces/CP1-24-final/OA001_sinewalktrial.csv'  # trial file path
    x, y = fin.get_coordinates(file_path)
    x_shift, y_shift = fin.shift_coordinates(np.array(x), np.array(y))
    x_equidistant, y_equidistant = fin.resample_data(np.array(x_shift),
                                                     np.array(y_shift), 3)

    assert np.isclose(x_equidistant[1], np.float64(2.47488176))
    assert np.isclose(y_equidistant[1], np.float64(-1.82826683))

def test_get_frequency_axis():
    '''
    unit test for get_frequency_axis function
    '''
    #testing the get_frequency_axis function against data from trial file
    file_path = '/workspaces/CP1-24-final/OA001_sinewalktrial.csv'  # trial file path
    x, y = fin.get_coordinates(file_path)
    x_shift, y_shift = fin.shift_coordinates(np.array(x), np.array(y))
    x_equidistant, y_equidistant = fin.resample_data(np.array(x_shift),
                                                     np.array(y_shift), 3)

    # getting frequency_axis for trial file
    freq_axis = fin.get_frequency_axis(x_equidistant, 1/100)

    #checking for values expected from trial file
    assert np.isclose(freq_axis[1], 5.050746343534408)
    assert np.isclose(y_equidistant[1]+freq_axis[2], np.float64(-1.82826683+10.101492687068816))

def test_is_equidistant():
    '''
    unit test for is_equidistant function
    '''
    x = [0, 3, 4, 8, 19] # trial non equidistant array
    assert fin.is_equidistant(x) is False # expecting false result

def test_numpy_wrapper_fft():
    '''
    unit test for numpy_wrapper_fft
    (raises warning as we are comparing real part of complex values)
    '''
    # getting data from trial file
    x, y = fin.get_coordinates('/workspaces/CP1-24-final/OA001_sinewalktrial.csv')

    # taking fft of trial data
    fft, freq = fin.numpy_wrapper_fft(x, y, 3)

    # comparing expected values
    assert np.isclose(freq[0], 0.)
    assert np.isclose(np.float64(fft[0]), 4.24510183e+07+0.j)

def test_numpy_wrapper_ifft():
    '''
    unit test for numpy_wrapper_ifft function
    '''
    # taking the inverse Fourier transform of the result from fft unit test

    ifft = fin.numpy_wrapper_ifft([4.24510183e+07 +0.j        , 7.33510844e+00-17.63326524j,
        7.31306734e+00 -7.28189651j, 7.29102623e+00 -3.00713057j,
        7.28189651e+00 +0.j        , 7.29102623e+00 +3.00713057j,
        7.31306734e+00 +7.28189651j, 7.33510844e+00+17.63326524j])

    # expecting the original function back
    assert np.isclose(ifft[0], 5306383.6854801)
