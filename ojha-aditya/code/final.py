"""
Module containing all importable functions namely, 
temperature converter, temperature reading function,
function to list all meta file names, non-linear fitting function, 
function for resampling (to 2^n points) and 
checking equidistant data, numpy implementations of Fourier transform 
and inverse Fourier transform along with
the pure python function to calculate frequency axis. 
"""

import os
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d

def fahrenheit_to_kelvin(temp_in_f):
    '''
    Function to convert temperature in Fahrenheit to Kelvin scale.

    Parameters:
    - temp_in_f: Temperature in Fahrenheit

    Returns:
    - temp_in_k: Temperature in Kelvin
    '''
    temp_in_k = ((temp_in_f - 32)*5/9) + 273.15
    return temp_in_k

def read_temperature(meta_file_path):
    '''
    Parser function to read the temperature of one markdown file.
    (gives an unphysical value if nothing is found in the file)

    Parameters:
    - meta_file_path: location of the meta file

    Returns:
    - temp_in_f: temperature in Fahrenheit
    '''
    with open(meta_file_path, 'r', encoding='utf-8') as meta:
        metadata = meta.readlines()

    temp_in_f = -10000 # default unphysical value
    for line in metadata:
        if line.startswith("Temperature ($\\text{\\textdegree}$F):"):
            temp_in_f = line.split(":")[1].strip()

    return float(temp_in_f)

def filename_lister(directory, filename_filter, extension):
    '''
    Function to generate a list of files with the required extension
    containing the filename_filter

    Parameters:
    - directory: directory containing the files
    - filename_filter: string required to be searched
    - extension: file extension such as '.md'

    Returns:
    - file_list: list of filenames containing the given filter and extension
    '''
    file_list = []

    for filename in os.listdir(directory):
        if filename.endswith(extension) and filename_filter in filename:
            file_list.append(filename)

    return file_list

def non_linear_fit(ansatz, x, y, initial_guess, n = 6):
    '''
    Function to perform non-linear least squares fitting using gradient-free iterative method
    with a tolerance of 1e-6 or 10000 interations (whichever occurs first).

    Parameters:
    - ansatz: fitting function (callable) such as np.sin()
    - x: x (independent) coordinate numpy array
    - y: y (dependent) coordinate numpy array
    - initial_guess: dictionary of initial guesses for the parameters of the ansatz
    - n: exponent for number of steps 2^n (default n = 6)

    Returns:
    - params: dictionary of best-fit parameters
    '''
    # resampling if not equidistant data
    if not is_equidistant(x):
        x, y = resample_data(x, y, n)

    params = initial_guess.copy()
    param_names = list(initial_guess.keys())
    #step_size = 1e-4  # Small step for numerical gradient approximation

    for _ in range(100000):
        # Compute the residuals
        y_pred = ansatz(x, **params)
        rss = np.sum((y - y_pred) ** 2)  # Residual sum of squares

        # Approximate numerical gradient
        gradients = {}
        for p in param_names:
            params_step = params.copy()
            params_step[p] += 1e-4
            y_pred_step = ansatz(x, **params_step)
            gradients[p] = (np.sum((y - y_pred_step) ** 2) - rss) / 1e-4

        # Update parameters
        params_new = {p: params[p] - 1e-4 * gradients[p] for p in param_names}

        # Check for convergence
        max_change = max(abs(params_new[p] - params[p]) for p in param_names)
        if max_change < 1e-6:
            break

        params = params_new

    return params

def get_coordinates(file_path):
    '''
    Function to read the CSV file and convert GPS data to x, y coordinates 
    using Mercator projection.
    
    Parameters:
    - file_path: path to the CSV file.
    
    Returns:
    - x: a numpy array of x values
    - y: a numpy array of y values
    '''
    # Read the CSV file into a pandas DataFrame
    df = pd.read_csv(file_path)

    # Extract latitude and longitude columns
    latitudes = df['Latitude (°)'].values
    longitudes = df['Longitude (°)'].values

    radius = 6371001 # radius of Earth in metre

    # Convert latitude and longitude from degrees to radians
    lat_rad = np.radians(latitudes)
    lon_rad = np.radians(longitudes)

    # Mercator projection
    x = radius * lon_rad
    y = radius * np.log(np.tan(np.pi / 4 + lat_rad / 2))

    return np.array(x), np.array(y)

def shift_coordinates(x, y):
    '''
    Function to shift coordinates to have origin at the starting point

    Parameters:
    - x: array of x values
    - y: array of y values

    Returns:
    - x_shift: shifted x array
    - y_shift: shited y array
    '''
    # shifting the origin to starting point
    x_shift = x - x[0]
    y_shift = y - y[0]

    return np.array(x_shift), np.array(y_shift)

def resample_data(x, y, n):
    '''
    Function to resample the nonequidistant data to nearest 2^n 
    equidistant data points.

    Parameters:
    - x: nonequidistant numpy x (independent) array
    - y: raw numpy y (dependent) array
    - n: the exponent for 2^n

    Returns:
    - x_equidistant: resampled equidistant x array
    - y_equidistant: resampled interpolated y array
    '''
    no_of_points = 2 ** n # number of points for fft
    x_equidistant = np.linspace(x[0], x[-1], no_of_points) # interpolating x values to 2^n

    y_equidistant = interp1d(x, y, kind='linear')(x_equidistant)
    #np.interp(x_equidistant, x, y)

    return np.array(x_equidistant), np.array(y_equidistant)

def get_frequency_axis(x_equidistant, unit = 1/100):
    '''
    Function to generate the frequency axis to 1/(100m) default unit

    Parameters:
    - x_equidistant: equidistant array for x values
    - unit: apprpriate unit for frequency axis in 1/m (default 1/100)

    Returns:
    - freq_axis = list of frequencies in given unit
    '''
    freq_axis = []
    x_eq = np.array(x_equidistant)
    no_of_points = len(x_equidistant)

    dx = x_eq[1] - x_eq[0]

    for i in range(no_of_points):
        f = i/(unit * no_of_points * dx) # frequency value for x[i]
        freq_axis.append(f)
    return freq_axis

def is_equidistant(x):
    '''
    Function to check if the given data is equidistant or not

    Parameters: 
    - data: input array of independent variable

    Returns:
    - bool: True for equidistant, False for non-equidistant
    '''
    tmp = np.array(x[1:]) - np.array(x[:-1])
    return np.allclose(tmp, tmp[0])

def numpy_wrapper_fft(x, y, n = 6):
    '''
    Function to implement Fourier transform using numpy fft function
    
    Parameters:
    - x: independent variable array
    - y: dependent variable array
    - n: exponent for number of steps 2^n
    
    Returns:
    - Fourier transform of the given data 
    - Frequency values
    '''
    if not is_equidistant(x):
        x, y = resample_data(x, y, n)

    return np.fft.fft(y), get_frequency_axis(x, 1/100)

def numpy_wrapper_ifft(fft_data):
    '''
    Function to implement numpy wrapper for inverse Furier transform

    Parameters:
    - fft_data: array of Furier tranform data

    Returns:
    - Inverse Fourier tranform of the given data
    '''
    return np.fft.ifft(fft_data)
