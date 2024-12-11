'''This module contains all of the functions necessary to complete 
the Final Exam for Computational Physics 1, 2024.'''

import os
import numpy as np
from scipy.optimize import curve_fit

def fahrenheit_to_kelvin(fahrenheit):
    '''This function converts a given temperature in Degrees Fahrenheit 
    to the equivalent absolute temperature in Kelvin.
    
    Parameters:
        -fahrenheit: Degrees Fahrenheit, to be converted
        
    Returns:
        -kelvin: Equivalent temperature in Kelvin
    '''
    kelvin = (fahrenheit-32)*5/9 + 273.15

    return kelvin

def get_temp(path):
    '''Reads the temperature of a given experiment from that 
    trials metadata markdown file
    
    Parameters:
        -path: String. Path to markdown file containing metadata
    
    Returns:
        -temp: temperature from metadata file'''
    data = np.loadtxt(path, unpack=True)
    temp = data[0]

    return temp


def list_files(directory, search_string, file_extension):
    """
    Generates a list of file paths in a directory (and its subdirectories)
    where the filenames contain a specific string.

    Parameters:
        directory (str): The path to the directory to search.
        search_string (str): The string to look for in filenames.
        file_extension (str): The file extension we're searching for

    Returns:
        list: A list of file paths matching the search criteria.
    """
    matching_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if search_string in file and file.endswith(file_extension):
                matching_files.append(os.path.join(root, file))
    return matching_files

def curvefitlin(xdata, ydata, xunc, yunc):
    '''
    Does nonlinear curve fitting for input data with uncertainty and function to be fit

    Parameters:
        -func: The fitting function
        -*args: Paremeters of func
        -xdata: x-data to be fit
        -ydata: y-data to be fit
        -xunc: uncertainty in x-data
        -yunc: uncertainty in y-data 

    Returns:
        -The fitting parameters of func
    '''

    n = len(xdata)

    if n<2:
        print('Error: Need more data!')
        exit()

    err = np.sum(np.sqrt(xunc**2+yunc**2))
    S = np.sum(1/err**2)

    if abs(S) < 0.00001 :
        print('Error: Denominator too small!')
        exit()

    S_x = np.sum(xdata/xunc)
    S_y = np.sum(ydata/yunc)

    t = (xdata - S_x/S)/err
    S_tt = np.sum(t**2)

    if abs(S_tt) < 0.00001:
        print('Error: Dnominator S too small!')
        exit()
    
    b = np.sum(t*ydata/err)/S_tt
    a = (S_y - S_x*b)/S
    sigma_a2 = (1+S_x**2/(S*S_tt))/S
    sigma_b2 = 1/S_tt

    if sigma_a2 < 0 or sigma_b2 < 0:
        print('Error: Negative Square Root')
        exit()
    
    sigma_a = np.sqrt(sigma_a2)
    sigma_b = np.sqrt(sigma_b2)

    chi_squared = np.sum(((ydata-a-b*xdata)/err)**2)

    return a, b, sigma_a, sigma_b, chi_squared

    import numpy as np

def curvefit(fitting_func, param_guess, xdata, ydata, yunc, learning_rate=1e-4, tol=1e-6, max_iter=10000):
    """
    Perform nonlinear curve fitting for input data with uncertainties and a fitting function.

    Parameters:
        - fitting_func (callable): The function to fit, f(x, *args)
        - param_guess (list): Initial guesses for the parameters
        - xdata (array): Independent variable data
        - ydata (array): Dependent variable data
        - yunc (array): Uncertainties in ydata
        - learning_rate (float): Step size for parameter updates
        - tol (float): Tolerance for stopping criteria
        - max_iter (int): Maximum number of iterations

    Returns:
        - optimized_args (list): Optimized parameters
        - sigma_args (list): Uncertainties in the optimized parameters
    """

    #Make Arrays:
    xdata = np.array(xdata)
    ydata = np.array(ydata)
    yunc = np.array(yunc)

    # Number of data points and parameters
    n = len(xdata)
    p = len(param_guess)
    
    if n < 2:
        print("Error: Need more data!")
    
    # Compute weights from uncertainties in y
    weights = 1 / (yunc**2)
    
    # Initialize parameters
    params = np.array(param_guess, dtype=float)
    
    # Gradient descent loop
    for iteration in range(max_iter):
        # Compute residuals and weighted sum of squared residuals
        residuals = ydata - fitting_func(xdata, *params)
        weighted_residuals = weights * residuals**2
        chi_squared = np.sum(weighted_residuals)
        
        # Compute gradients (partial derivatives with respect to parameters)
        gradients = np.zeros(p)
        for i in range(p):
            # Perturb parameter i slightly
            delta = 1e-3
            params_perturbed = params.copy()
            params_perturbed[i] += delta
            residuals_perturbed = ydata - fitting_func(xdata, *params_perturbed)
            
            # Numerical gradient
            gradients[i] = -2 * np.sum(weights * residuals * (residuals_perturbed - residuals) / delta)
        
        # Update parameters using gradient descent
        params -= learning_rate * gradients
        
        # Check for convergence
        if np.linalg.norm(gradients) < tol:
            break
        else:
            raise ValueError("Error: Maximum iterations reached without convergence.")
    
    # Estimate parameter uncertainties (approximation)
    J = np.zeros((n, p))  # Jacobian matrix
    for i in range(p):
        delta = 1e-5
        params_perturbed = params.copy()
        params_perturbed[i] += delta
        J[:, i] = (fitting_func(xdata, *params_perturbed) - fitting_func(xdata, *params)) / delta
    
    covariance_matrix = np.linalg.inv(J.T @ np.diag(weights) @ J)

    sigma_args = np.sqrt(np.diag(covariance_matrix))
    
    return params, sigma_args

def is_equidistant(data):
    """
    Check if the data points are equidistant.
    
    Parameters:
        data (array-like): Array of data points (e.g., x-coordinates).
    
    Returns:
        bool: True if data points are equidistant, False otherwise.
    """
    diffs = np.diff(data)
    return np.allclose(diffs, diffs[0])

def forward_fft(data):
    """
    Compute the Fast Fourier Transform (FFT) of the input data.
    
    Parameters:
        data (array-like): The input data for the FFT.
    
    Returns:
        array: The FFT of the input data.
    """
    return np.fft.fft(data)

def inverse_fft(data):
    """
    Compute the Inverse Fast Fourier Transform (IFFT) of the input data.
    
    Parameters:
        data (array-like): The input data for the IFFT.
    
    Returns:
        array: The IFFT of the input data.
    """
    return np.fft.ifft(data)

def fft_with_check(data, x_coords=None):
    """
    Compute the FFT, but first check if the data points are equidistant.
    
    Parameters:
        data (array-like): The input data for the FFT.
        x_coords (array-like, optional): The x-coordinates of the data points.
                                        If None, equidistant spacing is assumed.
    
    Returns:
        array: The FFT of the input data.
    
    Raises:
        ValueError: If the data points are not equidistant.
    """
    if x_coords is not None and not is_equidistant(x_coords):
        raise ValueError("FFT requires equidistant data points.")
    return forward_fft(data)

def ifft_with_check(data, x_coords=None):
    """
    Compute the IFFT, but first check if the data points are equidistant.
    
    Parameters:
        data (array-like): The input data for the IFFT.
        x_coords (array-like, optional): The x-coordinates of the data points.
                                         If None, equidistant spacing is assumed.
    
    Returns:
        array: The IFFT of the input data.
    
    Raises:
        ValueError: If the data points are not equidistant.
    """
    if x_coords is not None and not is_equidistant(x_coords):
        raise ValueError("IFFT requires equidistant data points.")
    return inverse_fft(data)

#TEMPORARY
def curvefit_wrapper(fitting_func, xdata, ydata, p0=None, yunc=None, bounds=(-np.inf, np.inf), return_cov=False):
    """
    A wrapper for scipy.optimize.curve_fit that adds uncertainty handling.

    Parameters:
    - fitting_func (callable): The model function, f(x, *params), to fit.
    - xdata (array-like): The independent variable data.
    - ydata (array-like): The dependent variable data.
    - p0 (array-like, optional): Initial guess for the parameters.
    - yunc (array-like, optional): Standard deviation of ydata (used as weights in the fit).
    - bounds (2-tuple of array-like, optional): Bounds on the parameters (default: no bounds).
    - return_cov (bool, optional): If True, return the covariance matrix along with fit parameters (default: False).

    Returns:
    - popt (array): Optimized parameters.
    - (optional) pcov (2D array): Covariance matrix of the parameters.
    """
    # Ensure data is numpy array
    xdata = np.array(xdata)
    ydata = np.array(ydata)

    if yunc is not None:
        yunc = np.array(yunc)

    # Perform the curve fitting
    try:
        popt, pcov = curve_fit(fitting_func, xdata, ydata, p0=p0, sigma=yunc, bounds=bounds, absolute_sigma=True)
    except RuntimeError as e:
        print(f"Error: Curve fitting did not converge. {e}")
        return None

    if return_cov:
        return popt, pcov
    return popt