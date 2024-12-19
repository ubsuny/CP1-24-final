'''This module contains all of the functions necessary to complete 
the Final Exam for Computational Physics 1, 2024.

My curvefit function that I could NOT get to work:

def curvefit(fitting_func, param_guess, xdata, ydata, yunc, learning_rate=1e-4,
             tol=1e-6, max_iter=10000):
    
    Perform nonlinear curve fitting for input data with uncertainties 
    and a fitting function.

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
        # Compute gradients (partial derivatives with respect to parameters)
        gradients = np.zeros(p)
        for i in range(p):
            # Perturb parameter i slightly
            delta = 1e-3
            params_perturbed = params.copy()
            params_perturbed[i] += delta
            residuals_perturbed = ydata - fitting_func(xdata, *params_perturbed)
            # Numerical gradient
            gradients[i] = -2 * np.sum(weights * residuals *
                                       (residuals_perturbed - residuals) / delta)
        # Update parameters using gradient descent
        params -= learning_rate * gradients
        # Check for convergence
        if np.linalg.norm(gradients) < tol:
            break
        raise ValueError("Error: Maximum iterations reached without convergence.")
    # Estimate parameter uncertainties (approximation)
    jacobian = np.zeros((n, p))  # Jacobian matrix
    for i in range(p):
        delta = 1e-5
        params_perturbed = params.copy()
        params_perturbed[i] += delta
        jacobian[:, i] = (fitting_func(xdata, *params_perturbed) -
                           fitting_func(xdata, *params)) / delta
    covariance_matrix = np.linalg.inv(jacobian.T @ np.diag(weights) @ jacobian)
    sigma_args = np.sqrt(np.diag(covariance_matrix))
    return params, sigma_args


    The scipy wrapper I can't use because of pytests and linting

    
    def sinefit_wrapper(xdata, ydata, p0=None, yunc=None, return_cov=False):
    """
    A wrapper for scipy.optimize.curve_fit that adds uncertainty handling.

    Parameters:
    - xdata (array-like): The independent variable data.
    - ydata (array-like): The dependent variable data.
    - p0 (array-like, optional): Initial guess for the parameters.
    - yunc (array-like, optional): Standard deviation of ydata (used as weights in the fit).
    - bounds (2-tuple of array-like, optional): Bounds on the parameters (default: no bounds).
    - return_cov (bool, optional): If True, return 
                                    the covariance matrix along with 
                                    fit parameters (default: False).

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
        popt, pcov = [scipy.optimize.curve_fit(sine, xdata, ydata, p0=p0,
                               sigma=yunc, bounds=(-np.inf, np.inf), absolute_sigma=True)[0],
                                scipy.optimize.curve_fit(sine, xdata, ydata, p0=p0,
                               sigma=yunc, bounds=(-np.inf, np.inf), absolute_sigma=True)[1]]
    except RuntimeError as e:
        print(f"Error: Curve fitting did not converge. {e}")
        return None

    if return_cov:
        return popt, pcov
    return popt
    '''

import os
import numpy as np

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
    data = np.loadtxt(path,
        skiprows=5,
        delimiter = ',',
        unpack=True)
    temp = data

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
    for root, _, files in os.walk(directory):
        for file in files:
            if search_string in file and file.endswith(file_extension):
                matching_files.append(os.path.join(root, file))
    return matching_files

def curvefitlin(xdata, ydata, xunc, yunc):
    '''
    Does nonlinear curve fitting for input data with uncertainty and function to be fit

    Parameters:
        -xdata: x-data to be fit
        -ydata: y-data to be fit
        -xunc: uncertainty in x-data
        -yunc: uncertainty in y-data 

    Returns:
        -The fitting parameters of func
    '''

    if len(xdata)<2:
        print('Error: Need more data!')

    if abs(np.sum(1/np.sum(np.sqrt(xunc**2+yunc**2))**2)) < 0.00001 :
        print('Error: Denominator too small!')

    sigma_x = np.sum(xdata/xunc)
    sigma_y = np.sum(ydata/yunc)

    t = (xdata - sigma_x/np.sum(1/np.sum(np.sqrt(xunc**2+
                                                 yunc**2))**2))/np.sum(np.sqrt(xunc**2+yunc**2))
    sigma_tt = np.sum(t**2)

    if abs(sigma_tt) < 0.00001:
        print('Error: Denominator S too small!')

    b = np.sum(t*ydata/np.sum(np.sqrt(xunc**2+yunc**2)))/sigma_tt
    a = (sigma_y - sigma_x*b)/np.sum(1/np.sum(np.sqrt(xunc**2+yunc**2))**2)
    sigma_a2 = (1+sigma_x**2/(np.sum(1/np.sum(np.sqrt(xunc**2+yunc**2))**2)
                              *sigma_tt))/np.sum(1/np.sum(np.sqrt(xunc**2+yunc**2))**2)
    sigma_b2 = 1/sigma_tt

    if sigma_a2 < 0 or sigma_b2 < 0:
        print('Error: Negative Square Root')
    sigma_a = np.sqrt(sigma_a2)
    sigma_b = np.sqrt(sigma_b2)

    chi_squared = np.sum(((ydata-a-b*xdata)/np.sum(np.sqrt(xunc**2+yunc**2)))**2)

    return a, b, sigma_a, sigma_b, chi_squared

def is_equidistant(data):
    """
    Check if the data points are equidistant.
    
    Parameters:
        data (array-like): Array of data points (e.g., x-coordinates).
    
    Returns:
        bool: True if data points are equidistant, False otherwise.
    """
    diffs = np.diff(data)
    if np.allclose(diffs, diffs[0]):
        return 'Your data is equidistant!'
    return 'Your data is not equidistant :('

def forward_fft(xdata, ydata):
    """
    Compute the Fast Fourier Transform (FFT) of the input data.
    
    Parameters:
        data (array-like): The input data for the FFT.
    
    Returns:
        array: The FFT of the input data.
    """
    spectrum = np.fft.fft(ydata)
    dx = xdata[1] - xdata[0]  # Sampling interval (check uniformity!)
    frequencies = np.fft.fftfreq(len(xdata), d=dx)
    scaled_frequencies = frequencies*100

    magnitude = np.abs(spectrum)

    return scaled_frequencies, magnitude, spectrum

def inverse_fft(spectrum):
    """
    Compute the Inverse Fast Fourier Transform (IFFT) of the input spectrum.

    Parameters:
        frequencies (array-like): The frequency data (e.g., output from FFT).
        spectrum (array-like): The FFT spectrum (e.g., output from FFT).
    
    Returns:
        tuple: Reconstructed x-axis data and reconstructed y-axis data.
    """
    #Compute inverse fast fourier transform
    og_data = np.fft.ifft(spectrum)

    return np.real(og_data)

#TEMPORARY
def sine(x, a, b, c, d):
    '''A sinusoidal function for the fitting of our data
    Parameters:
    -x: independent variable
    -a: amplitude
    -b: frequency
    -c: phase
    -d: vertical offset

    retruns:
    a*sin(bx+c)+d
    '''
    return a*np.sin(b*x+c) + d

##THE FIXED PART IS A LIE
def curve_fit_fixed(func, xdata, ydata, params):
    """
    A functn doing optimized curve fitting that returns parameters with uncertainties 

    Parameters:
    -fun: Function to be fit to data
    -xdata: independent variable data
    -ydata: dependent variable data
    -guess: best guess at parameters

    Returns:
    -params: optimized parameters
    -pcov: cavariance matrix from which we can get uncertainties by taking the trace
    """
    #Set the change in parameters for fitting and max number of iterations and tolerance for success
    param_change = 0.001
    max_iter = 1000
    tol = 1e-6

    prev_loss = float('inf')
    for _ in range(max_iter):
        grad = np.zeros_like(params)
        for i in range(len(params)):
            params_up = params.copy()
            params_down = params.copy()
            params_up[i] += 1e-8
            params_down[i] -= 1e-8

            grad[i] = ((ydata-func(xdata, *params_up))**2
                       - (ydata-func(xdata, *params_down)**2)) / (2 * 1e-8)

        # Update parameters
        params -= param_change * grad

        # Check convergence
        current_loss = np.sum((ydata - func(xdata, *params))**2)
        if np.abs(prev_loss - current_loss) < tol:
            break
        if current_loss > prev_loss:  # Safeguard against divergence
            param_change *= 0.5
        prev_loss = current_loss

    # Estimate covariance matrix (basic approximation)
    if len(ydata) - len(params) > 0:
        residual_var = np.sum((ydata - func(xdata, *params))**2) / (len(ydata) - len(params))
    else:
        residual_var = 0
    pcov = np.linalg.pinv(np.dot(grad[:, None], grad[None, :])) * residual_var

    return params, pcov
