'''This module contains all of the functions necessary to complete 
the Final Exam for Computational Physics 1, 2024.'''

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
    data = np.loadtxt(path, unpack=True)
    temp = data[0]

    return temp


def list_files_with_string(directory, search_string, file_extension):
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
    
    covariance_matrix = J.T @ np.diag(weights) @ J

    sigma_args = np.sqrt(np.diag(covariance_matrix))
    
    return params, sigma_args

class FFTWrapper:
    """
    A wrapper class for performing FFT and IFFT operations with checks for equidistant data.
    """

    @staticmethod
    def is_equidistant(x):
        """
        Check if the data points in `x` are equidistant.

        Parameters:
            x (array-like): The input array of x-values.

        Returns:
            bool: True if equidistant, False otherwise.
        """
        diffs = np.diff(x)
        return np.allclose(diffs, diffs[0])

    @staticmethod
    def fft(y, x=None):
        """
        Perform the Fast Fourier Transform (FFT).

        Parameters:
            y (array-like): The values to transform.
            x (array-like, optional): The corresponding x-values. If provided, checks for equidistance.

        Returns:
            tuple: (frequencies, fft_result)

        Raises:
            ValueError: If x is provided and the data is not equidistant.
        """
        y = np.array(y)
        
        if x is not None:
            x = np.array(x)
            if not FFTWrapper.is_equidistant(x):
                raise ValueError("x-values are not equidistant. FFT requires equidistant sampling.")

            dx = x[1] - x[0]  # Sampling interval
            n = len(x)
            freqs = np.fft.fftfreq(n, d=dx)
        else:
            n = len(y)
            freqs = np.fft.fftfreq(n)  # Assume unit sampling if x is not provided

        fft_result = np.fft.fft(y)
        return freqs, fft_result

    @staticmethod
    def ifft(fft_result):
        """
        Perform the Inverse Fast Fourier Transform (IFFT).

        Parameters:
            fft_result (array-like): The FFT result to invert.

        Returns:
            array: The inverse-transformed values.
        """
        return np.fft.ifft(fft_result)
