"""
This module provides functionality for processing data from CSV and Markdown files. 
It includes utilities for reading columns, extracting temperature information, 
performing nonlinear sine fitting, and computing Fourier Transforms (FFT) and inverse FFTs. 

Key functionalities:
1. Read and process data from CSV files.
2. Extract and convert temperature information from Markdown files.
3. Perform nonlinear sine fitting on data.
4. Compute and visualize FFT and inverse FFT of data.
"""
import csv
import re
import numpy as np

def read_first_two_columns(csv_file):
    """
    Reads the first two columns of a CSV file.

    Args:
        csv_file (str): Path to the CSV file.

    Returns:
        tuple: Two lists containing the first and second columns of the CSV file.
    """
    first_column = []
    second_column = []
    try:
        with open(csv_file, mode='r', newline='', encoding='utf-8') as file:
            reader = csv.reader(file)
            next(reader)
            for row in reader:
                if len(row) >= 2:
                    first_column.append(row[0])
                    second_column.append(row[1])
    except FileNotFoundError:
        print(f"Error: The file '{csv_file}' does not exist.")
    except IsADirectoryError:
        print(f"Error: Expected a file, but found a directory at '{csv_file}'.")
    except csv.Error as e:
        print(f"CSV error occurred: {e}")
    except PermissionError:
        print(f"Error: Permission denied to access the file '{csv_file}'.")
    except OSError as e:
        print(f"OS error occurred: {e}")
    return first_column, second_column

def extract_temperature_from_markdown(file_path):
    """
    Extracts temperature information from a Markdown file.

    Args:
        file_path (str): Path to the Markdown file.

    Returns:
        tuple: The temperature value and its unit ('°C', '°F', 'K'). 
               Returns (None, None) if no temperature is found or an error occurs.
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()
        match = re.search(r"Temperature:\s*([+-]?\d+(?:\.\d+)?)\s*(°C|°F|K)?", content)
        if match:
            temperature_value = float(match.group(1))
            unit = match.group(2) if match.group(2) else "C"
            return temperature_value, unit
        print("Temperature not found in the file.")
        return None, None
    except FileNotFoundError:
        print(f"Error: The file '{file_path}' does not exist.")
        return None, None
    except PermissionError:
        print(f"Error: Permission denied to access the file '{file_path}'.")
        return None, None
    except IOError as e:
        print(f"IO error occurred: {e}")
        return None, None

def fahrenheit_to_kelvin(fahrenheit):
    """
    Converts temperature from Fahrenheit to Kelvin.

    Args:
        fahrenheit (float): Temperature in Fahrenheit.

    Returns:
        float: Temperature in Kelvin.
    """
    kelvin = (fahrenheit - 32) * 5/9 + 273.15
    return kelvin

def curve_fit(func, xdata, ydata, p0, learning_rate=0.001):
    """
    Custom implementation of curve fitting using gradient descent.

    Parameters:
    - func: The model function, func(x, *params).
    - xdata: Array-like, the independent variable data.
    - ydata: Array-like, the dependent variable data.
    - p0: Initial guess for the parameters.
    - learning_rate: Learning rate for gradient descent (default 0.001).

    Returns:
    - popt: Optimal parameters for the model function.
    - pcov: Covariance matrix of the parameter estimates (simplified here).
    """
    p = np.array(p0, dtype=float)
    n_params = len(p)
    max_iter, tol = 1000, 1e-6

    for iteration in range(max_iter):
        residuals = ydata - func(xdata, *p)

        # Finite difference method for Jacobian with smaller perturbation
        perturb = 1e-6
        jacobian = np.array([
            (func(xdata, *(p + perturb * np.eye(n_params)[i])) - func(xdata, *p)) / perturb
            for i in range(n_params)
        ]).T

        gradient = -2 * np.dot(jacobian.T, residuals)
        p -= learning_rate * gradient

        # Dynamically adjust learning rate based on gradient norm and relative cost improvement
        if np.linalg.norm(gradient) < tol:
            print(f"Converged in {iteration} iterations.")
            break
        if np.linalg.norm(gradient) > 1e-3:
            learning_rate *= 0.8

    else:
        print("Maximum iterations reached without convergence.")

    try:
        pcov = np.linalg.inv(np.dot(jacobian.T, jacobian) + 1e-6 * np.eye(n_params))
    except np.linalg.LinAlgError:
        pcov = np.full((n_params, n_params), np.nan)

    return p, pcov

def nonlinear_sine_fit(csv_file, step_exponent=0):
    """
    Performs nonlinear sine fitting on data from a CSV file.

    Args:
        csv_file (str): Path to the CSV file.
        step_exponent (int): Exponent for downsampling the data (default: 0).

    Returns:
        tuple: Optimized parameters, covariance matrix, downsampled x_data, 
               and downsampled y_data. Returns (None, None, None, None) on failure.
    """
    x_data, y_data = read_first_two_columns(csv_file)
    try:
        x_data = np.array([float(x) for x in x_data])
        y_data = np.array([float(y) for y in y_data])
    except ValueError:
        print("Error: Columns contain non-numeric data.")
        return None, None, None, None

    step = 2 ** step_exponent
    x_data = x_data[::step]
    y_data = y_data[::step]

    def sine_wave(x, amplitude, frequency, phase_shift, offset):
        return amplitude * np.sin(frequency * x + phase_shift) + offset

    initial_guess = [1, 2 * np.pi / (x_data[-1] - x_data[0]), 0, np.mean(y_data)]
    try:
        # with full_output = False, always returns a 2-tuple
        # pylint: disable-next=unbalanced-tuple-unpacking
        popt, pcov = curve_fit(sine_wave, x_data, y_data, p0=initial_guess)
    except RuntimeError:
        print("Error: Curve fitting failed.")
        return None, None, None, None
    return popt, pcov, x_data, y_data

def compute_fft(y_data, x_data):
    """
    Computes the Fast Fourier Transform (FFT) of y_data.

    Args:
        y_data (ndarray): Data values to transform.
        x_data (ndarray): Corresponding x-axis values.

    Returns:
        tuple: Frequencies (scaled), magnitudes, and FFT result (complex numbers).
    """
    fft_result = np.fft.fft(y_data)
    fft_freq = np.fft.fftfreq(len(x_data), x_data[1] - x_data[0])
    fft_freq_scaled = fft_freq * 100
    fft_magnitude = np.abs(fft_result)
    return fft_freq_scaled, fft_magnitude, fft_result

def compute_inverse_fft(fft_result, filter_mask):
    """
    Computes the inverse FFT of filtered FFT data.

    Args:
        fft_result (ndarray): Original FFT result.
        filter_mask (ndarray): Boolean mask to apply as a filter.

    Returns:
        ndarray: Real part of the inverse FFT result after applying the filter.
    """
    filtered_fft = fft_result * filter_mask
    inverse_fft = np.fft.ifft(filtered_fft)
    return np.real(inverse_fft)
