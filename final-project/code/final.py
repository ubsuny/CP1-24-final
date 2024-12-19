import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

def fahrenheit_to_kelvin(fahrenheit):
    """
    Convert Fahrenheit to Kelvin.

    Parameters:
        fahrenheit (float): Temperature in Fahrenheit.

    Returns:
        float: Temperature in Kelvin.
    """
    kelvin = (fahrenheit - 32) * 5/9 + 273.15
    return kelvin

def parse_temperature_from_markdown(filepath):
    """
    Parse the temperature value from a markdown file.

    Parameters:
        filepath (str): Path to the markdown file.

    Returns:
        float: Temperature in Fahrenheit (as a float).
        None: If no temperature is found in the file.
    """
    try:
        # Read the markdown file using pandas
        with open(filepath, 'r') as file:
            lines = file.readlines()
        
        # Look for a line containing "Temperature | XX°F"
        for line in lines:
            if "Temperature" in line and "°F" in line:
                # Extract the numeric value before °F
                temperature = float(line.split('|')[1].strip().replace('°F', ''))
                return temperature
        
        print("No temperature found in the markdown file.")
        return None
    except FileNotFoundError:
        print(f"File not found: {filepath}")
        return None

def list_markdown_files(directory, filename_filter):
    """
    List all markdown files in a directory that match a specific filename filter.

    Parameters:
        directory (str): Path to the directory containing files.
        filename_filter (str): A string filter to match filenames (e.g., "sinewalk").

    Returns:
        list: A list of markdown file paths that match the filter.
    """
    try:
        # Get all files in the directory
        all_files = os.listdir(directory)

        # Filter files by those ending with '.md' and containing the filename_filter
        markdown_files = [
            os.path.join(directory, file) for file in all_files
            if file.endswith('.md') and filename_filter in file
        ]
        return markdown_files

    except FileNotFoundError:
        print(f"Directory not found: {directory}")
        return []
    except Exception as e:
        print(f"An error occurred: {e}")
        return []

def sine_function(x, amplitude, frequency, phase, offset):
    """
    Sine function for fitting.

    Parameters:
        x (array-like): Independent variable (e.g., time).
        amplitude (float): Amplitude of the sine wave.
        frequency (float): Frequency of the sine wave.
        phase (float): Phase shift of the sine wave.
        offset (float): Vertical offset of the sine wave.

    Returns:
        array-like: Computed sine values for the input x.
    """
    return amplitude * np.sin(2 * np.pi * frequency * x + phase) + offset

def non_linear_fitting(x_data, y_data, fit_function, initial_params, step_power, learning_rate=0.01, max_iter=1000):
    """
    Perform non-linear fitting using gradient descent.

    Parameters:
        x_data (array-like): The independent variable.
        y_data (array-like): The dependent variable.
        fit_function (callable): The non-linear function to fit the data.
        initial_params (list): Initial guess for the parameters of the fit function.
        step_power (int): Specify step number as 2^n for data subsampling.
        learning_rate (float): Step size for gradient descent.
        max_iter (int): Maximum number of iterations for gradient descent.

    Returns:
        list: Optimized parameters for the fit function.
    """
    # Validate step_power
    num_steps = 2**step_power
    if num_steps > len(x_data):
        raise ValueError(f"Step size 2^{step_power} exceeds data length.")

    # Subsample the data based on the step size
    step_size = len(x_data) // num_steps
    x_sampled = x_data[::step_size]
    y_sampled = y_data[::step_size]

    # Convert to numpy arrays for calculations
    x_sampled = np.array(x_sampled)
    y_sampled = np.array(y_sampled)

    # Gradient descent optimization
    params = np.array(initial_params)
    for iteration in range(max_iter):
        # Calculate predictions and residuals
        y_pred = fit_function(x_sampled, *params)
        residuals = y_pred - y_sampled

        # Calculate the gradient for each parameter
        gradients = []
        for i in range(len(params)):
            # Partial derivative with respect to params[i]
            params_step = params.copy()
            params_step[i] += 1e-5  # Small change to compute numerical gradient
            y_pred_step = fit_function(x_sampled, *params_step)
            gradient = np.sum(2 * residuals * (y_pred_step - y_pred) / 1e-5)
            gradients.append(gradient)

        # Update parameters using gradients
        params -= learning_rate * np.array(gradients)

        # Check for convergence (if the update is very small)
        if np.linalg.norm(learning_rate * np.array(gradients)) < 1e-6:
            break

    return params
