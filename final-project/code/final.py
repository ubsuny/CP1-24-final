"""
This module contains Signal Processing and Data Analysis code
for the final project of CP-1-24.

It includes functionality for:
- Temperature conversion (Fahrenheit to Kelvin).
- Parsing temperature values from markdown files.
- Listing markdown files matching a specific filter.
- Performing non-linear fitting for data analysis.
- Verifying equidistant data and performing FFT/inverse FFT.
- Calculating frequency axes for FFT analysis.

Dependencies:
- numpy
- pandas
- os
"""
import os
import numpy as np
import pandas as pd

def fahrenheit_to_kelvin(temp_fahrenheit):
    """
    Convert Fahrenheit to Kelvin.

    Parameters:
        temp_fahrenheit (float): Temperature in Fahrenheit.

    Returns:
        float: Temperature in Kelvin.
    """
    # Formula for Fahrenheit to Kelvin conversion
    return (temp_fahrenheit - 32) * 5 / 9 + 273.15

def parse_temperature_from_markdown(file_path):
    """
    Extract the temperature value from a markdown file.

    Parameters:
        file_path (str): Path to the markdown file.

    Returns:
        float: Extracted temperature in Fahrenheit, or None if not found.
    """
    try:
        # Open the markdown file in read mode
        with open(file_path, 'r', encoding='utf-8') as file:
            lines = file.readlines()

        # Look for the line containing temperature information
        for line in lines:
            if "Temperature" in line and "°F" in line:
                # Split the line into parts based on the delimiter '|'
                parts = line.split('|')
                if len(parts) > 2:
                    # Extract the temperature part, remove unwanted characters, and convert to float
                    temp_part = parts[2].strip().replace('°F', '').replace('Â', '')
                    return float(temp_part)
        # Return None if no temperature information is found
        return None
    except FileNotFoundError as err:
        print(f"File not found: {err}")
        return None
    except ValueError as err:
        print(f"Value error while parsing temperature: {err}")
        return None

def list_markdown_files(directory, filename_filter):
    """
    List markdown files in a directory based on a filename filter.

    Parameters:
        directory (str): Path to the directory.
        filename_filter (str): A substring to filter markdown files.

    Returns:
        list: Paths to markdown files that match the filter.
    """
    try:
        # Get all files in the specified directory
        all_files = os.listdir(directory)

        # Filter files that end with '.md' and contain the filename_filter
        return [
            os.path.join(directory, file)
            for file in all_files
            if file.endswith('.md') and filename_filter in file
        ]
    except FileNotFoundError as err:
        print(f"Directory not found: {err}")
        return []

def sine_function(data_x, amplitude, frequency, phase, offset):
    """
    A sine function for modeling and fitting data.

    Parameters:
        data_x (array-like): Independent variable (e.g., time or distance).
        amplitude (float): Amplitude of the sine wave.
        frequency (float): Frequency of the sine wave.
        phase (float): Phase shift of the sine wave.
        offset (float): Vertical offset of the sine wave.

    Returns:
        array-like: Sine values for the input data_x.
    """
    # Use the sine formula to generate sine wave values
    return amplitude * np.sin(2 * np.pi * frequency * data_x + phase) + offset

def non_linear_fitting(params):
    """
    Perform non-linear fitting using gradient descent.

    Parameters:
        params (dict): Dictionary containing:
            - x_data (array-like): Independent variable (e.g., time or distance).
            - y_data (array-like): Dependent variable (e.g., signal values).
            - fit_function (callable): Function to fit the data.
            - initial_params (list): Initial guess for function parameters.
            - step_power (int): Subsample factor as 2^step_power.
            - learning_rate (float): Step size for gradient descent.
            - max_iterations (int): Maximum iterations for gradient descent.

    Returns:
        list: Optimized parameters for the fit function.
    """
    x_data = params["x_data"]
    y_data = params["y_data"]
    fit_function = params["fit_function"]
    initial_params = params["initial_params"]
    step_power = params["step_power"]
    learning_rate = params.get("learning_rate", 0.01)
    max_iterations = params.get("max_iterations", 1000)

    # Validate if the step size is within allowable range
    num_steps = 2 ** step_power
    if num_steps > len(x_data):
        raise ValueError(f"Step size 2^{step_power} exceeds data length.")

    # Subsample the data based on the step size
    step_size = len(x_data) // num_steps
    x_sampled = x_data[::step_size]
    y_sampled = y_data[::step_size]

    # Initialize parameters for gradient descent
    params = np.array(initial_params)
    for _ in range(max_iterations):
        # Predict y values using the fit function
        y_pred = fit_function(x_sampled, *params)

        # Calculate residuals (differences between predicted and actual y values)
        residuals = y_pred - y_sampled

        # Compute gradients for all parameters using numerical approximation
        gradients = [
            np.sum(2 * residuals * (fit_function(
                x_sampled, *(params + np.eye(1, len(params), i)[0] * 1e-5)) - y_pred) / 1e-5)
            for i in range(len(params))
        ]

        # Update parameters using the gradients
        params -= learning_rate * np.array(gradients)

        # Convergence check: Stop if the gradient updates are very small
        if np.linalg.norm(learning_rate * np.array(gradients)) < 1e-6:
            break
    return params

def check_equidistant(data_points):
    """
    Verify if the points in the independent variable are equidistant.

    Parameters:
        data_points (array-like): Independent variable data.

    Returns:
        bool: True if data points are equidistant, False otherwise.
    """
    differences = np.diff(data_points)
    # Check if the differences between consecutive points are consistent
    return np.allclose(differences, differences[0], atol=1e-6)

def perform_fft(signal_data):
    """
    Compute FFT for signal data.

    Parameters:
        signal_data (array-like): Dependent variable data.

    Returns:
        array: FFT of the input signal.
    """
    return np.fft.fft(signal_data)

def perform_inverse_fft(frequency_data):
    """
    Compute inverse FFT from frequency data.

    Parameters:
        frequency_data (array-like): Frequency-domain data.

    Returns:
        array: Inverse FFT result.
    """
    return np.fft.ifft(frequency_data)

def fft_wrapper(file_path, column_x, column_y):
    """
    Wrapper to load data, check equidistance, and compute FFT.

    Parameters:
        file_path (str): Path to the CSV file.
        column_x (str): Independent variable column name.
        column_y (str): Dependent variable column name.

    Returns:
        dict: FFT and inverse FFT results or an error message.
    """
    # Load the data from a CSV file into a DataFrame
    data_frame = pd.read_csv(file_path)

    # Extract the independent and dependent variable data
    data_x = data_frame[column_x].values
    data_y = data_frame[column_y].values

    # Check if the independent variable is equidistant
    if not check_equidistant(data_x):
        return {"equidistant": False, "message": "Data isn't equidistant. FFT cannot be performed."}

    # Compute FFT and inverse FFT
    fft_result = perform_fft(data_y)
    inverse_fft_result = perform_inverse_fft(fft_result)
    return {"equidistant": True, "fft": fft_result, "inverse_fft": inverse_fft_result}

def calculate_frequency_axis(samples, rate):
    """
    Compute the frequency axis for FFT results.

    Parameters:
        samples (int): Number of samples in the signal.
        rate (float): Sampling rate in Hz.

    Returns:
        list: Frequency bins corresponding to FFT results.
    """
    # Calculate the frequency resolution
    resolution = rate / samples

    # Generate a list of frequencies for the FFT bins
    return [i * resolution for i in range(samples // 2 + 1)]
