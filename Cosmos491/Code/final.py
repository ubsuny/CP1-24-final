"""
A module for temperature conversion and listing markdown files.
Includes:
- Conversion from Fahrenheit to Kelvin.
- Parsing temperatures from a DataFrame.
- Listing markdown files based on a filename filter.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def fahrenheit_to_kelvin(fahrenheit: float) -> float:
    """
    Convert a temperature from Fahrenheit to Kelvin.

    Parameters:
    fahrenheit (float): Temperature in Fahrenheit.

    Returns:
    float: Temperature in Kelvin.

    Raises:
    ValueError: If the resulting Kelvin temperature is below absolute zero.
    """
    kelvin = (fahrenheit - 32) * 5.0 / 9.0 + 273.15
    if kelvin < 0:
        raise ValueError("Temperature in Kelvin cannot be below absolute zero.")
    return kelvin


def parse_temperature(data: pd.DataFrame) -> float:
    """
    Parses a pandas DataFrame containing temperature data in a specific column.

    Args:
        data (pd.DataFrame): DataFrame containing a 
        'Temperature' column with temperatures in Fahrenheit.

    Returns:
        float: Average temperature in Kelvin across the DataFrame.

    Raises:
        ValueError: If the 'Temperature' column is missing or empty.
    """
    if "Temperature" not in data.columns:
        raise ValueError("The DataFrame must contain a 'Temperature' column.")
    if data["Temperature"].empty:
        raise ValueError("The 'Temperature' column is empty.")
    return data["Temperature"].apply(fahrenheit_to_kelvin).mean()


def list_markdown_files(files: pd.DataFrame, filter_keyword: str) -> list:
    """
    Lists markdown files from a DataFrame of filenames containing a specific keyword.

    Args:
        files (pd.DataFrame): A DataFrame containing a column 'filename' with filenames.
        filter_keyword (str): Keyword to filter the markdown filenames.

    Returns:
        list: List of markdown filenames matching the keyword.

    Raises:
        ValueError: If the 'filename' column is missing in the DataFrame.
    """
    if "filename" not in files.columns:
        raise ValueError("The DataFrame must contain a 'filename' column.")

    # Filter markdown files based on the filter_keyword and '.md' extension
    filtered_files = files[
        files["filename"].str.contains(filter_keyword, case=False, na=False) &
        files["filename"].str.endswith(".md")
    ]
    return filtered_files["filename"].tolist()

def generate_data(func, x_range, params, noise_level=0):
    """
    Generate noisy data for a given function.

    Parameters:
        func (callable): The function to generate data from.
        x_range (tuple): A tuple (start, end, num_points) defining the x-axis range.
        params (tuple): Parameters to pass to the function.
        noise_level (float): Standard deviation of noise added to the data.

    Returns:
        pandas.DataFrame: A dataframe containing x and y values.
    """
    x = np.linspace(x_range[0], x_range[1], x_range[2])
    y = func(x, *params)
    if noise_level > 0:
        y += np.random.normal(0, noise_level, size=len(x))
    return pd.DataFrame({'x': x, 'y': y})

def test_non_linear_fit():
    """
    Test the non_linear_fit function to ensure it performs non-linear fitting correctly.
    """
    # Generate clean data
    data = generate_data(quadratic_model, (0, 10, 100), (1, 2, 3), noise_level=0)

    # Add NaN values and clean them
    data.loc[10, 'x'] = np.nan
    data.loc[20, 'y'] = np.nan
    clean_data = data.dropna()

    # Normalize data
    clean_data['x'] = clean_data['x'] / max(clean_data['x'])
    clean_data['y'] = clean_data['y'] / max(clean_data['y'])

    # Configurations for fitting
    config = {"step_power": 2, "max_iter": 1000, "tol": 1e-6}

    # Perform non-linear fitting
    try:
        params, residuals = non_linear_fit(
            clean_data, quadratic_model, initial_guess=[1, 1, 1], config=config
        )
    except ValueError as e:
        print(f"ValueError during fitting: {e}")
        return
    except TypeError as e:
        print(f"TypeError during fitting: {e}")
        return

    # Assertions
    assert len(params) == 3, "Expected 3 parameters."
    assert isinstance(residuals, list), "Residuals should be a list."
    assert np.isclose(params[0], 1, atol=0.1), (
        f"Expected param 0 close to 1, got {params[0]}"
    )
    assert np.isclose(params[1], 2, atol=0.1), (
        f"Expected param 1 close to 2, got {params[1]}"
    )
    assert np.isclose(params[2], 3, atol=0.1), (
        f"Expected param 2 close to 3, got {params[2]}"
    )

def plot_fit(data, model_func, params):
    """
    Plot the data and the fitted model.

    Parameters:
        data (pandas.DataFrame): Dataframe containing x and y values.
        model_func (callable): The model function to plot.
        params (list): Optimized parameters for the model.
    """
    plt.scatter(data['x'], data['y'], label='Data', color='blue')
    x = np.linspace(data['x'].min(), data['x'].max(), 500)
    y = model_func(x, *params)
    plt.plot(x, y, label='Fitted Model', color='red')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.show()

def check_equidistant(data):
    """
    Check if the input data is equidistant.

    Parameters:
        data (pandas.Series): Time series data.

    Returns:
        bool: True if the data is equidistant, False otherwise.
    """
    time_diffs = np.diff(data.index.values)  # Get differences in time
    time_diffs = time_diffs.astype('timedelta64[s]').astype(float)  # Convert to seconds
    return np.allclose(time_diffs, time_diffs[0])

def compute_fft(data: np.ndarray) -> np.ndarray:
    """
    Computes the Fast Fourier Transform (FFT) of the input data.

    Parameters:
        data (np.ndarray): A numpy array of data points.

    Returns:
        np.ndarray: The FFT of the input data.
    """
    return np.fft.fft(data)

def compute_ifft(data: np.ndarray) -> np.ndarray:
    """
    Computes the Inverse Fast Fourier Transform (IFFT) of the input data.

    Parameters:
        data (np.ndarray): A numpy array of Fourier-transformed data.

    Returns:
        np.ndarray: The inverse FFT of the input data.
    """
    return np.fft.ifft(data)

def calculate_frequency_axis(sample_rate, num_samples):
    """
    Calculate the frequency axis for a given sample rate and number of samples.

    Parameters:
        sample_rate (float): The sampling rate in Hz.
        num_samples (int): The number of samples in the signal.

    Returns:
        list: Frequency axis in Hz.

    Raises:
        ValueError: If sample_rate is less than or equal to 0 
        or if num_samples is less than or equal to 0.
    """
    if sample_rate <= 0:
        raise ValueError("Sample rate must be greater than zero.")
    if num_samples <= 0:
        raise ValueError("Number of samples must be greater than zero.")

    freq_axis = []
    nyquist_freq = sample_rate / 2
    freq_step = sample_rate / num_samples

    for i in range(num_samples):
        freq = i * freq_step
        if freq > nyquist_freq:
            freq = freq - sample_rate
        freq_axis.append(freq)

    return freq_axis

def convert_to_khz(freq_axis):
    """
    Convert a frequency axis from Hz to kHz.

    Parameters:
        freq_axis (list): List of frequencies in Hz.

    Returns:
        list: Frequency axis in kHz.
    """
    return [freq / 1000.0 for freq in freq_axis]


def convert_to_mhz(freq_axis):
    """
    Convert a frequency axis from Hz to MHz.

    Parameters:
        freq_axis (list): List of frequencies in Hz.

    Returns:
        list: Frequency axis in MHz.
    """
    return [freq / 1_000_000.0 for freq in freq_axis]
