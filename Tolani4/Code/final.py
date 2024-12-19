import os
import pandas as pd
import numpy as np
from scipy.optimize import curve_fit


# Convert Fahrenheit to Kelvin
def fahrenheit_to_kelvin(fahrenheit):
    """
    Converts temperature from Fahrenheit to Kelvin.

    Args:
    fahrenheit (float): Temperature in Fahrenheit.

    Returns:
    float: Temperature in Kelvin.
    """
    return (fahrenheit - 32) * 5.0 / 9.0 + 273.15


# Parse temperature from a markdown file
def parse_temperature(md_file):
    """
    Extracts the temperature from a markdown file.

    Args:
    md_file (str): Path to the markdown file.

    Returns:
    float: Temperature in Kelvin.
    """
    with open(md_file, 'r') as file:
        lines = file.readlines()
    
    # Find the line containing "Temperature ="
    for line in lines:
        if "Temperature =" in line:
            # Extract the temperature value and unit
            parts = line.split("=")
            temp_str = parts[1].strip()  # Get the value part, e.g., "70 F"
            temp_value, temp_unit = temp_str.split()  # Split into value and unit
            temp_value = float(temp_value)  # Convert temperature value to float
            
            if temp_unit.upper() == 'F':
                return fahrenheit_to_kelvin(temp_value)  # Convert to Kelvin if the unit is Fahrenheit
            else:
                raise ValueError(f"Unsupported temperature unit: {temp_unit}")
    
    raise ValueError("Temperature not found in markdown file.")


# List markdown files based on a filter string in the filename
def list_md_files(directory, filter_name="sinewalk"):
    """
    Generates a list of markdown files based on a filter string in the filename.

    Args:
    directory (str): Path to the directory.
    filter_name (str): Filter string to match files (default is 'sinewalk').

    Returns:
    list: List of markdown files matching the filter.
    """
    all_md_files = [f for f in os.listdir(directory) if f.endswith(".md")]
    filtered_files = [os.path.join(directory, f) for f in all_md_files if filter_name in f]
    return filtered_files


# Read CSV data and extract relevant columns
def read_csv_data(csv_file):
    """
    Reads CSV data and returns time, latitude, and longitude.

    Args:
    csv_file (str): Path to the CSV file.

    Returns:
    tuple: Time, latitude, and longitude as lists.
    """
    df = pd.read_csv(csv_file)
    time = df['Time (s)'].tolist()
    latitude = df['Latitude (°)'].tolist()
    longitude = df['Longitude (°)'].tolist()
    return time, latitude, longitude


# Sine wave function for fitting
def sine_wave(t, A, omega, phi, C):
    """
    Defines a sine wave.

    Args:
    t (array): Time values.
    A (float): Amplitude.
    omega (float): Angular frequency.
    phi (float): Phase shift.
    C (float): Vertical shift.

    Returns:
    array: Sine wave values at time t.
    """
    return A * np.sin(omega * t + phi) + C


# Fit sine wave to data
def fit_sine_wave(time, data, p0=None, step_number=2):
    """
    Fits a sine wave to the data.

    Args:
    time (array): Time values.
    data (array): Data values to fit.
    p0 (array, optional): Initial guess for the parameters [A, omega, phi, C]. 
                          If None, it is calculated from the data.
    step_number (int): Step number for sine wave fitting (default is 2).

    Returns:
    tuple: Fitted parameters (Amplitude, Angular Frequency, Phase, Vertical Shift).
    """
    # Use provided p0 or calculate the default initial guess
    if p0 is None:
        guess_amplitude = np.max(data) - np.min(data)  # Peak-to-peak amplitude
        guess_frequency = step_number / (time[-1] - time[0])  # Approximate frequency based on time span
        guess_phase = 0  # Phase guess
        guess_offset = np.mean(data)  # Offset as mean value of data
        p0 = [guess_amplitude, guess_frequency, guess_phase, guess_offset]

    # Fit the sine wave to the data
    popt, _ = curve_fit(sine_wave, time, data, p0=p0, maxfev=5000)  # Added maxfev=5000
    return popt


# Compute FFT of signal and check for equidistant data
def compute_fft(signal, time):
    # Check if time data is equidistant
    time_diff = np.diff(time)
    if not np.allclose(time_diff, time_diff[0]):
        raise ValueError("Time data is not equidistant. Interpolation will be applied.")
    
    # Perform FFT on the signal
    N = len(signal)
    dt = time[1] - time[0]
    fft_vals = np.fft.fft(signal)
    freqs = np.fft.fftfreq(N, dt)
    
    # Return only the positive frequencies (real signals are symmetric)
    positive_freqs = freqs[:N//2]
    positive_fft_vals = fft_vals[:N//2]
    
    return positive_freqs, positive_fft_vals



# Calculate the frequency axis in useful units (1/100m)
def calculate_frequency_axis(displacement, time):
    # Calculate frequency axis in Hz (1/seconds) using pure Python
    N = len(displacement)
    dt = time[1] - time[0]  # Time step
    
    # Calculate the frequency axis using FFT-like behavior without numpy
    freqs = [i / (N * dt) for i in range(N // 2)]
    
    # Convert the frequencies into 1/100 meters
    # Frequency spacing: Should be the inverse of the time step, scaled by 1/100 meters
    freq_100m = [f * (1 / 100) for f in freqs]  # This converts frequency to 1/100 meters
    
    return freq_100m