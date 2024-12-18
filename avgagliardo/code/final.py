"""
final.py

This module implements the functionality needed to complete the algorithm tasks
for the PHY410 final project. This includes unit converters, file parsers, and
data handling methods.
"""
import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# from matplotlib import cm, colormaps

def convert_f_to_k(f):
    """
    Convert a temperature from Fahrenheit to Kelvin.

    Parameters:
        fahrenheit (float): The temperature in degrees Fahrenheit.
    Returns:
        float: The temperature converted to Kelvin.
    """
    return (f - 32) * 5 / 9 + 273.15

def parse_markdown(file_path, field_name):
    """
    Parse a markdown file and extract the value of a specified field.

    Parameters:
        file_path (str): Path to the markdown file.
        field_name (str): The name of the field to extract (e.g., 'Temperature').

    Returns:
        float: The extracted value as a float, or None if the field is not found.
    """
    with open(file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()  # Read all lines into a list
        extracted_value = None
        for i, line in enumerate(lines):
            if line.strip() == f"##{field_name}":
                # The next line should contain the value in the format '33 F'
                # for the Temperature data field
                if i + 1 < len(lines):
                    match = re.match(r"(-?\d+\.?\d*)\s*F", lines[i + 1].strip())
                    if match:
                        extracted_value = float(match.group(1))
    return extracted_value

def extract_number_from_filename(fname):
    """
    Obtain the trailing numeric value from a filename.

    Parameters:
        fname (str): The filename to process.

    Returns:
        int: The numeric value found before the file extension.
             Returns a high default value for invalid filenames.
    """
    try:
        # Split by underscore and get last part
        last_part = fname.rsplit('_', 1)[-1]
        # Remove file extension
        number_str = last_part.split('.', 1)[0]
        # Convert to integer
        return int(number_str)
    except (ValueError, TypeError):
        # Return a large default value for filenames without numeric suffixes
        return float('inf')

def sort_filenames(file_list):
    """
    Sorts a list of filenames based on the trailing numeric value before the file extension.

    Parameters
    ----------
    file_list : list of str
        The list of filenames to be sorted.

    Returns
    -------
    list of str
        The input filenames sorted based on the trailing numeric value before the extension.
    """

    return sorted(file_list, key=extract_number_from_filename)

def filter_markdown_files(directory_path, filter_string):
    """
    Sort and list all markdown filenames in a directory that contain the filter string.

    Parameters:
        directory_path (str): The path to the directory to scan.
        filter_string (str): The string to filter filenames.

    Returns:
        list: A list of sorted markdown filenames containing the filter string.
    """
    # check for the directoryj
    if not os.path.exists(directory_path):
        raise FileNotFoundError(f"The directory '{directory_path}' does not exist.")
    # complain if its missing
    if not os.path.isdir(directory_path):
        raise NotADirectoryError(f"'{directory_path}' is not a valid directory.")

    # List all files in the directory and filter markdown files with the filter string
    filtered_files = [
        filename for filename in os.listdir(directory_path)
        if filename.endswith(".md") and filter_string in filename
    ]
    # return the filenames to caller
    sorted_files = sort_filenames(filtered_files)


    return sorted_files

# task 4
def degrees_to_meters(lat, lon):
    """
    Convert latitude and longitude in degrees to meters.
    Scales longitude based on latitude for local, small-scale distances.

    Args:
        lat (np.array): Latitude in degrees.
        lon (np.array): Longitude in degrees.

    Returns:
        tuple: Relative (x, y) coordinates in meters.
    """
    meters_per_degree_lat = 111320  # Meters per degree of latitude
    mean_lat = np.radians(lat[0])  # Scale longitude using the first latitude
    meters_per_degree_lon = meters_per_degree_lat * np.cos(mean_lat)

    lat_rel = lat - lat[0]
    lon_rel = lon - lon[0]

    x = lon_rel * meters_per_degree_lon
    y = lat_rel * meters_per_degree_lat
    return x, y

def resample_to_2n_segments(data: pd.DataFrame, n: int) -> pd.DataFrame:
    """
    Resample the dataset into 2^n equidistant time-based segments using interpolation.

    Args:
        data (pd.DataFrame): Original DataFrame with time in the first column, 'Latitude (°)',
        and 'Longitude (°)'.
        n (int): Determines the number of segments as 2^n.

    Returns:
        pd.DataFrame: Resampled data with the same structure.
    """
    num_points = 2**n  # Number of equidistant segments
    original_time = data.iloc[:, 0].to_numpy()  # Use the time column (0th column)
    new_time = np.linspace(original_time.min(), original_time.max(), num_points)

    # Linear interpolation for latitude and longitude
    lat_interp = np.interp(new_time, original_time, data['Latitude (°)'].to_numpy())
    lon_interp = np.interp(new_time, original_time, data['Longitude (°)'].to_numpy())

    # Create resampled data
    resampled_data = pd.DataFrame({
        data.columns[0]: new_time,  # Resampled time column
        'Latitude (°)': lat_interp,
        'Longitude (°)': lon_interp
    })

    return resampled_data

def sine_fit(x, y):
    """
    Fit a sine wave to the given data with optimized frequency, amplitude, phase, and offset.

    Args:
        x (np.array): Input x values (time or distance).
        y (np.array): Input y values.

    Returns:
        dict: Fitted sine wave parameters: amplitude, frequency, phase, and offset.
    """
    # Remove the mean to simplify fitting
    y_mean = np.mean(y)
    y_centered = y - y_mean

    # Approximate frequency: Assume signal completes 2 oscillations
    t = x[-1] - x[0]
    freq_estimate = 2 / t

    # Generate sine and cosine components at the estimated frequency
    sine = np.sin(2 * np.pi * freq_estimate * x)
    cosine = np.cos(2 * np.pi * freq_estimate * x)

    # Solve for coefficients using least squares
    a = np.column_stack((sine, cosine))
    coeffs, _, _, _ = np.linalg.lstsq(a, y_centered, rcond=None)
    a_sin, a_cos = coeffs

    # Calculate amplitude and phase
    amplitude = np.sqrt(a_sin**2 + a_cos**2)
    phase = np.arctan2(a_cos, a_sin)

    # Return results
    return {
        'amplitude': amplitude,
        'frequency': freq_estimate,
        'phase': phase % (2 * np.pi),
        'offset': y_mean
    }

def check_equidistant_x(x_values: np.ndarray, tolerance: float = 1e-3) -> bool:
    """
    Check if the x values are equidistant within a specified tolerance.

    Args:
    x_values (np.ndarray): Array of x values.
    tolerance (float): Allowable deviation for equidistant checks.

    Returns:
    bool: True if x values are equidistant, False otherwise.
    """
    diffs = np.diff(x_values)  # Compute differences between consecutive x values
    return np.all(np.abs(diffs - diffs[0]) <= tolerance)


def rotate_to_x_axis(data: pd.DataFrame,
        resample: bool = False,
        n: int = 6,
        fit_curve: bool = False):
    """
    Rotate GPS coordinates so the path aligns with the x-axis and center it at the origin.
    Optionally resample the data into 2^n equidistant segments and fit a sine wave.

    Args:
        data (pd.DataFrame): DataFrame with time in the first column, 'Latitude (°)',
        and 'Longitude (°)' columns.
        resample (bool): If True, resample the data before rotation.
        n (int): Determines the number of segments as 2^n if resampling.
        fit_curve (bool): If True, fit a sine wave to the curve.

    Returns:
        pd.DataFrame: Rotated coordinates with columns ['Time', 'x', 'y'].
    """
    results = {
        'data': resample_df_to_2n_segments(data, n) if resample else data,
        'coords': {},
        'rotation': {},
        'fit': {}
    }

    # Convert to meters
    lat = results['data']['Latitude (°)'].to_numpy()
    lon = results['data']['Longitude (°)'].to_numpy()
    results['coords']['x'], results['coords']['y'] = degrees_to_meters(lat, lon)

    # Center coordinates at the origin
    coords_array = np.vstack((results['coords']['x'], results['coords']['y'])).T
    results['coords']['centered'] = coords_array - coords_array[0]

    # PCA for rotation
    covariance_matrix = np.cov(results['coords']['centered'].T)
    eigenvalues, eigenvectors = np.linalg.eig(covariance_matrix)
    results['rotation']['principal'] = eigenvectors[:, np.argmax(eigenvalues)]

    # Rotation matrix
    angle = np.arctan2(results['rotation']['principal'][1], results['rotation']['principal'][0])
    results['rotation']['matrix'] = np.array([
        [np.cos(-angle), -np.sin(-angle)],
        [np.sin(-angle), np.cos(-angle)]
    ])

    # Apply rotation
    results['rotation']['rotated'] = results['coords']['centered'] @ results['rotation']['matrix'].T
    rotated_df = pd.DataFrame(results['rotation']['rotated'], columns=['x', 'y'])
    rotated_df.insert(0, 'Time', results['data'].iloc[:, 0].to_numpy())

    # Fit sine wave if specified
    if fit_curve:
        results['fit'] = sine_fit(rotated_df['x'].to_numpy(), rotated_df['y'].to_numpy())
        rotated_df['Fitted Sine'] = (
            results['fit']['amplitude'] *
            np.sin(2 * np.pi * results['fit']['frequency'] *
            rotated_df['x'] + results['fit']['phase'])
            + results['fit']['offset']
        )

    return rotated_df

def prepare_and_fit(data, n=6, resample=False):
    """
    Prepares the input data by rotating it to align with the x-axis, optionally resampling
    it, and fitting a sine curve using a nonlinear fitting method.

    Args:
        data (pd.DataFrame): Input DataFrame with at least three columns:
                             - Time (0th column)
                             - Latitude (°) (1st column)
                             - Longitude (°) (2nd column)
        n (int, optional): Determines the number of points to resample the data into as 2^n.
                           Default is 6, resulting in 64 points.
        resample (bool, optional): If True, resamples the data into 2^n equidistant segments
            before applying the rotation and fitting. Default
            is False.

    Returns:
        pd.DataFrame: A DataFrame with:
                      - Time in the 0th column
                      - Rotated x and y coordinates in the 1st and 2nd columns
                      - Nonlinearly fitted sine curve values in the 4th column.
    """
    refit_data = rotate_to_x_axis(data, resample, n, fit_curve=True)

    # Check if x values are equidistant
    x_values = refit_data['x'].to_numpy()
    if not check_equidistant_x(x_values):
        print("Warning: x values are not equidistant!")
    else:
        print("x values are equidistant.")

    return refit_data


# task 5
def resample_df_to_2n_segments(data: pd.DataFrame, n: int = 6) -> pd.DataFrame:
    """
    Resample the dataset into 2^n equidistant segments based on the 'x' column using interpolation.

    Args:
        data (pd.DataFrame): Original DataFrame with 'Time', 'x', 'y', and other numeric columns.
                             The second column (index 1) is treated as the x-axis for interpolation.
        n (int): Determines the number of segments as 2^n.

    Returns:
        pd.DataFrame: Resampled data with the same structure as the original.
    """
    num_points = 2**n  # Number of equidistant segments
    original_x = data.iloc[:, 1].to_numpy()  # Use the 'x' column (index 1) for interpolation
    new_x = np.linspace(original_x.min(), original_x.max(), num_points)

    # Interpolate all other columns
    resampled_columns = {data.columns[1]: new_x}  # Initialize with resampled 'x' column
    for col in data.columns:
        if col != data.columns[1]:  # Skip the 'x' column since it's already resampled
            resampled_columns[col] = np.interp(new_x, original_x, data[col].to_numpy())

    # Create resampled DataFrame
    resampled_data = pd.DataFrame(resampled_columns)

    # Ensure original column order is preserved
    return resampled_data[data.columns]

def calculate_frequencies(x: np.ndarray, n: int, scale: float = 100):
    """
    Calculate FFT frequencies in pure Python, assuming x values are equidistant.

    Args:
        x (np.ndarray): Array of x values.
        n (int): Number of data points.
        scale (float): Scaling factor to adjust units (default: 100 for 1/100 meters).

    Returns:
        list: Frequencies in the desired units.
    """
    # Check if x values are equidistant
    if not check_equidistant_x(x):
        raise ValueError("x values must be equidistant to calculate frequencies.")

    # Calculate spatial step size Δx (use the first interval since x is equidistant)
    dx = x[1] - x[0]

    # Calculate frequencies
    frequencies = []
    for k in range(n):
        freq = k / (n * dx)  # Base frequency calculation
        if k > n // 2:      # Handle negative frequencies for FFT symmetry
            freq -= 1 / dx
        frequencies.append(freq * scale)
    return frequencies

def apply_fft_to_arrays(x: np.ndarray, y: np.ndarray):
    """
    Apply FFT to a NumPy array of y values, normalize the FFT values, and return results.

    Args:
        x (np.ndarray): Array of equidistant x values.
        y (np.ndarray): Array of y values.

    Returns:
        tuple: (frequencies, fft_values) where frequencies are in units of 1/100 meters.
    """
    n = len(y)
    fft_values = np.fft.fft(y)  # Raw FFT
    fft_frequencies = calculate_frequencies(x, n)

    # Normalize FFT values
    normalized_fft_values = fft_values / n

    return fft_frequencies, normalized_fft_values

def apply_fft(data: pd.DataFrame):
    """
    Wrapper function to apply FFT to a DataFrame column.

    Args:
        data (pd.DataFrame): DataFrame with equidistant 'x' and 'y' values.

    Returns:
        tuple: (frequencies, fft_values) where frequencies are in units of 1/100 meters.
    """
    x = data['x'].to_numpy()
    y = data['y'].to_numpy()
    return apply_fft_to_arrays(x, y)

def apply_ifft(fft_values: np.ndarray, reverse_norm: bool = True ) -> np.ndarray:
    """
    Apply the Inverse FFT (IFFT) and scale the result to reconstruct the original signal.

    Args:
        fft_values (np.ndarray): Normalized FFT values.

    Returns:
        np.ndarray: Reconstructed signal in the original domain.
    """
    n = len(fft_values)
    reconstructed_y = None

    # undo normalization by default
    if reverse_norm is True:
        reconstructed_y = np.fft.ifft(fft_values * n)   # Reverse normalization

    return np.real(reconstructed_y)  # real parts only


def plot_fft(frequencies: np.ndarray, fft_values: np.ndarray,
             title: str = "FFT Spectrum", threshold: float = 0.01):
    """
    Plot the FFT spectrum showing the normalized amplitude vs frequency with intercepts
    for the first and second harmonic peaks. Also plot the average between these peaks.

    Args:
        frequencies (np.ndarray): Array of frequency values from FFT.
        fft_values (np.ndarray): Complex FFT values.
        title (str): Title for the plot.
        threshold (float): Threshold for normalized amplitude to determine insignificance.
    """
    # Create a dictionary to store data
    fft_data = {}

    # Take the positive half of the FFT
    n = len(frequencies)
    fft_data['positive_freqs'] = frequencies[:n // 2]
    fft_data['positive_amplitudes'] = np.abs(fft_values)[:n // 2]  # Magnitude of FFT

    # Normalize amplitudes
    fft_data['normalized_amplitudes'] = (fft_data['positive_amplitudes'] /
        fft_data['positive_amplitudes'].max())

    # Apply threshold to cut off insignificant parts
    significant_indices = fft_data['normalized_amplitudes'] >= threshold
    fft_data['significant_freqs'] = fft_data['positive_freqs'][significant_indices]
    fft_data['significant_amplitudes'] = fft_data['normalized_amplitudes'][significant_indices]

    # Find the first harmonic (maximum amplitude)
    max_index = np.argmax(fft_data['significant_amplitudes'])
    fft_data['max_frequency'] = fft_data['significant_freqs'][max_index]
    fft_data['max_amplitude'] = fft_data['significant_amplitudes'][max_index]

    # Find the second harmonic (second largest peak)
    sorted_indices = np.argsort(fft_data['significant_amplitudes'])[::-1]
    second_max_index = sorted_indices[1]
    fft_data['second_max_frequency'] = fft_data['significant_freqs'][second_max_index]
    fft_data['second_max_amplitude'] = fft_data['significant_amplitudes'][second_max_index]

    # Calculate average frequency between first and second harmonics
    fft_data['average_frequency'] = (fft_data['max_frequency'] +
        fft_data['second_max_frequency']) / 2

    # Plot the FFT spectrum
    plt.figure(figsize=(10, 6))
    plt.plot(fft_data['significant_freqs'], fft_data['significant_amplitudes'],
             label="Normalized Amplitude Spectrum", color="b")

    # First harmonic intercepts
    plt.axvline(x=fft_data['max_frequency'], color='g', linestyle='--',
                label=f"1st Harmonic: {fft_data['max_frequency']:.3f}")
    plt.axhline(y=fft_data['max_amplitude'], color='g', linestyle='--')
    plt.scatter(fft_data['max_frequency'],
        fft_data['max_amplitude'],
        color='green', zorder=5)
    plt.text(fft_data['max_frequency'], fft_data['max_amplitude'] - 0.05,
             f"1st: {fft_data['max_frequency']:.3f}", color='green', fontsize=12)

    # Second harmonic intercepts
    plt.axvline(x=fft_data['second_max_frequency'], color='r', linestyle='--',
                label=f"2nd Harmonic: {fft_data['second_max_frequency']:.3f}")
    plt.axhline(y=fft_data['second_max_amplitude'], color='r', linestyle='--')
    plt.scatter(fft_data['second_max_frequency'],
        fft_data['second_max_amplitude'],
        color='red', zorder=5)
    plt.text(fft_data['second_max_frequency'], fft_data['second_max_amplitude'] - 0.05,
             f"2nd: {fft_data['second_max_frequency']:.3f}", color='red', fontsize=12)

    # Average frequency intercept
    plt.axvline(x=fft_data['average_frequency'], color='purple', linestyle='--',
                label=f"Avg: {fft_data['average_frequency']:.3f}")
    plt.scatter(fft_data['average_frequency'], 0, color='purple', zorder=5)
    plt.text(fft_data['average_frequency'], 0.05, f"Avg: {fft_data['average_frequency']:.3f}",
             color='purple', fontsize=12)

    # Labels and title
    plt.title(title)
    plt.xlabel("Frequency (1/100 meters)")
    plt.ylabel("Normalized Amplitude")
    plt.grid()
    plt.legend()
    plt.show()
