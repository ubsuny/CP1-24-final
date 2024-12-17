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

def sine_fit(x, y, freq_grid=100):
    """
    Fit a sine wave to the given data with optimized frequency, amplitude, phase,
    and offset. Uses a grid search for frequency optimization.

    Args:
        x (np.array): Input x values (time or distance).
        y (np.array): Input y values.
        freq_grid (int): Number of frequency grid points to search over.

    Returns:
        dict: Optimized sine wave parameters:
              - 'amplitude': Amplitude of the sine wave
              - 'frequency': Optimized frequency
              - 'phase': Phase shift
              - 'offset': Vertical offset
    """
    params = {
        'freq_min': 1 / (2 * (x[-1] - x[0])),
        'freq_max': 5 / (2 * (x[-1] - x[0])),
        'best': {'error': np.inf, 'frequency': None, 'coeffs': []},
        'grid': []
    }

    def sine_basis(f, x):
        """Generate sine and cosine basis for a given frequency."""
        return np.vstack([
            np.sin(2 * np.pi * f * x),
            np.cos(2 * np.pi * f * x),
            np.ones_like(x)
        ]).T

    # Generate frequency grid
    params['grid'] = list(np.linspace(params['freq_min'], params['freq_max'], freq_grid))

    # Search for best frequency
    for f in params['grid']:
        x_upper = sine_basis(f, x)
        coeffs = np.linalg.lstsq(x_upper, y, rcond=None)[0]
        error = np.sum((y - x_upper @ coeffs) ** 2)
        if error < params['best']['error']:
            params['best']['error'] = error
            params['best']['frequency'] = f
            params['best']['coeffs'] = coeffs.tolist()

    # Extract optimized parameters
    coeffs = params['best']['coeffs'] if len(params['best']['coeffs']) == 3 else [0.0, 0.0, 0.0]
    a_sin, a_cos, offset = coeffs

    amplitude = np.sqrt(a_sin**2 + a_cos**2)
    phase = np.arctan2(a_cos, a_sin)

    return {
        'amplitude': amplitude,
        'frequency': params['best']['frequency'],
        'phase': phase,
        'offset': offset
    }




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
        'data': resample_to_2n_segments(data, n) if resample else data,
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
    Prepares the input data by rotating it to align with the x-axis, optionally resampling it,
    and fitting a sine curve using a nonlinear fitting method.

    Args:
        data (pd.DataFrame): Input DataFrame with at least three columns:
                             - Time (0th column)
                             - Latitude (°) (1st column)
                             - Longitude (°) (2nd column)
        n (int, optional): Determines the number of points to resample the data into as 2^n.
                           Default is 6, resulting in 64 points.
        resample (bool, optional): If True, resamples the data into 2^n equidistant segments
                                   before applying the rotation and fitting. Default is False.

    Returns:
        pd.DataFrame: A DataFrame with:
                      - Time in the 0th column
                      - Rotated x and y coordinates in the 1st and 2nd columns
                      - Nonlinearly fitted sine curve values in the 4th column.
    """
    refit_data = rotate_to_x_axis(data, resample, n, fit_curve=True)

    # return the data with fitted curve in the i=4 column
    return refit_data
