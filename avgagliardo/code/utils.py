"""
This module contains utility functions that do not belong as first order
members of the Trip class.

This includes unit converters, plotting methods, and .csv handling functions.
"""
from datetime import datetime
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class CSVImportError(Exception):
    """
    Custom exception raised when a CSV file cannot be imported.
    Provides an error message with details about the failure.
    """
    def __init__(self, message):
        super().__init__(message)

def convert_to_meters(value, unit):
    """
    Converts a given value from a specified unit to meters. This function
    uses exact quantities for conversion factors exclusively.

    Args:
        value (number): The numerical value to be converted.
        unit (str): The unit of the value to convert from. Supported units
        include:
                    'km' (kilometers), 'cm' (centimeters), 'mm' (millimeters),
                    'inch' (inches), 'ft' (feet), 'yd' (yards), 'mile' (miles).

    Returns:
        Number: The equivalent value in meters (a float).

    Raises:
        ValueError: If the specified unit is not supported.
    """
    # Let's store the unit coverstion factors neatly within in a dict.
    conversion_factors = {
        'km': 1000.0,          # Kilometers to meters (exact quantity)
        'cm': 0.0100,          # Centimeters to meters (exact quantity)
        'mm': 0.0010,         # Millimeters to meters (exact quantity)
        'inch': 0.0254,      # Inches to meters (exact quantity)
        'in': 0.0254,      # Inches to meters (exact quantity)
        'ft': 0.3048,        # Feet to meters (exact quantity)
        'feet': 0.3048,        # Feet to meters (exact quantity)
        'yd': 0.9144,        # Yards to meters (exact quantity)
        'yard': 0.9144,        # Yards to meters (exact quantity)
        'mile': 1609.344,      # Miles to meters (exact quantity)
        'ml': 1609.344      # Miles to meters (exact quantity)
    }

    # sanitize plural units
    if unit.endswith('s'):  # if the unit is pluralized
        unit = unit[:-1]    # make it singular

    # make sure the unit is in our dict
    if unit not in conversion_factors:
        raise ValueError(f"Unknown unit: {unit}")
    # Convert the value using the conversion factor for the unit
    quantity = value * conversion_factors[unit]
    # and return it
    return quantity

def convert_to_feet(value, unit):
    """
    Converts a given value from a specified unit to feet. This function
    uses exact quantities for conversion factors where possible.

    Args:
        value (number): The numerical value to be converted.
        unit (str): The unit of the value to convert from. Supported units
        include:
                    'm' (meters), 'km' (kilometers), 'inch' (inches),
                    'yd' (yards), 'mile' (miles), 'cm' (centimeters), 'mm' (millimeters).

    Returns:
        Number: The equivalent value in feet (a float).

    Raises:
        ValueError: If the specified unit is not supported.
    """
    # sanitize units
    if unit.endswith('s'):  # if the unit is pluralized
        unit = unit[:-1]    # make it singular

    # Define conversion factors for various units to feet
    conversion_factors_to_feet = {
        'm': 1250/381,         # Exact: 1 m = 1250/381 feet
        'meter': 1250/381,     # Exact based on IYPA of 1959 (see references)
        'km': 1250000/381,     # Kilometers to feet (exact)
        'inch': 1/12,          # Inches to feet (exact, 1 foot = 12 inches)
        'in': 1/12,            # Inches to feet (exact)
        'cm': 1250/381000,     # Centimeters to feet (exact)
        'mm': 1250/3810000,    # Millimeters to feet (exact)
        'yd': 3.0,             # Yards to feet (exact)
        'yard': 3.0,           # Yards to feet (exact)
        'mile': 5280.0,        # Miles to feet (exact)
        'mi': 5280.0           # Miles to feet (exact)
    }

    # make sure the unit key exists
    if unit not in conversion_factors_to_feet:
        raise ValueError(f"Unknown unit: {unit}")
    # Convert the value using the conversion factor for the unit
    quantity = value * conversion_factors_to_feet[unit]
    # and return it
    return quantity

def import_csv(csv_path):
    """
    Imports data from a CSV file (that uses commas, not semicolons) into a
     pandas DataFrame. It is then returned to the caller.

    Args:
        csv_path (str): The path to the CSV file.

    Returns:
        DataFrame: The data from the CSV file as a pandas DataFrame.
    """
    # Attempt to read in the csv, or throw the proper exception
    try:
        # Read the CSV file into a pandas DataFrame
        df = pd.read_csv(csv_path)
        # Return it to the calling function
        return df

    # If the the file can't be found
    # Throw the exception, print the error, and return None
    except FileNotFoundError as e:
        print(f"Error: The file '{csv_path}' was not found.")
        print(f"Details: {str(e)}")
        return None
    # If the file is empty
    # Throw an exception, print the error, and return None
    except pd.errors.EmptyDataError as e:
        print(f"Error: The file '{csv_path}' is empty.")
        print(f"Details: {str(e)}")
        return None
    # If the file is malformed and cannot be parsed
    # Throw an exception, print the error, and return None
    except pd.errors.ParserError as e:
        print(f"Error: The file '{csv_path}' could not be parsed.")
        print(f"Details: {str(e)}")
        return None

def timecode_to_unix(time_str):
    """
    Converts a time string in the ddefault format 'YYYY-MM-DD HH:MM:SS.sss
    UTC±HH:MM' to Unix time.

    Args:
        time_str (str): A string representing the time with timezone.

    Returns:
        int: Unix timestamp corresponding to the given time.
    """
    # Parse the string into a datetime object with timezone info
    dt_with_tz = datetime.strptime(time_str, '%Y-%m-%d %H:%M:%S.%f %Z%z')

    # Convert to Unix timestamp
    unix_time = int(dt_with_tz.timestamp())

    return unix_time

def haversine_with_altitude(start_point, end_point):
    """
    Calculate the 3D great-circle distance between two points on the spherical
    Earth surface, including the Euclidean altitude difference.

    The output distance is in meters.

    Args:
        start_point (dict): A dictionary with 'lat', 'lon', and 'alt' keys
            representing the start point's
        latitude, longitude, and altitude.
        end_point (dict): A dictionary with 'lat', 'lon', and 'alt' keys
            representing the end point's latitude, longitude, and altitude.

    Returns:
        distance (float): The 3D distance between the two points in meters.
    """
    radius_earth = 6371.0  # Radius of Earth in kilometers

    # Convert degrees to radians
    lat1 = np.radians(start_point['lat'])
    lon1 = np.radians(start_point['lon'])
    lat2 = np.radians(end_point['lat'])
    lon2 = np.radians(end_point['lon'])

    # Differences in latitudes, longitudes, and altitudes
    dlat = lat2 - lat1  # Latitude difference in radians
    dlon = lon2 - lon1  # Longitude difference in radians
    dalt = end_point['alt'] - start_point['alt']  # Altitude difference in meters

    # Haversine formula for the distance over the Earth's surface (in kilometers)
    a = np.sin(dlat / 2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    surface_distance_km = radius_earth * c  # Distance in kilometers

    # Convert surface distance from km to meters
    surface_distance_m = surface_distance_km * 1000  # Convert to meters

    # 3D Euclidean distance, including altitude
    distance_3d = np.sqrt(surface_distance_m**2 + dalt**2)

    return distance_3d

def plot_gpstrip_segments(gps_trip):
    """
    Plots the segments of a GpsTrip object using longitude as x and latitude as y in a 2D plot.

    Args:
        gps_trip (GpsTrip): A GpsTrip object with segment data containing longitude and latitude.

    Returns:
        None: Displays the plot using Matplotlib.
    """
    if gps_trip.segments is None:
        print("Error: No segment data available to plot.")

    # get the trip segments
    segments = gps_trip.segments

    # Check if necessary columns exist in the segments DataFrame
    if not {'start_long', 'start_lat', 'end_long', 'end_lat'}.issubset(segments.columns):
        print("Error: Missing longitude or latitude data in segments.")

    plt.figure(figsize=(10, 6))

    # Plot each segment
    for index, row in segments.iterrows():
        plt.plot([row['start_long'], row['end_long']],
                 [row['start_lat'], row['end_lat']], 'bo-', label='Segment' if index == 0 else "")

    # Label the axes
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.title('GPS Trip Segments (Longitude vs Latitude)')

    # Add grid and show plot
    plt.grid(True)
    plt.legend()
    plt.show()

def plot_gpstrip_segments_with_color(gps_trip, save_path=None, title=None):
    """
    Plots the segments of a GpsTrip object using longitude as x and latitude as
    y in a 2D plot, with the color of each point corresponding to its relative
    time in the experiment.

    The colors transition from red (start) to purple (end) based on time, just
    like the visible light spectrum!

    Args:
        gps_trip (GpsTrip): A GpsTrip object with segment data containing
        longitude and latitude.

    Returns:
        None: Displays the plot using Matplotlib.
    """
    if gps_trip.segments is None:
        print("Error: No segment data available to plot.")

    segments = gps_trip.segments

    # Check if necessary columns exist in the segments DataFrame
    if not {
        'start_long',
        'start_lat',
        'end_long',
        'end_lat',
        'start_t',
        'stop_t'
    }.issubset(segments.columns):
        # columns are missing, report
        print("Error: Missing longitude, latitude, or time data in segments.")

    # Normalize time data to range from 0 to 1 for color mapping
    times = segments['start_t'].values
    norm = plt.Normalize(times.min(), times.max())  # Normalize the time range

    # Create the color map (from red to purple)
    cmap = "plasma"

    plt.figure(figsize=(10, 6))

    # Plot each segment with color corresponding to its start time
    longitudes = np.concatenate([segments['start_long'].values, segments['end_long'].values])
    latitudes = np.concatenate([segments['start_lat'].values, segments['end_lat'].values])

    # use the start times for coloring
    colors = np.concatenate([segments['start_t'].values, segments['start_t'].values])

    # Use scatter to plot points with colors based on the time
    sc = plt.scatter(longitudes, latitudes, c=colors, cmap=cmap, norm=norm, marker='o')

    # Create colorbar to indicate time progression
    cbar = plt.colorbar(sc)
    cbar.set_label('Time (s)')

    # Label the axes
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    if title is None:
        plt.title('GPS Trip Segments with Direction of Travel (Time Progression)')
    else:
        plt.title(title)
    # Add grid and show plot
    plt.grid(True)

    # save
    if save_path is not None:
    # sanitize plural units
        plt.savefig(save_path+gps_trip.experiment_name+".png")
    plt.show()

def plot_gps_2d_projection(
        gps_trip,
        ref_latitude=None,
        ref_longitude=None,
        save_path=None,
        title=None):
    """
    Projects GPS data (in WGS84) onto a 2D XY plane and plots it, coloring points
    based on relative time.

    Args:
        gps_trip (GpsTrip): A GpsTrip object with GPS data including 'Time (s)',
            'Latitude (°)', and 'Longitude (°)' columns.
        ref_latitude (float, optional): Reference latitude for projection.
            Defaults to the mean latitude in gps_trip.
        ref_longitude (float, optional): Reference longitude for projection.
            Defaults to the mean longitude in gps_trip.
        save_path (str, optional): If provided, saves the plot to this path.
        title (str, optional): Title for the plot.

    Returns:
        None: Displays the plot of the projected GPS coordinates.
    """
    # Dictionary to hold grouped values for the projection
    d = {
        'ref_lat': ref_latitude or gps_trip.data['Latitude (°)'].mean(),
        'ref_lon': ref_longitude or gps_trip.data['Longitude (°)'].mean(),
        'earth_radius': 6378137,  # WGS84 semi-major axis
        'lat_rad': np.radians(gps_trip.data['Latitude (°)']),
        'lon_rad': np.radians(gps_trip.data['Longitude (°)']),
    }
    d['ref_lat_rad'] = np.radians(d['ref_lat'])
    d['ref_lon_rad'] = np.radians(d['ref_lon'])

    # Apply equirectangular projection formula
    x = d['earth_radius'] * (d['lon_rad'] - d['ref_lon_rad']) * np.cos(d['ref_lat_rad'])
    y = d['earth_radius'] * (d['lat_rad'] - d['ref_lat_rad'])

    # Normalize time for color mapping
    d['times'] = gps_trip.data['Time (s)'].values
    d['norm'] = plt.Normalize(d['times'].min(), d['times'].max())

    # Plotting the projected points with time-based colors
    plt.figure(figsize=(10, 6))
    sc = plt.scatter(x, y, c=d['times'], cmap="plasma", norm=d['norm'], marker='o')
    cbar = plt.colorbar(sc)
    cbar.set_label('Time (s)')

    plt.xlabel('X (meters)')
    plt.ylabel('Y (meters)')
    plt.title(title or 'Equirectangular Projection of GPS Coordinates (Colored by Time)')
    plt.grid(True)

    # Save plot if path is provided
    if save_path is not None:
        plt.savefig(save_path + gps_trip.experiment_name + "_projection.png")
    plt.show()

def plot_acceltrip_acceleration_with_color(accel_trip, **kwargs):
    """
    Plots the specified acceleration component (x, y, z, or total) of an AccelTrip
    object over time, with color representing the direction and magnitude of
    acceleration. Red indicates higher positive acceleration, and purple indicates
    lower or negative acceleration.

    Args:
        accel_trip (AccelTrip): An AccelTrip object with segment data containing
        time and acceleration components.
        **kwargs:
            - save_path (str): Path to save the plot as an image (optional).
            - title (str): Title of the plot.
            - component (str): The acceleration component to plot ('x', 'y', 'z',
                or 'total'). Defaults to 'total'.
            - compression_factor (float): Factor to adjust the color spectrum.
                Defaults to 1.0.
            - step (int): Plot every Nth point. Defaults to 1.
    """
    # Set up argparse to handle keyword arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--save_path',
        type=str,
        default=None,
        help='Path to save the plot image.'
    )
    parser.add_argument(
        '--title',
        type=str,
        default=None,
        help='Title of the plot.'
    )
    parser.add_argument(
        '--component',
        type=str,
        default='total',
        choices=['x', 'y', 'z', 'total'],
        help="Acceleration component to plot ('x', 'y', 'z', or 'total')."
    )
    parser.add_argument(
        '--compression_factor',
        type=float,
        default=1.0,
        help="Factor for color spectrum adjustment."
    )
    parser.add_argument(
        '--step',
        type=int,
        default=1,
        help="Downsampling factor, plotting every Nth point."
    )
    parser.add_argument(
        '--connect_points',
        action='store_true',
        default=True,
        help="Whether to connect points with lines."
    )

    # hide data so linter stops complaining
    extra = {}

    args, _ = parser.parse_known_args()
    save_path = kwargs.get('save_path', args.save_path)
    extra['title'] = kwargs.get('title', args.title)
    extra['c'] = kwargs.get('component', args.component)
    extra['cf'] = kwargs.get('compression_factor', args.compression_factor)
    extra['s'] = kwargs.get('step', args.step)
    extra['cp'] = kwargs.get('connect_points', args.connect_points)

    if accel_trip.segments is None:
        print("Error: No segment data available to plot.")
        return

    segments = accel_trip.segments
    accel_column = ('accel_'+str(extra['c']) if
        extra['c'] in ['x', 'y', 'z'] else 'total_acceleration')

    if accel_column not in segments.columns:
        print(f"Error: Missing {extra['c']}-acceleration data in segments.")
        return

    # Grouping related data into a dictionary to reduce the number of local variables
    plot_data = {
        'times': segments['start_t'].values[::extra['s']],
        'acceleration': segments[accel_column].values[::extra['s']],
        'compressed_acceleration': None,
        'norm': None
    }

    # Apply compression to the acceleration values for color scaling
    plot_data['compressed_acceleration'] = (
        np.sign(plot_data['acceleration']) *
        (np.abs(plot_data['acceleration']) ** (1 / extra['cf']))
    )
    plot_data['norm'] = plt.Normalize(
        plot_data['compressed_acceleration'].min(),
        plot_data['compressed_acceleration'].max()
    )

    # Set up the plot
    plt.figure(figsize=(10, 6))

    # Optionally connect points with lines
    line = None
    if extra['cp']:
        line, = plt.plot(
            plot_data['times'],
            plot_data['acceleration'],
            color='gray',
            alpha=0.7,
            label=str(extra['c']).capitalize()+' Acceleration'  # Ensure the label is set
        )

    # Scatter plot with color based on compressed acceleration
    sc = plt.scatter(
        plot_data['times'],
        plot_data['acceleration'],
        c=plot_data['compressed_acceleration'],
        cmap="nipy_spectral",
        norm=plot_data['norm'],
        marker='o',
        label=str(extra['c']).capitalize()+' Data Points'  # Label for scatter plot
    )
    cbar = plt.colorbar(sc)
    cbar.set_label('Scaled '+str(extra['c']).capitalize()+' Acceleration (m/s²)')

    plt.xlabel('Time (s)')
    plt.ylabel(str(extra['c']).capitalize()+' Acceleration (m/s²)')

    # shorten the line
    extra['at0'] = ' Acceleration over Time for AccelTrip'
    extra['at1'] = extra['title'] or str(extra['c']).capitalize()+ extra['at0']
    plt.title(extra['at1'])
    plt.grid(True)

    # Manually create a legend with line and scatter labels if applicable
    handles, labels = [], []
    if extra['cp'] and line is not None:
        handles.append(line)
        labels.append(line.get_label())
    handles.append(sc)
    labels.append(sc.get_label())

    plt.legend(handles, labels)

    if save_path:
        n = save_path+accel_trip.experiment_name+'_'+extra['c']+'_accel_with_color.png'
        plt.savefig(n)
    plt.show()

def plot_acceltrip_velocity(accel_trip, component='z'):
    """
    Plots the specified velocity component (x, y, or z) of an AccelTrip object
    over time.

    Args:
        accel_trip (AccelTrip): An AccelTrip object with segment data containing
            time and velocity components.
        component (str): The velocity component to plot ('x', 'y', or 'z').
            Defaults to 'z'.

    Returns:
        None: Displays the plot using Matplotlib.
    """
    if accel_trip.segments is None:
        print("Error: No segment data available to plot.")
        return

    segments = accel_trip.segments

    # Map component to the appropriate column name
    velocity_column = f'velocity_{component}'
    if not {'start_t', velocity_column}.issubset(segments.columns):
        print(f"Error: Missing time or {component}-velocity data in segments.")
        return

    # Extract the time and chosen velocity columns
    times = segments['start_t']
    velocity = segments[velocity_column]

    # Create the plot
    plt.figure(figsize=(10, 6))
    plt.plot(times, velocity, 'bo-', label=f'{component.upper()}-Velocity')

    # Label the axes
    plt.xlabel('Time (s)')
    plt.ylabel(f'{component.upper()}-Velocity (m/s)')
    plt.title(f'{component.upper()}-Velocity over Time for AccelTrip')

    # Add grid and legend
    plt.grid(True)
    plt.legend()

    # Show the plot
    plt.show()

def plot_acceltrip_velocity_with_color(accel_trip, **kwargs):
    """
    Plots the specified velocity component (x, y, or z) of an AccelTrip object
    over time, with color representing the direction of acceleration. Red
    indicates positive acceleration, and purple indicates negative acceleration.

    Args:
        accel_trip (AccelTrip): An AccelTrip object with segment data containing
        time and velocity components.
        **kwargs:
            - save_path (str): Path to save the plot as an image (optional).
            - title (str): Title of the plot.
            - component (str): The velocity component to plot ('x', 'y', or
                'z'). Defaults to 'z'.
            - compression_factor (float): Factor to adjust the color spectrum.
                Defaults to 1.0.
            - step (int): Plot every Nth point. Defaults to 10.
    """
    # Set up argparse to handle keyword arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--save_path',
        type=str,
        default=None,
        help='Path to save the plot image.'
    )
    parser.add_argument(
        '--title',
        type=str,
        default=None,
        help='Title of the plot.'
    )
    parser.add_argument(
        '--component',
        type=str,
        default='z',
        choices=['x', 'y', 'z'],
        help="Velocity component to plot ('x', 'y', or 'z')."
    )
    parser.add_argument(
        '--compression_factor',
        type=float,
        default=1.0,
        help="Factor for color spectrum adjustment."
    )
    parser.add_argument(
        '--step',
        type=int,
        default=10,
        help="Downsampling factor, plotting every Nth point."
    )
    parser.add_argument(
        '--connect_points',
        action='store_true',
        help="Whether to connect points with lines."
    )
    args, _unknown = parser.parse_known_args()

    e = {}
    e['sp'] = kwargs.get('save_path', args.save_path)
    e['t'] = kwargs.get('title', args.title)
    component = kwargs.get('component', args.component)
    compression_factor = kwargs.get('compression_factor', args.compression_factor)
    step = kwargs.get('step', args.step)
    e['cp'] = kwargs.get('connect_points', args.connect_points)

    if accel_trip.segments is None:
        print("Error: No segment data available to plot.")
        return

    segments = accel_trip.segments
    velocity_column = f'velocity_{component}'
    if velocity_column not in segments.columns:
        print(f"Error: Missing {component}-velocity data in segments.")
        return

    # Using a dictionary to group related variables for plotting
    plot_data = {
        'times': segments['start_t'].values[::step],
        'velocity': segments[velocity_column].values[::step],
    }

    # Calculate acceleration and compressed acceleration, storing in the dictionary
    plot_data['acceleration'] = np.diff(plot_data['velocity'], prepend=plot_data['velocity'][0])
    plot_data['compressed_acceleration'] = (
        np.sign(plot_data['acceleration'])*
        (np.abs(plot_data['acceleration'])**(1 / compression_factor))
    )
    plot_data['norm'] = plt.Normalize(
        plot_data['compressed_acceleration'].min(),
        plot_data['compressed_acceleration'].max()
    )

    # Start plotting
    plt.figure(figsize=(10, 6))
    plt.plot(
        plot_data['times'],
        plot_data['velocity'],
        color='gray',
        alpha=0.7,
        label=f'{component.upper()}-Velocity'
    )

    # Optionally connect points with lines
    if e['cp']:
        plt.plot(
            plot_data['times'],
            plot_data['acceleration'],
            color='gray',
            alpha=0.7,
            label=f'{component.capitalize()} Acceleration'
        )

    # Scatter plot with color based on compressed acceleration
    sc = plt.scatter(
        plot_data['times'],
        plot_data['velocity'],
        c=plot_data['compressed_acceleration'],
        cmap="nipy_spectral",
        norm=plot_data['norm'],
        marker='o'
    )
    cbar = plt.colorbar(sc)
    cbar.set_label('Scaled Acceleration (m/s²)')
    plt.xlabel('Time (s)')
    plt.ylabel(f'{component.upper()}-Velocity (m/s)')
    plt.title(e['t'] or f'{component.upper()}-Velocity over Time for AccelTrip')
    plt.grid(True)
    plt.legend()

    if e['sp']:
        n = e['sp']+accel_trip.experiment_name+'_'+component+'_velocity_with_color.png'
        plt.savefig(n)
    plt.show()

def plot_3d_trajectory(accel_trip, title=None, save_path=None):
    """
    Plots the 3D trajectory of an AccelTrip object using cumulative sums of the
    velocity components to approximate position. The color of the trajectory
    represents the time progression. The axes are normalized so that the axis
    with the largest range determines the scale for all axes.

    Args:
        accel_trip (AccelTrip): An AccelTrip object with segment data containing
        velocity components.

    Returns:
        None: Displays the 3D plot using Matplotlib.
    """
    if accel_trip.segments is None:
        print("Error: No segment data available to plot.")
        return

    segments = accel_trip.segments

    if not {'start_t', 'velocity_x', 'velocity_y', 'velocity_z'}.issubset(segments.columns):
        print("Error: Missing velocity or time data in segments.")
        return

    # Extract necessary data into dictionaries for better organization
    data = {
        'times': segments['start_t'].values,
        'velocities': {
            'x': segments['velocity_x'].values,
            'y': segments['velocity_y'].values,
            'z': segments['velocity_z'].values
        }
    }

    # Estimate positions by integrating velocity over time
    positions = {
        axis: np.cumsum(data['velocities'][axis] * np.diff(data['times'], prepend=data['times'][0]))
        for axis in ['x', 'y', 'z']
    }

    # Find ranges of each axis
    ranges = {
        axis: positions[axis].max() - positions[axis].min()
        for axis in ['x', 'y', 'z']
    }

    max_range = max(ranges.values())  # Find the largest range

    # Normalize positions based on the largest axis range
    positions_normalized = {
        axis: (positions[axis] - positions[axis].min()) / ranges[axis] * max_range
        for axis in ['x', 'y', 'z']
    }

    # Normalize time for color mapping
    norm = plt.Normalize(data['times'].min(), data['times'].max())

    # Create 3D plot
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Plot the 3D trajectory, coloring by time
    sc = ax.scatter(positions_normalized['x'], positions_normalized['y'], positions_normalized['z'],
                    c=data['times'], cmap='plasma', norm=norm)

    # Add colorbar to indicate time progression
    cbar = plt.colorbar(sc, ax=ax)
    cbar.set_label('Time (s)')

    # Label the axes with normalization indicated
    ax.set_xlabel('X Position (normalized)')
    ax.set_ylabel('Y Position (normalized)')
    ax.set_zlabel('Z Position (normalized)')

    if title is None:
        title = accel_trip.experiment_name
    plt.title(title)

    # Set equal scale for all axes
    ax.set_box_aspect([1, 1, 1])  # Equal aspect ratio

    # save
    if save_path is not None:
    # sanitize plural units
        plt.savefig(save_path+accel_trip.experiment_name+"_3d.png")
    # Show the plot
    plt.show()
