"""
trip.py

This module defines the TripBase class, which serves as a foundational class for handling trip data.
The class is designed to import trip-related CSV files and metadata, extract important information
such as start/stop times, and calculate the duration of trips.

Classes:
    TripBase: A base class that handles the import and processing of trip data and metadata.
              It includes methods to retrieve raw trip data and metadata, as well as functionality
              to extract time-related details (e.g., start/stop times in Unix and UTC formats).

Functions:
    _import_metadata: Imports metadata from a CSV file and extracts trip timing information.
    _get_raw_frame: Retrieves the raw trip data (stored in a private variable).
    _get_raw_frame_meta: Retrieves the metadata (stored in a private variable).

Exceptions:
    CSVImportError: Custom exception raised when an error occurs during CSV import.
"""

import pandas as pd
import numpy as np

from utils import import_csv, CSVImportError
from utils import haversine_with_altitude as haversine
from utils import timecode_to_unix

class TripBase:
    """
    A base class that imports a CSV and stores it in a private variable `raw_frame`
    and metadata in `raw_frame_meta`. It also extracts metadata such as start/stop
    times and calculates the trip duration.

    This class is not meant to be used directly. Instead you should use either
    the GpsTrip or AccelTrip classes.
    """
    def __init__(self, csv_unzip_name):
        """
        Initializes the TripBase class by importing trip data and metadata from
        CSV files.

        Args:
            csv_name (str): Name of the folder with CSV file that is named
            "Raw Data.csv". If you unzip PhyFox data export (as a complete dir),
            this will be the name of the new directory.
        """
        self.experiment_name = csv_unzip_name

        # Ensure that the directory path ends with a slash
        if not csv_unzip_name.endswith('/'):
            csv_dir = 'avgagliardo/data/' + csv_unzip_name + '/'  # Directory path
        else:
            csv_dir = 'avgagliardo/data/' + csv_unzip_name        # Directory path

        # Prepare the csv paths
        csv_path = csv_dir + "Raw Data" + '.csv'                 # Filename
        csv_meta_path = csv_dir + 'meta/time.csv'  # Meta name

        # initilize the times dict
        self.times = {
            "start_time_unix": None,
            "stop_time_unix": None,
            "start_time_utc": None,
            "stop_time_utc": None,
            "duration": None,
        }

        # Initialize the raw and meta data frames as private members
        # print("Importing data from: " + csv_path)
        self.__csv_path = csv_path
        # print("Importing meta from: " + csv_dir + "meta/time.csv" )
        self.__csv_meta_path = csv_meta_path

        # Import and store the raw data and metadata
        self.__raw_frame = import_csv(self.__csv_path)
        self.__raw_frame_meta = self.import_metadata(self.__csv_meta_path)

        # Define a prefix for labels
        self.trip_type = "BASE"

        # # Define the member for the subclassed data frames and trip segments
        # self.data = None
        # self.segments = None

    def get_raw_frame(self):
        """
        Protected method to access the private raw_frame. This can be accessed
        by child objects to extract data relevant to their context.

        Returns:
            DataFrame: The trip data stored in the private variable
            `__raw_frame`.
        """
        return self.__raw_frame

    def get_raw_frame_meta(self):
        """
        Protected method to access the private raw_frame_meta. This can be
        accessed
        by child objects to extract metadata relevant to their context.

        Returns:
            DataFrame: The metadata stored in the private variable
            `__raw_frame_meta`.
        """
        return self.__raw_frame_meta

    def import_metadata(self, meta_csv_path):
        """
        Imports metadata CSV and extracts start and stop times in Unix and UTC
        formats.
        Stores the duration of the experiment as well.

        Args:
            meta_csv_path (str): Path to the metadata CSV file.

        Returns:
            DataFrame: The metadata DataFrame stored in the private variable
            `__raw_frame_meta`.
        """
        try:
            # sanitize path
            if meta_csv_path.endswith('/'):
                meta_csv_path = meta_csv_path[:-1]

            meta_df = import_csv(meta_csv_path)

            if meta_df is not None:
                # Extract start and stop times from the metadata
                start_event = meta_df[meta_df['event'] == 'START'].iloc[0]
                stop_event = meta_df[meta_df['event'] == 'PAUSE'].iloc[0]

                # Extract Unix and human-readable times from the CSV
                start_time_unix = timecode_to_unix(start_event['system time text'])
                stop_time_unix = timecode_to_unix(stop_event['system time text'])
                self.times['start_time_unix'] = start_time_unix
                self.times['stop_time_unix'] = stop_time_unix
                self.times['start_time_utc'] = start_event['system time text']
                self.times['stop_time_utc'] = stop_event['system time text']

                # Calculate the duration in seconds
                dur = self.times['stop_time_unix'] - self.times['start_time_unix']
                self.times['duration'] = dur
            else:
                # create an error message
                msg = f"Failed to import metadata from" \
                      f"{meta_csv_path}. DataFrame is None."
                # raise the exception with the details in msg
                raise CSVImportError(msg)

            return meta_df

        except Exception as e:
            msg = "An error occurred while importing the" \
                  f" metadata CSV: {str(e)}"
            raise CSVImportError(msg) from e

    def report_trip_summary(self):
        """
        Public method that generates a summary report of the trip details.
        It reports the trip type, start time in UTC, trip duration, and the
        number of frames in the raw trip data.

        Returns:
            str: A formatted string containing the trip type, start time (UTC),
            duration (in seconds), and the number of data frames.
        """
        experiment_name = self.experiment_name
        trip_type = self.trip_type
        start_time_utc = self.times['start_time_utc'] if self.times['start_time_utc'] else 'Unknown'
        start_time_unix = (self.times['start_time_unix']
            if self.times['start_time_unix'] else 'Unknown')
        duration = self.times['duration'] if self.times['duration'] is not None else 'Unknown'
        num_frames = len(self.__raw_frame) if self.__raw_frame is not None else 0

        return(
            "\n--------------------\n"
            f"Trip Summary:\n"
            "--------------------\n"
            f"Experiment Name: {experiment_name}\n"
            f"Type of trip: {trip_type}\n"
            f"Start time (UTC): {start_time_utc}\n"
            f"Start time (Unix): {start_time_unix}\n"
            f"Duration: {duration} seconds\n"
            f"Number of frames: {num_frames}"
        )

class GpsTrip(TripBase):
    """
    A class that inherits from TripBase, specifically for trips that include GPS data.
    This class adds functionality to handle GPS coordinates and other related metadata.

    Attributes:
        trip_type (str): The type of the trip, overridden to be 'GPS' for this class.
        data (DataFrame): A DataFrame holding specific GPS data (Time, Latitude,
        Longitude, Altitude, Altitude WGS84).
    """
    def __init__(self, csv_name):
        """
        Initializes the GpsTrip class by calling the base class initializer to handle
        trip data and metadata import. Additionally, it loads GPS-specific data.

        Args:
            csv_name (str): Name of the CSV file (without extension) that contains trip data.
        """
        # Call the base class initializer
        super().__init__(csv_name)

        self.experiment_name = csv_name
        # Set the trip_type to "GPS" for this specific trip
        self.trip_type = "GPS"

        # Extract GPS-specific data and store in self.data
        self.data = self.extract_gps_data()
        self.segments = self.calculate_segments()

        # report on the trip's status
        print(self.report_trip_summary())

    def extract_gps_data(self):
        """
        Extracts specific GPS data columns from the raw trip data, including:
        "Time (s)", "Latitude (°)", "Longitude (°)", "Altitude (m)",
        "Altitude WGS84 (m)".

        If any of the required columns are missing, it prints an error message.

        Returns:
            DataFrame: A DataFrame containing GPS data if all required columns
            are present, otherwise returns None.
        """
        required_columns = [
            "Time (s)",
            "Latitude (°)",
            "Longitude (°)",
            "Altitude (m)",
            "Altitude WGS84 (m)"
        ]

        # Check if the raw frame has been loaded
        rf = self.get_raw_frame()
        if rf is not None:
            # Check if all required columns are present in the raw data
            if all(col in rf.columns for col in required_columns):
                # Extract the required columns and store them in a new DataFrame
                gps_data = rf[required_columns]
                return gps_data

            missing_cols = [col for col in required_columns if col not in rf.columns]
            print(f"Error: Missing GPS columns in the raw data: {missing_cols}")
        else:
            print("Error: Raw frame is empty.")

        # Return None if extraction failed
        return None

    def calculate_segments(self):
        """
        Takes a DataFrame with 'Latitude (°)', 'Longitude (°)', and 'Altitude (m)' columns
        and calculates the Euclidean distance, Haversine distance, and 3D distance between
        each pair of sequential GPS points. Stores the start point, end point, and distances
        in a new DataFrame called 'segments'.

        Returns:
            DataFrame: A DataFrame containing the start point, end point, and calculated distances.
        """
        # Get the dataframe and make a copy
        df = self.data.copy()

        # print(self.get_raw_frame())
        # print(df)
        # Shift latitude, longitude, and altitude to get the next point's coordinates
        df['Latitude_next'] = df['Latitude (°)'].shift(-1)
        df['Longitude_next'] = df['Longitude (°)'].shift(-1)
        df['Altitude_next (m)'] = df['Altitude (m)'].shift(-1)
        df['Altitude WGS84_next (m)'] = df['Altitude WGS84 (m)'].shift(-1)
        df['Time_next'] = df['Time (s)'].shift(-1)

        # Drop the last row, as it will have NaN values due to shifting
        df = df.dropna(
            subset=[
            'Latitude_next',
            'Longitude_next',
            'Altitude_next (m)',
            'Altitude WGS84_next (m)',
            'Time_next'
            ]
        )


        # Calculate Latitude and Longitude displacements
        df['Latitude_Displacement'] = df['Latitude_next'] - df['Latitude (°)']
        df['Longitude_Displacement'] = df['Longitude_next'] - df['Longitude (°)']

        # Calculate the Euclidean distance between each pair of points (ignoring
        # altitude for now)
        df['Degree_Distance'] = np.sqrt(
            (df['Latitude_next'] - df['Latitude (°)'])**2 +
            (df['Longitude_next'] - df['Longitude (°)'])**2
        )

        # Convert latitude and longitude displacements to meters
        # note the extra factor in longitude displacement
        # Latitude displacement in meters
        df['Latitude_Displacement_m'] = df['Latitude_Displacement'] * 111320
        # Longitude displacement in meters
        df['Longitude_Displacement_m'] = (df['Longitude_Displacement'] * 111320
            * np.cos(np.radians(df['Latitude (°)'])))


        # Calculate the planar distance in meters
        df['Planar_Distance'] = np.sqrt(df['Latitude_Displacement_m']**2 +
            df['Longitude_Displacement_m']**2)

        #compose dicts to pass as the segment's two points
        start_point = {
            'lat': df['Latitude (°)'],
            'lon': df['Longitude (°)'],
            'alt': df['Altitude WGS84 (m)']
        }

        end_point = {
            'lat': df['Latitude_next'],
            'lon': df['Longitude_next'],
            'alt': df['Altitude WGS84_next (m)']
        }



        # Calculate the 3D curved distance (Haversine + Altitude difference)
        df['Curved_Distance'] = haversine(start_point, end_point)

        # Create the segments DataFrame with start point, end point, planar
        # distance, and curved distance
        segments = pd.DataFrame({
            # units of s
            'start_t': df['Time (s)'],
            # units of s
            'stop_t': df['Time_next'],
            # units of degrees
            'start_lat': df['Latitude (°)'],
            # units of degrees
            'start_long': df['Longitude (°)'],
            # units of meters
            'start_alt': df['Altitude (m)'],
            # units of degrees
            'end_lat': df['Latitude_next'],
            # units of degrees
            'end_long': df['Longitude_next'],
            # units of meters
            'end_alt': df['Altitude_next (m)'],
            # Displacement for Latitude, degrees
            'lat_delta': df['Latitude_Displacement'],
            # Displacement for Longitude. degrees
            'long_delta': df['Longitude_Displacement'],
            # Euclidean distance in degrees
            'degree_distance': df['Degree_Distance'],
            # Planar distance in meters
            'planar_distance': df['Planar_Distance'],
            # 3D distance in meters
            'curved_distance': df['Curved_Distance']
        })

        return segments

    def report_trip_summary(self):
        """
        Public method that generates a summary report of the trip details.
        It reports the trip type, start time in UTC, trip duration, the number
        of frames in the raw trip data, and the total distance traveled (both
        planar and curved).

        Returns:
            str: A formatted string containing the trip type, start time (UTC),
            duration (in seconds),
            number of data frames, total planar distance, and total curved distance.
        """
        experiment_name = self.experiment_name
        trip_type = self.trip_type
        start_time_utc = self.times['start_time_utc'] if self.times['start_time_utc'] else 'Unknown'
        start_time_unix = (self.times['start_time_unix']
            if self.times['start_time_unix'] else 'Unknown')
        duration = (self.times['duration']
            if self.times['duration'] is not None else 'Unknown')
        num_frames = len(self.get_raw_frame()) if self.get_raw_frame() is not None else 0

        # Calculate total planar and curved distances
        total_planar_distance = (self.segments['planar_distance'].sum()
            if self.segments is not None else 0)
        total_curved_distance = (self.segments['curved_distance'].sum()
            if self.segments is not None else 0)

        return (
            "\n--------------------\n"
            f"Trip Summary:\n"
            "--------------------\n"
            f"Experiment Name: {experiment_name}\n"
            f"Type of trip: {trip_type}\n"
            f"Start time (UTC): {start_time_utc}\n"
            f"Start time (Unix): {start_time_unix}\n"
            f"Duration: {duration} seconds\n"
            f"Number of frames: {num_frames}\n"
            f"Total planar distance traveled: {total_planar_distance:.2f} meters\n"
            f"Total curved distance traveled: {total_curved_distance:.2f} meters"
        )

class AccelTrip(TripBase):
    """
    A class that inherits from TripBase, specifically for trips that include
    accelerometer data.

    This class stubs out methods that will eventually handle accelerometer-
    specific data.

    Attributes:
        trip_type (str): The type of the trip, overridden to be 'ACCEL' for
        this class.
        data (DataFrame): A DataFrame holding specific accelerometer data (e.g.,
        time, x, y, z axes).
        original_data (DataFrame): A copy of the original raw data for rethresholding.
    """
    def __init__(self, csv_name, accel_thresholds=None, velocity_thresholds=None):
        """
        Initializes the AccelTrip class by calling the base class initializer
        to handle trip data and metadata import. Additionally, it will eventually
        handle accelerometer-specific data.

        The accel_thresholds dict has two keys, "upper" and "lower" that control
        the data thresholding of the acceleration frame data. The velocity_thresholds
        dict controls thresholding for velocity data.

        Args:
            csv_name (str): Name of the CSV file (without extension) that
            contains trip data.
            accel_thresholds (dict): A dictionary containing 'lower' and 'upper'
            thresholds for acceleration.
            velocity_thresholds (dict): A dictionary containing 'lower' and 'upper'
            thresholds for velocity.
        """
        # Call the base class initializer
        super().__init__(csv_name)

        # Set default threshold values if not provided
        if accel_thresholds is None:
            accel_thresholds = {'lower': -50, 'upper': 50.0}
        if velocity_thresholds is None:
            velocity_thresholds = {'lower': -10.0, 'upper': 10.0}

        # Set the trip_type to "ACCEL" for this specific trip
        self.experiment_name = csv_name
        self.trip_type = "ACCEL"

        # Extract accelerometer data and store it
        self.data = self.extract_accel_data()

        # Store the original, unmodified data for rethresholding
        self.original_data = self.data.copy()

        # Apply the initial thresholding and calculate segments
        self.segments = self.calculate_segments(accel_thresholds, velocity_thresholds)

    def rethreshold_data(self, new_accel_thresholds=None, new_velocity_thresholds=None):
        """
        Reapplies thresholding to the original accelerometer data and updates
        the segments based on the new thresholds.

        Args:
            new_accel_thresholds (dict): New thresholds for acceleration.
            new_velocity_thresholds (dict): New thresholds for velocity.

        Returns:
            None
        """
        # Apply the new thresholds to the original data
        self.segments = self.calculate_segments(new_accel_thresholds, new_velocity_thresholds)

    def extract_accel_data(self):
        """
        Extracts accelerometer data from the raw trip data, including:
        - Time (s)
        - Linear Acceleration x (m/s^2)
        - Linear Acceleration y (m/s^2)
        - Linear Acceleration z (m/s^2)
        - Absolute acceleration (m/s^2)

        The columns are renamed to more convenient names for processing:
        - Time (s) → time
        - Linear Acceleration x (m/s^2) → accel_x
        - Linear Acceleration y (m/s^2) → accel_y
        - Linear Acceleration z (m/s^2) → accel_z
        - Absolute acceleration (m/s^2) → accel_absolute

        Returns:
            DataFrame: A DataFrame containing the renamed accelerometer
            data columns.
        """
        raw_frame = self.get_raw_frame()
        if raw_frame is not None:
            required_columns = [
                "Time (s)",
                "Linear Acceleration x (m/s^2)",
                "Linear Acceleration y (m/s^2)",
                "Linear Acceleration z (m/s^2)",
                "Absolute acceleration (m/s^2)"
            ]

            if all(col in raw_frame.columns for col in required_columns):
                accel_data = raw_frame[required_columns].rename(columns={
                    "Time (s)": "time",
                    "Linear Acceleration x (m/s^2)": "accel_x",
                    "Linear Acceleration y (m/s^2)": "accel_y",
                    "Linear Acceleration z (m/s^2)": "accel_z",
                    "Absolute acceleration (m/s^2)": "accel_absolute"
                })
            else:
                missing_cols = [col for col in required_columns if col not in raw_frame.columns]
                print(f"Error: Missing accelerometer columns in the raw data: {missing_cols}")
                accel_data = None
        else:
            print("Error: Raw frame is empty.")
            accel_data = None
        return accel_data

    def calculate_segments(self, accel_thresholds=None, velocity_thresholds=None):
        """
        Calculates segments from the accelerometer data, including start and
        stop times, delta time, raw accelerations, and velocity for each segment,
        with optional thresholding for acceleration and velocity.

        Args:
            accel_thresholds (dict): A dictionary containing 'lower' and 'upper'
            keys for acceleration thresholds.
            velocity_thresholds (dict): A dictionary containing 'lower' and
            'upper' keys for velocity thresholds.

        Returns:
            DataFrame: Segments data with filtered acceleration and velocity
            based on thresholds.
        """
        if self.original_data is None:
            print("Error: No accelerometer data available.")
            return None

        df = self.original_data.copy()

        # Shift the time column to calculate the stop time for each segment
        df['time_next'] = df['time'].shift(-1)

        # Drop the last row since it will have NaN values after shifting
        df = df.dropna(subset=['time_next'])

        # Calculate delta_t (duration of each segment) using relative times
        df['delta_t'] = df['time_next'] - df['time']

        # Filter out rows where delta_t is too small (this ensures stable integration)
        df = df[df['delta_t'] > 1e-5]

        # Use raw accelerations (not averaged) for each axis
        df['accel_x'] = df['accel_x']
        df['accel_y'] = df['accel_y']
        df['accel_z'] = df['accel_z']

        # Calculate the total acceleration magnitude for reference if needed
        df['total_acceleration'] = (
            (df['accel_x']**2 + df['accel_y']**2 + df['accel_z']**2)**0.5
        )

        # Calculate velocity for each segment independently (without accumulation)
        df['velocity_x'] = df['accel_x'] * df['delta_t']
        df['velocity_y'] = df['accel_y'] * df['delta_t']
        df['velocity_z'] = df['accel_z'] * df['delta_t']

        # Apply acceleration thresholding if provided
        if accel_thresholds:
            lower_accel = accel_thresholds.get('lower', -np.inf)
            upper_accel = accel_thresholds.get('upper', np.inf)
            df = df[(df['total_acceleration'] >= lower_accel) &
                    (df['total_acceleration'] <= upper_accel)]

        # Apply velocity thresholding if provided
        if velocity_thresholds:
            lower_vel = velocity_thresholds.get('lower', -np.inf)
            upper_vel = velocity_thresholds.get('upper', np.inf)
            df = df[(df['velocity_x'] >= lower_vel) & (df['velocity_x'] <= upper_vel) &
                    (df['velocity_y'] >= lower_vel) & (df['velocity_y'] <= upper_vel) &
                    (df['velocity_z'] >= lower_vel) & (df['velocity_z'] <= upper_vel)]

        # Create and return the segments DataFrame
        segments = pd.DataFrame({
            'start_t': df['time'],
            'stop_t': df['time_next'],
            'delta_t': df['delta_t'],
            'accel_x': df['accel_x'],
            'accel_y': df['accel_y'],
            'accel_z': df['accel_z'],
            'total_acceleration': df['total_acceleration'],
            'velocity_x': df['velocity_x'],
            'velocity_y': df['velocity_y'],
            'velocity_z': df['velocity_z'],
        })

        return segments

    def report_trip_summary(self):
        """
        Public method that generates a summary report of the trip details.
        It reports the trip type, start time in UTC, trip duration, and the
        number of frames in the raw trip data.

        Returns:
            str: A formatted string containing the trip type, start time (UTC),
            duration (in seconds), and the number of data frames.
        """
        experiment_name = self.experiment_name
        trip_type = self.trip_type
        start_time_utc = (self.times['start_time_utc'] if
            self.times['start_time_utc'] else 'Unknown')
        start_time_unix = (self.times['start_time_unix'] if
            self.times['start_time_unix'] else 'Unknown')
        duration = (self.times['duration'] if
            self.times['duration'] is not None else 'Unknown')
        num_frames = (len(self.get_raw_frame()) if
            self.get_raw_frame() is not None else 0)

        return (
            "\n--------------------\n"
            f"Trip Summary:\n"
            "--------------------\n"
            f"Experiment Name: {experiment_name}\n"
            f"Type of trip: {trip_type}\n"
            f"Start time (UTC): {start_time_utc}\n"
            f"Start time (Unix): {start_time_unix}\n"
            f"Duration: {duration} seconds\n"
            f"Number of frames: {num_frames}"
        )
