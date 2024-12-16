"""
test_trip.py

Tests the different classes in trip.py
"""

from unittest.mock import patch
import pytest
import pandas as pd
from trip import AccelTrip, GpsTrip, TripBase, CSVImportError

# Mock data for the DataFrames
mock_raw_frame = pd.DataFrame({"Column1": [1, 2, 3], "Column2": [4, 5, 6]})
mock_meta_frame = pd.DataFrame(
    {"event": ["START", "PAUSE"],
    "system time text": ["2023-01-01 00:00:00.000 UTC+0000",
    "2023-01-01 01:00:00.000 UTC+0000"]}
)

@patch("utils.import_csv", side_effect=[mock_raw_frame, mock_meta_frame])
def test_tripbase_initialization(mock_import_csv):
    """Test that TripBase initializes with the correct experiment name."""
    print(mock_import_csv)
    trip = TripBase("AVG001_gps_circle_walk_1")
    n = "AVG001_gps_circle_walk_1"
    m = "Experiment name should match the initialized value."
    assert trip.experiment_name == n, m

@patch("utils.import_csv", side_effect=[mock_raw_frame, mock_meta_frame])
def test_tripbase_get_raw_frame(mock_import_csv):
    """Test that get_raw_frame method returns a DataFrame."""
    print(mock_import_csv)
    trip = TripBase("AVG001_gps_circle_walk_1")
    raw_frame = trip.get_raw_frame()
    assert raw_frame is not None, "Raw frame should not be None."
    assert isinstance(raw_frame, pd.DataFrame), "Raw frame should be a DataFrame."

@patch("utils.import_csv", side_effect=[mock_raw_frame, mock_meta_frame])
def test_tripbase_get_raw_frame_meta(mock_import_csv):
    """Test that get_raw_frame_meta method returns a DataFrame."""
    print(mock_import_csv)
    trip = TripBase("AVG001_gps_circle_walk_1")
    raw_frame_meta = trip.get_raw_frame_meta()
    assert raw_frame_meta is not None, "Raw frame meta should not be None."
    assert isinstance(raw_frame_meta, pd.DataFrame), "Raw frame meta should be a DataFrame."

@patch("utils.import_csv", side_effect=[mock_raw_frame, mock_meta_frame])
def test_tripbase_report_trip_summary(mock_import_csv):
    """Test that report_trip_summary returns a formatted string."""
    print(mock_import_csv)
    trip = TripBase("AVG001_gps_circle_walk_1")
    summary = trip.report_trip_summary()
    assert isinstance(summary, str), "Report summary should be a string."
    assert "Trip Summary" in summary, "Summary should contain 'Trip Summary'."

@patch("utils.import_csv", side_effect=Exception("File not found"))
def test_tripbase_import_metadata_invalid_path(mock_import_csv):
    """Test that import_metadata raises an error with an invalid path."""
    print(mock_import_csv)
    trip = TripBase("AVG001_gps_circle_walk_1")
    with pytest.raises(CSVImportError):
        trip.import_metadata("invalid_path.csv")

# Mock data for testing GpsTrip
mock_raw_frame = pd.DataFrame({
    "Time (s)": [0, 1, 2, 3],
    "Latitude (°)": [37.7749, 37.7750, 37.7751, 37.7752],
    "Longitude (°)": [-122.4194, -122.4195, -122.4196, -122.4197],
    "Altitude (m)": [10, 15, 20, 25],
    "Altitude WGS84 (m)": [9, 14, 19, 24]
})
mock_meta_frame = pd.DataFrame({
    "event": ["START", "PAUSE"],
    "system time text": ["2023-01-01 00:00:00.000 UTC+0000", "2023-01-01 01:00:00.000 UTC+0000"]
})

@patch("utils.import_csv", side_effect=[mock_raw_frame, mock_meta_frame])
def test_gpstrip_initialization(mock_import_csv):
    """Test that GpsTrip initializes with the correct attributes."""
    print(mock_import_csv)
    gps_trip = GpsTrip("AVG001_gps_circle_walk_1")
    assert gps_trip.experiment_name == "AVG001_gps_circle_walk_1"
    assert gps_trip.trip_type == "GPS"

@patch("utils.import_csv", side_effect=[mock_raw_frame, mock_meta_frame])
def test_gpstrip_extract_gps_data(mock_import_csv):
    """Test that extract_gps_data method returns a DataFrame with correct columns."""
    print(mock_import_csv)
    gps_trip = GpsTrip("AVG001_gps_circle_walk_1")
    gps_data = gps_trip.extract_gps_data()
    expected_columns = [
        "LinterFix",
        "Latitude (°)",
        "Longitude (°)",
        "Altitude (m)",
        "Altitude WGS84 (m)"
    ]
    expected_columns[0] = "Time (s)"
    assert gps_data is not None, "GPS data should not be None."
    assert isinstance(gps_data, pd.DataFrame), "GPS data should be a DataFrame."
    m1 = "All expected columns should be present."
    assert all(col in gps_data.columns for col in expected_columns), m1

@patch("utils.import_csv", side_effect=[mock_raw_frame, mock_meta_frame])
def test_gpstrip_calculate_segments(mock_import_csv):
    """Test that calculate_segments returns a DataFrame with segment data."""
    print(mock_import_csv)
    gps_trip = GpsTrip("AVG001_gps_circle_walk_1")
    segments = gps_trip.calculate_segments()
    expected_columns = [
        'start_lat', 'start_long', 'start_alt',
        'end_lat', 'end_long', 'end_alt',
        'planar_distance', 'curved_distance'
    ]
    assert segments is not None, "Segments data should not be None."
    assert isinstance(segments, pd.DataFrame), "Segments data should be a DataFrame."
    m1 = "All expected columns should be present in segments."
    assert all(col in segments.columns for col in expected_columns), m1

@patch("utils.import_csv", side_effect=[mock_raw_frame, mock_meta_frame])
def test_gpstrip_report_trip_summary(mock_import_csv):
    """Test that report_trip_summary includes key trip details."""
    print(mock_import_csv)
    gps_trip = GpsTrip("AVG001_gps_circle_walk_1")
    summary = gps_trip.report_trip_summary()
    assert isinstance(summary, str), "Report summary should be a string."
    assert "Trip Summary" in summary, "Summary should contain 'Trip Summary'."
    m1 = "Summary should include total planar distance."
    m2 = "Summary should include total curved distance."
    assert "Total planar distance traveled" in summary, m1
    assert "Total curved distance traveled" in summary, m2

# Mock data for testing AccelTrip
mock_raw_frame_accel = pd.DataFrame({
    "Time (s)": [0, 1, 2, 3],
    "Linear Acceleration x (m/s^2)": [0.1, 0.2, 0.15, 0.1],
    "Linear Acceleration y (m/s^2)": [0.0, -0.1, -0.2, -0.15],
    "Linear Acceleration z (m/s^2)": [9.8, 9.7, 9.6, 9.8],
    "Absolute acceleration (m/s^2)": [9.81, 9.72, 9.63, 9.81]
})
mock_meta_frame_accel = pd.DataFrame({
    "event": ["START", "PAUSE"],
    "system time text": ["2023-01-01 00:00:00.000 UTC+0000", "2023-01-01 01:00:00.000 UTC+0000"]
})

@patch("utils.import_csv", side_effect=[mock_raw_frame_accel, mock_meta_frame_accel])
def test_acceltrip_initialization(mock_import_csv):
    """Test that AccelTrip initializes with the correct attributes."""
    print(mock_import_csv)
    accel_trip = AccelTrip("AVG001_accel_elevator_up_1")
    assert accel_trip.experiment_name == "AVG001_accel_elevator_up_1"
    assert accel_trip.trip_type == "ACCEL"

@patch("utils.import_csv", side_effect=[mock_raw_frame_accel, mock_meta_frame_accel])
def test_acceltrip_extract_accel_data(mock_import_csv):
    """Test that extract_accel_data method returns a DataFrame with correct columns."""
    print(mock_import_csv)
    accel_trip = AccelTrip("AVG001_accel_elevator_up_1")
    accel_data = accel_trip.extract_accel_data()
    expected_columns = ["time", "accel_x", "accel_y", "accel_z", "accel_absolute"]
    assert accel_data is not None, "Accelerometer data should not be None."
    assert isinstance(accel_data, pd.DataFrame), "Accelerometer data should be a DataFrame."
    m1 = "All expected columns should be present."
    assert all(col in accel_data.columns for col in expected_columns), m1

@patch("utils.import_csv", side_effect=[mock_raw_frame_accel, mock_meta_frame_accel])
def test_acceltrip_calculate_segments(mock_import_csv):
    """Test that calculate_segments returns a DataFrame with segment data."""
    print(mock_import_csv)
    accel_trip = AccelTrip("AVG001_accel_elevator_up_1")
    segments = accel_trip.calculate_segments()
    expected_columns = [
        'start_t', 'stop_t', 'delta_t',
        'accel_x', 'accel_y', 'accel_z',
        'velocity_x', 'velocity_y', 'velocity_z'
    ]
    assert segments is not None, "Segments data should not be None."
    assert isinstance(segments, pd.DataFrame), "Segments data should be a DataFrame."
    m1 = "All expected columns should be present in segments."
    assert all(col in segments.columns for col in expected_columns), m1

@patch("utils.import_csv", side_effect=[mock_raw_frame_accel, mock_meta_frame_accel])
def test_acceltrip_report_trip_summary(mock_import_csv):
    """Test that report_trip_summary includes key trip details."""
    print(mock_import_csv)
    accel_trip = AccelTrip("AVG001_accel_elevator_up_1")
    summary = accel_trip.report_trip_summary()
    assert isinstance(summary, str), "Report summary should be a string."
    assert "Trip Summary" in summary, "Summary should contain 'Trip Summary'."
