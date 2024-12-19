"""
This module provides unit testing for functions in your GPS module.
"""
import pytest
import pandas as pd
import numpy as np
import gps_location as gm

class TestGPSCoordinates:
    """
    Testing class for the gps_coordinates function in your GPS module.
    """
    def test_gps_coordinates_valid_input(self):
        """
        Test gps_coordinates function with valid csv input.
        Placeholder csv is called and then reset by pandas.
        """
        # Create a sample dataset and save as csv
        data = {
            'Longitude (°)': [0.0, 10.0, 20.0],
            'Latitude (°)': [0.0, -10.0, -20.0]
        }
        test_csv = 'test_placeholder.csv'
        pd.DataFrame(data).to_csv(test_csv, index=False)
        longitudes, latitudes = gm.gps_coordinates(test_csv)
        expected_longitudes = np.array([0.0, 10.0, 20.0])
        expected_latitudes = np.array([0.0, -10.0, -20.0])
        np.testing.assert_array_equal(longitudes, expected_longitudes)
        np.testing.assert_array_equal(latitudes, expected_latitudes)
        pd.DataFrame().to_csv(test_csv, index=False) #This resets csv. Source in resources.

    def test_gps_coordinates_file_not_found(self):
        """
        Test that the function raises FileNotFoundError when csv file is missing.
        """
        with pytest.raises(FileNotFoundError):
            gm.gps_coordinates('non_existent_file.csv')

class TestConvertGPSSpherical:
    """
    Testing class for the convert_gps_spherical function.
    """
    def test_convert_gps_spherical_valid_input(self):
        """
        Test the function with valid longitude and latitude inputs.
        """
        longitudes = [0.0, 90.0, 180.0]
        latitudes = [0.0, 45.0, 90.0]
        theta_expected = np.deg2rad(longitudes)
        phi_expected = np.pi / 2 - np.deg2rad(latitudes)
        theta, phi = gm.convert_gps_spherical(longitudes, latitudes)
        np.testing.assert_array_almost_equal(theta, theta_expected)
        np.testing.assert_array_almost_equal(phi, phi_expected)

    def test_convert_gps_spherical_invalid_input_type(self):
        """
        Test that the function raises TypeError when input types are incorrect.
        """
        with pytest.raises(TypeError):
            gm.convert_gps_spherical('invalid_input', 'invalid_input')

def test_spherical_to_cartesian_valid_input():
    """
    Test the function with valid theta and phi inputs.
    """
    theta = np.array([0.0, np.pi / 2])
    phi = np.array([np.pi / 2, np.pi / 2])
    radius = 1
    x_expected = np.array([1.0, 0.0])
    y_expected = np.array([0.0, 1.0])
    z_expected = np.array([0.0, 0.0])
    x, y, z = gm.spherical_to_cartesian(theta, phi, radius)
    np.testing.assert_array_almost_equal(x, x_expected)
    np.testing.assert_array_almost_equal(y, y_expected)
    np.testing.assert_array_almost_equal(z, z_expected)

class TestTotalDistance:
    """
    Testing class for the total_distance function.
    """
    def test_total_distance_same_point(self):
        """
        Test that the distance between the same point is zero.
        """
        dist = gm.total_distance(0,0,0,0)
        assert dist == pytest.approx(0.0, abs=1e-6)

    def test_total_distance_known_values(self):
        """
        Test the function with coordinates to Sydney Australia
        and London England. This test has 100m wiggle room, due to scale.
        """
        dist = gm.total_distance(151.21, -33.8651, -0.118092, 51.5099)
        expected_dist = 16993000
        assert dist == pytest.approx(expected_dist, abs=1e2)

class TestRemoveFirstValues:
    """
    Testing class for the remove_first_values function.
    """
    def test_remove_first_values_default(self):
        """
        Test the function with default parameter n=1.
        """
        longitudes = [0.0, 10.0, 20.0]
        latitudes = [0.0, -10.0, -20.0]
        long_filtered, lat_filtered = gm.remove_first_values(longitudes, latitudes)
        expected_long = [10.0, 20.0]
        expected_lat = [-10.0, -20.0]
        assert long_filtered == expected_long
        assert lat_filtered == expected_lat

    def test_remove_first_values_n_zero(self):
        """
        Test the function with n=0 (no removal).
        """
        longitudes = [0.0, 10.0, 20.0]
        latitudes = [0.0, -10.0, -20.0]
        long_filtered, lat_filtered = gm.remove_first_values(longitudes, latitudes, n=0)
        assert long_filtered == longitudes
        assert lat_filtered == latitudes

def test_adjacent_distances_simple_case():
    """
    Test the function with a csv file containing coordinates to Sydney Australia
    and London England. This test has 100m wiggle room, due to scale.
    Placeholder csv is called and then reset by pandas.
    """
    data = {
        'Longitude (°)': [151.21, -0.118092],
        'Latitude (°)': [-33.8651, 51.5099]
    }
    test_csv = 'test_placeholder.csv'
    pd.DataFrame(data).to_csv(test_csv, index=False)
    distances = gm.adjacent_distances(test_csv)
    expected_distance = [16993000]
    assert distances == pytest.approx(expected_distance, abs=1e2)
    pd.DataFrame().to_csv(test_csv, index=False)

def test_convert_xy_simple_case():
    """
    Test the function with simple longitude and latitude inputs.
    """
    longitudes = [0.0, 0.0]
    latitudes = [0.0, 1.0]
    x_values, y_values = gm.convert_xy(longitudes, latitudes)
    expected_x = [0.0, 0.0]
    expected_y = [0.0, gm.EARTH_R * np.deg2rad(1.0)]
    np.testing.assert_array_almost_equal(x_values, expected_x, decimal=2)
    np.testing.assert_array_almost_equal(y_values, expected_y, decimal=2)
