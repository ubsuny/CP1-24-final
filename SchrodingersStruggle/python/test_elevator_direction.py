"""
This module provides unit testing for functions in unit_conversion.py
"""
import pytest
import numpy as np
import elevator_direction as ed

class TestAcceleration:
    """
    Testing class acceleration funtion in elevator_direction.py
    """
    def test_file_not_found(self):
        """
        Test that function raises FileNotFoundError for a missing file.
        """
        csv = 'non_existent_file.csv'
        with pytest.raises(FileNotFoundError):
            ed.acceleration(csv)

    def test_invalid_input_type(self):
        """
        Test that function raises TypeError when input is not a string.
        """
        csv = 12345
        with pytest.raises(TypeError):
            ed.acceleration(csv)

class TestVelocity:
    """
    Testing class for the velocity function in elevator_direction.py
    """
    def test_velocity_simple_case(self):
        """
        Test the velocity function with simple acceleration and time data.
        """
        accelerations = [
            np.array([0.0, 1.0, 1.0]),  # x-axis
            np.array([0.0, 0.0, 0.0]),  # y-axis
            np.array([9.8, 9.8, 9.8])   # z-axis
        ]
        times = np.array([0.0, 1.0, 2.0])
        velocities = ed.velocity(accelerations, times)
        expected_velocities_x = [0, 0, 1]
        expected_velocities_y = [0, 0, 0]
        expected_velocities_z = [0, 9.8, 19.6]
        assert velocities[0] == expected_velocities_x
        assert velocities[1] == expected_velocities_y
        assert velocities[2] == expected_velocities_z
    def test_invalid_input_type(self):
        """
        Test that function raises TypeError when input is not as intended.
        """
        csv = 12345
        with pytest.raises(TypeError):
            ed.acceleration(csv)

class TestMovementDirections:
    """
    Testing class for the movement_directions function in elevator_direction.py
    """
    def test_movement_directions_specific_case(self):
        """
        Test the function in special case which often causes issues.
        """
        velocity = [
            [0], [-1], [-1]
        ]
        expected_vel_r = [np.sqrt(2)]
        expected_vel_theta = [3 * np.pi / 2]
        expected_vel_phi = [3 * np.pi / 4]
        vectors_sphere = ed.movement_directions(velocity)
        assert vectors_sphere[0][0] == pytest.approx(expected_vel_r[0], abs=1e-4)
        assert vectors_sphere[1][0] == pytest.approx(expected_vel_theta[0], abs=1e-4)
        assert vectors_sphere[2][0] == pytest.approx(expected_vel_phi[0], abs=1e-4)

    def test_movement_directions_zero_velocity(self):
        """
        Test the movement_directions function with zero velocities.
        """
        velocity = [
            [0], [0], [0]
        ]
        expected_vel_r = [0.0]
        expected_vel_theta = [0.0]
        expected_vel_phi = [0.0]
        vectors_sphere = ed.movement_directions(velocity)
        assert pytest.approx(vectors_sphere[0]) == expected_vel_r
        assert pytest.approx(vectors_sphere[1]) == expected_vel_theta
        assert pytest.approx(vectors_sphere[2]) == expected_vel_phi
