"""Unit tests for motion_direction module, testing direction calculation function."""

from motion_direction import calculate_direction

def test_calculate_direction():
    """Test the calculate_direction function with various XY-plane cases."""
    assert round(calculate_direction(1, 1, 0), 2) == 45.0
    assert round(calculate_direction(0, 1, 0), 2) == 90.0
    assert round(calculate_direction(-1, 0, 0), 2) == 180.0
    assert round(calculate_direction(0, -1, 0), 2) == -90.0

    try:
        calculate_direction(0, 0, 0)
    except ZeroDivisionError:
        assert True

    assert round(calculate_direction(1e-9, 1e-9, 0), 2) == 45.0
    assert round(calculate_direction(-1e-9, -1e-9, 0), 2) == -135.0
    assert round(calculate_direction(1, 1, 1), 2) == 45.0
