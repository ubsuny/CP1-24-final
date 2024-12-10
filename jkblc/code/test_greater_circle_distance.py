"""Test greater cirle distance module"""

from greater_circle_distance import haversine

def test_haversine_same_point():
    """Test haversine function at the same point"""
    assert haversine((0, 0), (0, 0)) == 0

def test_haversine_known_distance():
    """Tests haversine function given a known distance"""
    # Distance between Paris (48.8566, 2.3522) and London (51.5074, -0.1278) ~ 343,774 meters
    distance = haversine((48.8566, 2.3522), (51.5074, -0.1278))
    assert 343000 < distance < 344000
