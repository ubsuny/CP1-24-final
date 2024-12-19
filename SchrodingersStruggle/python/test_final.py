"""
Unit tests for the final.py module functions used in the CP1-24 final project.

This test module covers:
- Temperature conversion from Fahrenheit to Kelvin
- Temperature data parsing from markdown files
- Filename filtering and listing
- Non-linear fitting functionality
- FFT analysis validation
- Frequency axis calculations

All tests follow pytest conventions and test both valid and edge cases.
"""

import pytest
import final as f

class TestFahrenheitToKelvin:
    """
    Test class for the fahrenheit_to_kelvin temperature conversion function.
    Tests standard conversions, edge cases, and error handling.
    """

    def test_room_temperature(self):
        """Test conversion of room temperature (72°F)"""
        assert f.fahrenheit_to_kelvin(72) == pytest.approx(295.372, rel=1e-6)

    def test_absolute_zero(self):
        """Test conversion of absolute zero (-459.67°F)"""
        assert f.fahrenheit_to_kelvin(-459.67) == pytest.approx(0.0, rel=1e-6)
        
    def test_below_absolute_zero(self):
        """Test that temperatures below absolute zero raise ValueError"""
        with pytest.raises(ValueError):
            f.fahrenheit_to_kelvin(-460)
            
    def test_invalid_input_type(self):
        """Test that non-numeric inputs raise TypeError"""
        with pytest.raises(TypeError):
            f.fahrenheit_to_kelvin("72")

class TestCalculateFrequencyAxis:
    """Test suite for frequency axis calculation function"""
    
    def test_basic_frequency_axis(self):
        """Test basic frequency axis calculation"""
        freqs = f.calculate_frequency_axis(sample_rate=100, num_points=4)
        expected = [0, 25, -50, -25]
        assert len(freqs) == 4
        assert all(abs(a - b) < 1e-10 for a, b in zip(freqs, expected))
    
    def test_invalid_inputs(self):
        """Test error handling for invalid inputs"""
        with pytest.raises(TypeError):
            f.calculate_frequency_axis("100", 4)
        with pytest.raises(TypeError):
            f.calculate_frequency_axis(100, 4.5)
        with pytest.raises(ValueError):
            f.calculate_frequency_axis(-100, 4)
        with pytest.raises(ValueError):
            f.calculate_frequency_axis(100, 0)
            
    def test_nyquist_frequency(self):
        """Test that maximum frequency is half the sample rate"""
        freqs = f.calculate_frequency_axis(sample_rate=1000, num_points=8)
        assert max(freqs) == pytest.approx(500, rel=1e-10)

class TestParseTemperature:
    """Test suite for markdown temperature parser"""
    
    def test_basic_temperature_parse(self, tmp_path):
        """Test parsing of standard temperature format"""
        test_md = tmp_path / "test.md"
        test_md.write_text("Date: 2024-01-20\nEnvironment temperature: 72.5°F\nNotes: None")
        assert f.parse_temperature_from_markdown(str(test_md)) == 72.5

    def test_alternative_formats(self, tmp_path):
        """Test parsing of alternative temperature formats"""
        formats = [
            "Temperature: 72.5°F",
            "Temp: 72.5°F",
            "Environment temperature: 72.5F"
        ]
        for fmt in formats:
            test_md = tmp_path / "test.md"
            test_md.write_text(fmt)
            assert f.parse_temperature_from_markdown(str(test_md)) == 72.5

    def test_file_not_found(self):
        """Test error handling for missing file"""
        with pytest.raises(FileNotFoundError):
            f.parse_temperature_from_markdown("nonexistent.md")

    def test_no_temperature(self, tmp_path):
        """Test error handling for file without temperature"""
        test_md = tmp_path / "test.md"
        test_md.write_text("Date: 2024-01-20\nNotes: None")
        with pytest.raises(ValueError):
            f.parse_temperature_from_markdown(str(test_md))
