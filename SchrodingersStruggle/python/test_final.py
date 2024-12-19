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
import math as mt
import numpy as np
import random as rd
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

class TestListMarkdownFiles:
    """Test suite for markdown file lister"""
    
    def test_basic_file_listing(self, tmp_path):
        """Test basic file listing with pattern"""
        # Create test files
        # The .touch() part at the end creates an empty file
        (tmp_path / "test1_sinewalk.md").touch()
        (tmp_path / "test2_sinewalk.md").touch()
        (tmp_path / "other.md").touch()
        
        files = f.list_markdown_files(str(tmp_path), "sinewalk")
        assert len(files) == 2
        assert all("sinewalk" in f for f in files)
        
    def test_empty_directory(self, tmp_path):
        """Test listing files in empty directory"""
        files = f.list_markdown_files(str(tmp_path), "sinewalk")
        assert len(files) == 0
        
    def test_no_matching_files(self, tmp_path):
        """Test when no files match pattern"""
        (tmp_path / "test.md").touch()
        files = f.list_markdown_files(str(tmp_path), "sinewalk")
        assert len(files) == 0
        
    def test_directory_not_found(self):
        """Test error handling for nonexistent directory"""
        with pytest.raises(FileNotFoundError):
            f.list_markdown_files("nonexistent_dir", "sinewalk")
            
    def test_empty_pattern(self, tmp_path):
        """Test error handling for empty pattern"""
        with pytest.raises(ValueError):
            f.list_markdown_files(str(tmp_path), "")
class TestFFTWrapper:
    """Test suite for FFT wrapper functions"""
    
    def test_basic_fft(self):
        """Test basic FFT of sine wave"""
        import numpy as np
        t = np.linspace(0, 1, 128)
        data = np.sin(2 * np.pi * 10 * t)  # 10 Hz sine wave
        freqs, fft_result = f.fft_wrapper(data, t)
        magnitude = np.abs(fft_result)
        # Find the frequency with maximum magnitude
        peak_freq = abs(freqs[np.argmax(magnitude)])
        assert abs(peak_freq - 10) < 0.1  # Should find 10 Hz peak
        
    def test_non_equidistant_data(self):
        """Test error for non-equidistant data"""
        import numpy as np
        t = [0, 0.1, 0.3, 0.5]  # Non-equidistant
        data = [0, 1, 0, -1]
        with pytest.raises(ValueError):
            f.fft_wrapper(data, t)
            
    def test_ifft_recovery(self):
        """Test that IFFT recovers original signal"""
        t = np.linspace(0, 1, 128)
        data = np.sin(2 * np.pi * 10 * t)
        _, fft_result = f.fft_wrapper(data, t)
        recovered = f.ifft_wrapper(fft_result)
        assert np.allclose(data, recovered, rtol=1e-10)
