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
import numpy as np
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
    """Test class for markdown temperature parser"""

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
    """Test class for markdown file lister"""

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

class TestFFTiFFTWrappers:
    """Test class for FFT wrapper functions"""

    def test_basic_fft(self):
        """Test basic FFT of sine wave"""
        t = np.linspace(0, 1, 128)
        data = np.sin(2 * np.pi * 10 * t)  # 10 Hz sine wave
        freqs, fft_result = f.fft_wrapper(data, t)
        magnitude = np.abs(fft_result)
        # Find the frequency with maximum magnitude
        peak_freq = abs(freqs[np.argmax(magnitude)])
        assert abs(peak_freq - 10) < 0.1  # Should find 10 Hz peak

    def test_non_equidistant_data(self):
        """Test error for non-equidistant data"""
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

class TestFitNonlinear:
    """Test class for the fit_nonlinear function"""

    def test_basic_sine_fit(self):
        """Test fitting a basic sine wave"""
        x = np.linspace(0, 2 * np.pi, 100)
        y = 3 * np.sin(x + np.pi/4) + 1
        params = f.fit_nonlinear(x, y, 5)
        assert params['amplitude'] == pytest.approx(3, rel=1e-1)
        assert params['frequency'] == pytest.approx(1 / (2 * np.pi), rel=1e-1)
        assert params['phi'] == pytest.approx(np.pi/4, rel=1e-1)
        assert params['constant'] == pytest.approx(1, rel=1e-1)

    def test_noisy_sine_fit(self):
        """Test fitting a noisy sine wave"""
        np.random.seed(0)
        x = np.linspace(0, 2 * np.pi, 100)
        y = 2 * np.sin(3 * x + 1) + 0.5 + np.random.normal(0, 0.1, x.shape)
        params = f.fit_nonlinear(x, y, 5)
        # Larger error since it is random.
        assert params['amplitude'] == pytest.approx(2, rel=2e-1)
        assert params['frequency'] == pytest.approx(3 / (2 * np.pi), rel=2e-1)
        assert params['phi'] == pytest.approx(1, rel=2e-1)
        assert params['constant'] == pytest.approx(0.5, rel=2e-1)

    def test_zero_crossings(self):
        """Test fitting with zero crossings"""
        x = np.linspace(0, 4 * np.pi, 100)
        y = 1.5 * np.sin(0.5 * x + 0.2) + 0.3
        params = f.fit_nonlinear(x, y, 5)
        assert params['amplitude'] == pytest.approx(1.5, rel=1e-1)
        assert params['frequency'] == pytest.approx(0.5 / (2 * np.pi), rel=1e-1)
        assert params['phi'] == pytest.approx(0.2, rel=1e-1)
        assert params['constant'] == pytest.approx(0.3, rel=1e-1)

    def test_invalid_input_type(self):
        """Test that non-numeric inputs raise TypeError"""
        with pytest.raises(TypeError):
            f.fit_nonlinear("invalid", [1, 2, 3], 2)

    def test_empty_data(self):
        """Test that empty data raises ValueError"""
        with pytest.raises(ValueError):
            f.fit_nonlinear([], [], 2)

class TestCalculateFrequencyAxis:
    """Test class for the calculate_frequency_axis function"""

    def test_basic_frequency_axis(self):
        """Test basic frequency axis calculation"""
        sample_rate = 10.0
        n_points = 8
        expected_freqs = [0.0, 1.25, 2.5, 3.75, -5.0, -3.75, -2.5, -1.25]
        freqs = f.calculate_frequency_axis(sample_rate, n_points)
        assert np.allclose(freqs, expected_freqs, rtol=1e-6)

    def test_odd_number_of_points(self):
        """Test frequency axis calculation with odd number of points"""
        sample_rate = 10.0
        n_points = 7
        expected_freqs = [0.0, 1.428571, 2.857143, 4.285714, -4.285714, -2.857143, -1.428571]
        freqs = f.calculate_frequency_axis(sample_rate, n_points)
        assert np.allclose(freqs, expected_freqs, rtol=1e-6)

    def test_invalid_n_points_type(self):
        """Test that non-integer n_points raises TypeError"""
        with pytest.raises(TypeError):
            f.calculate_frequency_axis(10.0, "8")

    def test_negative_n_points(self):
        """Test that negative n_points raises ValueError"""
        with pytest.raises(ValueError):
            f.calculate_frequency_axis(10.0, -8)

    def test_zero_n_points(self):
        """Test that zero n_points raises ValueError"""
        with pytest.raises(ValueError):
            f.calculate_frequency_axis(10.0, 0)

    def test_negative_sample_rate(self):
        """Test that negative sample_rate raises ValueError"""
        with pytest.raises(ValueError):
            f.calculate_frequency_axis(-10.0, 8)

    def test_zero_sample_rate(self):
        """Test that zero sample_rate raises ValueError"""
        with pytest.raises(ValueError):
            f.calculate_frequency_axis(0.0, 8)
class TestRotateToHorizontal:
    """Test class for the rotate_to_horizontal function"""

    def test_basic_rotation(self):
        """Test basic rotation to horizontal"""
        x = [0, 1, 2]
        y = [0, 1, 2]
        x_rot, y_rot = f.rotate_to_horizontal(x, y)
        assert np.allclose(x_rot, [0, 1, 2], rtol=1e-6)
        assert np.allclose(y_rot, [0, 0, 0], rtol=1e-6)

    def test_horizontal_line(self):
        """Test rotation of an already horizontal line"""
        x = [0, 1, 2]
        y = [1, 1, 1]
        x_rot, y_rot = f.rotate_to_horizontal(x, y)
        assert np.allclose(x_rot, [0, 1, 2], rtol=1e-6)
        assert np.allclose(y_rot, [0, 0, 0], rtol=1e-6)

    def test_vertical_line(self):
        """Test rotation of a vertical line"""
        x = [1, 1, 1]
        y = [0, 1, 2]
        x_rot, y_rot = f.rotate_to_horizontal(x, y)
        assert np.allclose(x_rot, [0, 0, 0], rtol=1e-6)
        assert np.allclose(y_rot, [0, 1, 2], rtol=1e-6)

    def test_negative_slope(self):
        """Test rotation of a line with negative slope"""
        x = [0, 1, 2]
        y = [2, 1, 0]
        x_rot, y_rot = f.rotate_to_horizontal(x, y)
        assert np.allclose(x_rot, [0, 1, 2], rtol=1e-6)
        assert np.allclose(y_rot, [0, 0, 0], rtol=1e-6)

    def test_random_points(self):
        """Test rotation of random points"""
        x = [1, 2, 3]
        y = [4, 5, 6]
        x_rot, y_rot = f.rotate_to_horizontal(x, y)
        assert np.allclose(x_rot, [0, 1, 2], rtol=1e-6)
        assert np.allclose(y_rot, [0, 0, 0], rtol=1e-6)

    def test_invalid_input_type(self):
        """Test that non-numeric inputs raise TypeError"""
        with pytest.raises(TypeError):
            f.rotate_to_horizontal("invalid", [1, 2, 3])
        with pytest.raises(TypeError):
            f.rotate_to_horizontal([1, 2, 3], "invalid")
