
"""
test_final.py

Tests the final.py module's functionality.
"""
import tempfile
import os
import pytest
import numpy as np
import pandas as pd

# Task 1 imports
from final import convert_f_to_k
# Task 2 imports
from final import parse_markdown
# Task 3 imports
from final import filter_markdown_files
from final import sort_filenames
# Task 4 imports
from final import degrees_to_meters, resample_to_2n_segments
from final import sine_fit, check_equidistant_x
from final import rotate_to_x_axis, prepare_and_fit

# Task 4 imports
from final import apply_fft, apply_ifft, apply_fft_to_arrays


# Task 1  tests
def test_convert_f_to_k():
    """
    Test the convert_f_to_k function with multiple cases.

    Ensures that the function correctly converts temperatures
    from Fahrenheit to Kelvin using known inputs and expected outputs.

    Test Cases:
        - Freezing point of water (32°F -> 273.15K)
        - Boiling point of water (212°F -> 373.15K)
        - Absolute zero (-459.67°F -> 0K)
        - Arbitrary values for additional checks
    """
    # Test Cases: Input Fahrenheit, Expected Kelvin
    test_cases = [
        (32, 273.15),   # Freezing point of water
        (212, 373.15),  # Boiling point of water
        (-459.67, 0),   # Absolute zero
        (0, 255.372),   # Arbitrary test case
        (100, 310.928)  # Another test case
    ]

    # Tolerance for floating-point comparisons. Might be necessary.
    tolerance = 1e-3

    # Test each case.
    for fahrenheit, expected_kelvin in test_cases:
        # Invoke the function
        result = convert_f_to_k(fahrenheit)
        # Assert truth.
        assert abs(result - expected_kelvin) < tolerance, f"Failed for input: {fahrenheit}F"

# Task 2  tests
def create_temp_markdown_file(content):
    """
    Helper function to create a temporary markdown file.

    Parameters:
        content (str): The content to write into the file.

    Returns:
        str: The file path of the temporary file.
    """
    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".md") as temp_file:
        temp_file.write(content)
        return temp_file.name

def test_parse_markdown():
    """
    Test the parse_markdown function for extracting a specific field's value.
    """

    # Define test cases: content -> field_name -> expected result
    test_cases = [
        # Valid cases
        ("##Temperature\n33 F", "Temperature", 33.0),  # Basic valid case
        ("##Temperature\n-10 F", "Temperature", -10.0),  # Negative temperature
        ("##Temperature\n45.5 F", "Temperature", 45.5),  # Decimal temperature
        ("##Humidity\n75 F", "Humidity", 75.0),  # Extracting Humidity field
        ("##Pressure\n1013 F", "Pressure", 1013.0),  # Extracting Pressure field

        # Invalid cases
        ("##Temperature\nInvalid F", "Temperature", None),  # Non-numeric temperature
        ("##Temperature\n33", "Temperature", None),  # Missing 'F'
        ("##Temperature", "Temperature", None),  # No value after header
        ("No temperature here", "Temperature", None),  # Missing header
        ("##temperature\n33 F", "Temperature", None),  # Case-sensitive check
    ]

    for content, field_name, expected_result in test_cases:
        # Create a temporary markdown file
        temp_file_path = create_temp_markdown_file(content)

        try:
            # Run the function
            result = parse_markdown(temp_file_path, field_name)

            # Assert the result matches the expectation
            assert result == expected_result, (
                f"Failed for field: '{field_name}', content:\n{content}\n"
                f"Expected {expected_result}, got {result}."
            )
        finally:
            # Clean up: Delete the temporary file
            os.remove(temp_file_path)

# Task 3  tests
def test_filter_markdown_files():
    """
    Test the filter_markdown_files function.
    """
    # create a temp directory with test markdown files
    with tempfile.TemporaryDirectory() as temp_dir:
        filenames = [
            "AVG001_gps_sine_walk_0.md",
            "AVG002_gps_sine_walk_1.md",
            "README.md",
            "NOTES_summary.md",
        ]

        # Create the files in the temp directory
        for filename in filenames:
            with open(os.path.join(temp_dir, filename), 'w', encoding="utf-8") as temp_file:
                temp_file.close()
            # open(os.path.join(temp_dir, filename), 'w').close()

        # Test filtering with 'sine_walk' string
        result = filter_markdown_files(temp_dir, "sine_walk")
        assert sorted(result) == ["AVG001_gps_sine_walk_0.md", "AVG002_gps_sine_walk_1.md"]

        # Test filtering with 'NOTES'
        result = filter_markdown_files(temp_dir, "NOTES")
        assert result == ["NOTES_summary.md"]

        # Test filtering with a non-matching string
        result = filter_markdown_files(temp_dir, "nonexistent")
        assert result == []

def test_sort_filenames_with_non_numeric_suffix():
    """
    Test that sort_filenames handles filenames with non-numeric suffixes correctly.
    """
    filenames = [
        "file_10.md",
        "file_2.md",
        "file_final.md",
        "file_1.md",
    ]
    expected_output = [
        "file_1.md",
        "file_2.md",
        "file_10.md",
        "file_final.md",  # Non-numeric filename appears last
    ]
    assert sort_filenames(filenames) == expected_output

def test_sort_filenames():
    """
    Test that sort_filenames sorts filenames based on their trailing numeric values.
    """
    # Sample unsorted filenames
    filenames = [
        "file_10.md",
        "file_2.md",
        "file_1.md",
        "file_20.md",
    ]
    # Expected sorted order
    expected_output = [
        "file_1.md",
        "file_2.md",
        "file_10.md",
        "file_20.md",
    ]
    # Test sorting
    assert sort_filenames(filenames) == expected_output

    # Edge case: empty list
    assert sort_filenames([]) == []

    # Edge case: single file
    assert sort_filenames(["file_42.md"]) == ["file_42.md"]

# Task 4  tests
# Sample DataFrame (replace with your actual data)
sample_data = {
    'Time': [0, 1, 2, 3, 4, 5],
    'Latitude (°)': [37.7749, 37.7751, 37.7753, 37.7755, 37.7757, 37.7759],
    'Longitude (°)': [-122.4194, -122.4192, -122.4190, -122.4188, -122.4186, -122.4184]
}

def test_degrees_to_meters():
    """
    Test the degrees_to_meters function for correct conversion of lat/lon to meters.
    """
    lat = np.array([0.0, 1.0, 2.0])  # Degrees latitude
    lon = np.array([0.0, 1.0, 2.0])  # Degrees longitude
    x, y = degrees_to_meters(lat, lon)

    assert x.shape == lon.shape
    assert y.shape == lat.shape
    assert np.isclose(x[0], 0.0)  # First point must be the origin
    assert np.isclose(y[0], 0.0)
    assert y[1] > 0  # Latitude increases linearly
    assert x[1] > 0  # Longitude increases after scaling

def test_resample_to_2n_segments():
    """
    Test the resample_to_2n_segments function for correct interpolation.
    """
    data = pd.DataFrame({
        'Time': [0, 10, 20],
        'Latitude (\u00b0)': [0.0, 10.0, 20.0],
        'Longitude (\u00b0)': [0.0, 20.0, 40.0]
    })
    n = 3  # Resample to 2^3 = 8 points
    resampled_data = resample_to_2n_segments(data, n)

    assert resampled_data.shape[0] == 2**n
    assert 'Time' in resampled_data.columns
    assert np.isclose(resampled_data['Latitude (\u00b0)'].iloc[0], 0.0)
    assert np.isclose(resampled_data['Longitude (\u00b0)'].iloc[-1], 40.0)

def test_sine_fit_with_dataframe():
    """Test sine_fit function with a sample DataFrame."""
    data = {
        'Time': [4.863376, 6.355408, 7.847441, 9.339474, 10.831506, 92.893296,
            94.385329, 95.877361, 97.369394, 98.861426],
        'x': [0.000000, 0.429122, 1.000524, 1.718506, 2.515790, 69.592420,
            69.416795, 69.121662, 68.896881, 68.860733],
        'y': [0.000000, 1.641653, 3.060481, 4.544753, 6.013064, 0.099075,
            2.126554, 4.153333, 6.029915, 7.674301]
    }
    expected_results = {
        'amplitude': 3.8,  # Adjust this value based on your expected fit
        'frequency': 0.1,   # Adjust this value based on your expected fit
    }
    df = pd.DataFrame(data)
    result = sine_fit(df['x'].to_numpy(), df['y'].to_numpy())

    # Adjust tolerance as needed
    assert np.isclose(result['amplitude'], expected_results['amplitude'], rtol=1)
    assert np.isclose(result['frequency'], expected_results['frequency'], rtol=1)

def test_check_equidistant_x():
    """
    Test the check_equidistant_x function for verifying equidistant x values.
    """
    x_equidistant = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    x_non_equidistant = np.array([0.0, 1.0, 2.1, 3.0, 4.0])  # Perturbation at index 2

    # Test equidistant array with a reasonable tolerance (should return True)
    assert check_equidistant_x(x_equidistant, tolerance=1)

    # Test non-equidistant array with a reasonable tolerance (should return False)
    assert not check_equidistant_x(x_non_equidistant,
        tolerance=0.1), "Test failed for non-equidistant values"

    # Slight perturbation within tolerance (should still return True)
    x_perturbed = np.array([0.0, 1.0, 2.000001, 3.0, 4.0])
    assert check_equidistant_x(x_perturbed, tolerance=1e-5)

    # Slight perturbation beyond tolerance (should return False)
    assert not check_equidistant_x(x_perturbed,
        tolerance=1e-7), "Test failed for perturbed values with small tolerance"

    # Testing large tolerance (should return True)
    x_large_tolerance = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    assert check_equidistant_x(x_large_tolerance, tolerance=10)

    # Testing very small tolerance where values are clearly not equidistant
    x_small_tolerance = np.array([0.0, 1.0, 2.1, 3.0, 4.0])
    assert not check_equidistant_x(x_small_tolerance,
        tolerance=0.05), "Test failed for small tolerance with non-equidistant values"

    # Extreme case where values are slightly off
    x_extreme_perturbation = np.array([0.0, 1.0, 2.0000001, 3.0, 4.0])
    assert check_equidistant_x(x_extreme_perturbation, tolerance=1e-7)

def test_rotate_to_x_axis_resample():
    """
    Test rotate_to_x_axis with resampling.
    """
    df = pd.DataFrame(sample_data)
    rotated_df = rotate_to_x_axis(df, resample=True, n=3)
    assert len(rotated_df) == 2**3  # Check if resampling to 2^n points occurred

def test_rotate_to_x_axis_no_resample():
    """
    Test rotate_to_x_axis without resampling.
    """
    df = pd.DataFrame(sample_data)
    rotated_df = rotate_to_x_axis(df, resample=False)
    assert len(rotated_df) == len(df)  # Check if no resampling occurred

def test_prepare_and_fit_resample():
    """
    Test prepare_and_fit with resampling.
    """
    df = pd.DataFrame(sample_data)
    refit_data = prepare_and_fit(df, resample=True)
    assert len(refit_data) == 2**6  # Check if resampling occurred
    assert 'Fitted Sine' in refit_data.columns  # Check if sine curve fitting was performed

def test_prepare_and_fit_no_resample():
    """
    Test prepare_and_fit without resampling.
    """
    df = pd.DataFrame(sample_data)
    refit_data = prepare_and_fit(df, resample=False)
    assert len(refit_data) == len(df)  # Check if no resampling occurred
    assert 'Fitted Sine' in refit_data.columns  # Check if sine curve fitting was performed

# Task 5  tests
def create_sample_data(n_points=128):
    """
    Creates a sample DataFrame with equidistant x values and a simple sine wave.

    Args:
        n_points (int, optional): Number of data points. Defaults to 100.

    Returns:
        pd.DataFrame: DataFrame with 'x' and 'y' columns.
    """
    x = np.linspace(0, 100, n_points)
    y = np.sin(2 * np.pi * x)
    return pd.DataFrame({'x': x, 'y': y})

def test_apply_fft():
    """
    Test the apply_fft function.
    """
    data = create_sample_data()
    frequencies, fft_values = apply_fft(data)
    assert len(frequencies) == len(fft_values)
    assert np.all(np.isfinite(frequencies))
    assert np.all(np.isfinite(fft_values))

def test_apply_ifft():
    """
    Test the apply_ifft function.
    """
    # Create a test signal (simple sine wave)
    x = np.linspace(0, 2 * np.pi, 100)
    y = np.sin(x)

    # Apply FFT and IFFT
    fft_frequencies, fft_values = apply_fft_to_arrays(x, y)
    reconstructed_y = apply_ifft(fft_values)
    print(f"FFT Frequency array length {len(fft_frequencies)}")

    # Debugging: Print mismatched values if any
    if not np.allclose(reconstructed_y, y, atol=1):
        print("Original y:", y[:10])
        print("Reconstructed y:", reconstructed_y[:10])

    # Verify that reconstructed signal matches the original within tolerance
    assert len(reconstructed_y) == len(y)
    assert np.allclose(reconstructed_y, y, atol=1)

# Pytest main hook
if __name__ == "__main__":
    pytest.main()
