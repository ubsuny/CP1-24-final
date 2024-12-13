"""This is the unit test functions for Kylemasc917 final algorithm"""

from unittest import mock
import pytest
from final import fahrenheit_to_kelvin, extract_temp_from_markdown, list_markdown_files
from final import gradient_descent, compute_loss

def test_conversion():
    """Test standard conversions"""
    assert pytest.approx(fahrenheit_to_kelvin(32), 0.01) == 273.15
    assert pytest.approx(fahrenheit_to_kelvin(212), 0.01) == 373.15
    assert pytest.approx(fahrenheit_to_kelvin(-459.67), 0.01) == 0

def test_absolute_zero():
    """Test boundary condition at absolute zero"""
    assert fahrenheit_to_kelvin(-459.67) == 0

def test_below_absolute_zero():
    """Test input below absolute zero"""
    with pytest.raises(ValueError):
        fahrenheit_to_kelvin(-500)

def test_non_numeric_input():
    """Test non-numeric input handling"""
    with pytest.raises(TypeError):
        fahrenheit_to_kelvin("invalid input")

def test_extract_temp_no_temperatures():
    """Test case for a file containing no temperatures"""
    file_content = """This is a markdown file with no temperatures."""

    with mock.patch("builtins.open", mock.mock_open(read_data=file_content)):
        result = extract_temp_from_markdown("test_file.md")

    assert result == []

def test_extract_temp_invalid_format():
    """ Test case for invalid temperature format"""
    file_content = """This is a test with invalid temperature values.
    Invalid temp: 72A, another invalid: 85X."""

    with mock.patch("builtins.open", mock.mock_open(read_data=file_content)):
        result = extract_temp_from_markdown("test_file.md")

    assert result == []


def test_list_markdown_files_valid():
    """Test case for a valid directory with matching markdown files"""
    mock_files = ['experimentname_results.md', 'experimentname_data.md',
    'otherfile.txt', 'README.md']
    with mock.patch('os.listdir', return_value=mock_files):
        result = list_markdown_files('/mock/directory', 'experimentname')

    assert result == ['experimentname_results.md', 'experimentname_data.md']

def test_list_markdown_files_no_match():
    """Test case for a directory with no matching markdown files"""
    mock_files = ['experiment1_results.md', 'experiment2_data.md', 'README.md']
    with mock.patch('os.listdir', return_value=mock_files):
        result = list_markdown_files('/mock/directory', 'experimentname')

    assert result == []

def test_list_markdown_files_no_md_files():
    """Test case for a directory with no markdown files"""
    mock_files = ['otherfile.txt', 'image.jpg', 'README.md']
    with mock.patch('os.listdir', return_value=mock_files):
        result = list_markdown_files('/mock/directory', 'experimentname')

    assert result == []

def test_list_markdown_files_directory_not_found():
    """Test case for a non-existent directory"""
    with mock.patch('os.listdir', side_effect=FileNotFoundError):
        result = list_markdown_files('/mock/nonexistent_directory', 'experimentname')

    assert result == 'Directory not found: /mock/nonexistent_directory'

def test_gradient_descent_empty_data():
    """Test case for empty data"""
    x_data = []
    y_data = []

    a_init = 1.0
    b_init = 1.0
    n_steps = -2

    with mock.patch("builtins.print"):
        a_fit, b_fit = gradient_descent(x_data, y_data, a_init, b_init, n_steps)

    assert abs(a_fit - 1.0) < 0.1
    assert abs(b_fit - 1.0) < 0.1

def test_compute_loss():
    """Test case for correct computation of loss function"""
    x_data = [1, 2, 3, 4, 5]
    y_data = [2, 8, 18, 32, 50]

    a = 2.0
    b = 2.0

    loss = compute_loss(x_data, y_data, a, b)

    expected_loss = sum((y - a * x**b) ** 2 for x, y in zip(x_data, y_data))

    assert abs(loss - expected_loss) < 0.0001
