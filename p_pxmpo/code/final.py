"""
This module provides functionality for processing data from CSV and Markdown files. 
It includes utilities for reading columns, extracting temperature information, 
performing nonlinear sine fitting, and computing Fourier Transforms (FFT) and inverse FFTs. 

Key functionalities:
1. Read and process data from CSV files.
2. Extract and convert temperature information from Markdown files.
3. Perform nonlinear sine fitting on data.
4. Compute and visualize FFT and inverse FFT of data.
"""
import re
import numpy as np
import pandas as pd

def load_csv_data(file_path):
    """Load CSV data from the specified file."""
    try:
        data = pd.read_csv(file_path)
        return data['Time (s)'].values, data['Latitude (\u00b0)'].values
    except FileNotFoundError as exc:
        raise ValueError(f"File {file_path} not found.") from exc
    except pd.errors.EmptyDataError as exc:
        raise ValueError(f"File {file_path} is empty.") from exc
    except pd.errors.ParserError as exc:
        raise ValueError(f"Error parsing {file_path}.") from exc
    except Exception as exc:
        raise ValueError(f"Error loading {file_path}: {exc}") from exc


def sine_model_function(x, a, b, c):
    """A sine model function."""
    return a * np.sin(b * x) + c

def fit_curve(x, y, model_function):
    """Fit a sine curve using least squares method without using scipy."""

    # Initial guess for a, b, and c
    a_guess = np.max(y) - np.min(y)
    b_guess = 2 * np.pi / (x[-1] - x[0])
    c_guess = np.mean(y)

    def residuals(params):
        a, b, c = params
        return model_function(x, a, b, c) - y

    def sum_of_squares(params):
        return np.sum(residuals(params)**2)

    def gradient(params, epsilon=1e-5):
        grad = np.zeros_like(params)
        for i in range(len(params)):
            params_plus = params.copy()
            params_minus = params.copy()
            params_plus[i] += epsilon
            params_minus[i] -= epsilon
            grad[i] = (sum_of_squares(params_plus) - sum_of_squares(params_minus)) / (2 * epsilon)
        return grad

    # Gradient descent optimization
    params = np.array([a_guess, b_guess, c_guess])
    learning_rate = 1e-3
    max_iter = 1000
    tolerance = 1e-6

    for _ in range(max_iter):
        grad = gradient(params)
        params -= learning_rate * grad
        if np.linalg.norm(grad) < tolerance:
            break

    return params

def fahrenheit_to_kelvin(fahrenheit):
    """Convert Fahrenheit to Kelvin."""
    return (fahrenheit - 32) * 5/9 + 273.15

def extract_temperature_from_markdown(file_path, keyword=None):
    """Extract temperature data from markdown files."""
    temperature_data = []

    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()
            if keyword:
                matches = re.findall(rf'{keyword}:\s*(-?\d+(\.\d+)?)', content)
            else:
                matches = re.findall(r'Temperature:\s*(-?\d+(\.\d+)?)', content)

            for match in matches:
                temp = float(match[0])
                if 'F' in file_path:
                    temp = fahrenheit_to_kelvin(temp)
                temperature_data.append(temp)

    except FileNotFoundError as exc:
        raise FileNotFoundError(f"File {file_path} not found.") from exc
    except ValueError as e:
        print(f"Error converting temperature data in {file_path}: {e}")
    except re.error as e:
        print(f"Regex error while processing {file_path}: {e}")
    except OSError as e:
        print(f"OS error while accessing {file_path}: {e}")
    except Exception as e:
        # Log the exception for debugging, then re-raise to avoid silencing unexpected issues
        print(f"Unexpected error occurred while processing {file_path}: {e}")
        raise

    return temperature_data


def compute_fft(y_data, x_data):
    """
    Computes the Fast Fourier Transform (FFT) of y_data.
    Args:
        y_data (ndarray): Data values to transform.
        x_data (ndarray): Corresponding x-axis values.
    Returns:
        tuple: Frequencies (scaled), magnitudes, and FFT result (complex numbers).
    """
    fft_result = np.fft.fft(y_data)
    fft_freq = np.fft.fftfreq(len(x_data), x_data[1] - x_data[0])
    fft_freq_scaled = fft_freq * 100
    fft_magnitude = np.abs(fft_result)
    return fft_freq_scaled, fft_magnitude, fft_result

def compute_inverse_fft(fft_result, filter_mask):
    """
    Computes the inverse FFT of filtered FFT data.

    Args:
        fft_result (ndarray): Original FFT result.
        filter_mask (ndarray): Boolean mask to apply as a filter.

    Returns:
        ndarray: Real part of the inverse FFT result after applying the filter.
    """
    filtered_fft = fft_result * filter_mask
    inverse_fft = np.fft.ifft(filtered_fft)
    return np.real(inverse_fft)
