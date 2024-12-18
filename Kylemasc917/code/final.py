import os
import math
import csv
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt


def fahrenheit_to_kelvin(fahrenheit):
    """Convert a temperature from Fahrenheit to Kelvin."""
    if fahrenheit < -459.67:
        raise ValueError("Temperature in Fahrenheit can't be lower than (-459.67°F).")
    kelvin = (fahrenheit + 459.67) * 5 / 9
    return kelvin

def extract_temp_from_markdown(file_path):
    """Search a markdown file for the temp in degrees F."""
    temperatures = []

    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            for line in file:
                words = line.split()
                for word in words:
                    if word.endswith('F'):
                        try:
                            temp = float(word.rstrip('°F'))
                            temperatures.append(temp)
                        except ValueError:
                            continue
    except FileNotFoundError:
        return f"File not found: {file_path}"

    return temperatures

def list_markdown_files(directory, filename_filter):
    """Markdown file lister from directory."""
    markdown_files = []

    try:
        for filename in os.listdir(directory):
            if filename.endswith('.md') and filename_filter in filename:
                markdown_files.append(filename)

    except FileNotFoundError:
        return f"Directory not found: {directory}"

    return markdown_files

def sine_wave(x_data, amplitude, frequency, phase, offset):
    """Sine wave function for fitting."""
    x_data = np.array(x_data)  # Ensure x is a numpy array for element-wise operations
    return amplitude * np.sin(2 * np.pi * frequency * x_data + phase) + offset

def nonlinear_fit(x_data, y_data, initial_guess):
    """Fit sine wave to data using nonlinear least squares fitting."""
    amplitude, frequency, phase, offset = curve_fit(sine_wave, x_data, y_data, p0=initial_guess)[0]
    return amplitude, frequency, phase, offset

def is_non_equidistant(x_data):
    """Check if the x_data points are non-equidistant."""
    diffs = np.diff(x_data)
    return not np.allclose(diffs, diffs[0])

def fft(data, step_number=10):
    """Wrapper for FFT, returns the frequency components."""
    if is_non_equidistant(data[0]):
        raise ValueError("Input data is non-equidistant. Please resample first.")

    num_samples = len(data)
    total_time = 2**step_number
    sampling_interval = total_time / num_samples
    fft_result = np.fft.fft(data)
    freqs = np.fft.fftfreq(num_samples, sampling_interval)

    return fft_result, freqs

def ifft(fft_data):
    """Wrapper for Inverse FFT."""
    return np.fft.ifft(fft_data)

def calculate_frequency_axis(x_data, total_time, f_unit='1/m'):
    """Calculate frequency axis in pure Python (no numpy), in inverse meters."""
    num_samples = len(x_data)
    fs = num_samples / total_time  # Spatial sampling frequency (number of samples per meter)
    freq_axis = [i * fs / num_samples for i in range(num_samples // 2)]
    
    if f_unit == '1/m':
        return freq_axis
    elif f_unit == 'k1/m':
        return [f / 1000 for f in freq_axis]
    else:
        raise ValueError(f"Unsupported frequency unit {f_unit}")

def load_csv_data(files):
    """Load data from multiple CSV files using DictReader to access by header."""
    data = []
    for file in files:
        x_values = []
        y_values = []
        with open(file, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    x_values.append(float(row['Latitude (°)']))
                    y_values.append(float(row['Longitude (°)']))
                except KeyError as e:
                    print(f"Column not found: {e}")
                except ValueError:
                    print(f"Invalid value in row: {row}")
        data.append((x_values, y_values))
    return data

def plot_fft_and_filtered_ifft(data, output_image_prefix):
    """Plot FFT, Inverse FFT, and Inverse FFT of the filtered frequency components."""
    
    # Plot FFTs
    plt.figure(figsize=(10, 6))
    for i, (x_data, y_data) in enumerate(data):
        # Shift the data to make sure the first point is the origin (0, 0)
        x_data_shifted = np.array(x_data) - x_data[0]
        y_data_shifted = np.array(y_data) - y_data[0]
        
        # Perform FFT
        y_fft = np.fft.fft(y_data_shifted)
        
        # Calculate frequency axis
        freqs = np.fft.fftfreq(len(y_data_shifted), d=(x_data_shifted[1] - x_data_shifted[0]))
        
        # Plot the FFT
        plt.plot(freqs, np.abs(y_fft), label=f"FFT of Data {i+1}")
    
    plt.title("FFT of All Data Files")
    plt.xlabel("Frequency (1/100m)")
    plt.ylabel("Amplitude")
    plt.legend()
    fft_image = f"{output_image_prefix}_fft.png"
    plt.tight_layout()
    plt.savefig(fft_image)
    print(f"FFT plot saved to {fft_image}")
    plt.close()

    # Plot Inverse FFTs
    plt.figure(figsize=(10, 6))
    for i, (x_data, y_data) in enumerate(data):
        # Shift the data to make sure the first point is the origin (0, 0)
        x_data_shifted = np.array(x_data) - x_data[0]
        y_data_shifted = np.array(y_data) - y_data[0]
        
        # Perform FFT
        y_fft = np.fft.fft(y_data_shifted)
        
        # Perform Inverse FFT
        y_ifft = np.fft.ifft(y_fft)
        
        # Plot the inverse FFT
        plt.plot(x_data_shifted, np.real(y_ifft), label=f"Inverse FFT of Data {i+1}")
    
    plt.title("Inverse FFT of All Data Files")
    plt.xlabel("X (Shifted to start at 0)")
    plt.ylabel("Y (Recovered from Inverse FFT)")
    plt.legend()
    ifft_image = f"{output_image_prefix}_ifft.png"
    plt.tight_layout()
    plt.savefig(ifft_image)
    print(f"Inverse FFT plot saved to {ifft_image}")
    plt.close()

    # Now plot the filtered Inverse FFT
    plt.figure(figsize=(10, 6))
    for i, (x_data, y_data) in enumerate(data):
        # Shift the data to make sure the first point is the origin (0, 0)
        x_data_shifted = np.array(x_data) - x_data[0]
        y_data_shifted = np.array(y_data) - y_data[0]

        # Perform FFT
        y_fft = np.fft.fft(y_data_shifted)

        # Calculate the frequency axis
        freqs = np.fft.fftfreq(len(y_data_shifted), d=(x_data_shifted[1] - x_data_shifted[0]))

        # Filter the FFT components to retain the mean value of the frequency components
        mean_freq = np.mean(np.abs(y_fft))
        threshold = 0.1  # Adjust this threshold based on how strict you want the filter to be
        filtered_fft = y_fft * (np.abs(y_fft) > threshold * mean_freq)

        # Perform Inverse FFT on the filtered data
        y_ifft_filtered = np.fft.ifft(filtered_fft)

        # Plot the result
        plt.plot(x_data_shifted, np.real(y_ifft_filtered), label=f"Filtered Inverse FFT of Data {i+1}")

    plt.title("Inverse FFT of Filtered Frequency Components")
    plt.xlabel("X (Shifted to start at 0)")
    plt.ylabel("Y (Recovered from Filtered Inverse FFT)")
    plt.legend()
    filtered_ifft_image = f"{output_image_prefix}_filtered_ifft.png"
    plt.tight_layout()
    plt.savefig(filtered_ifft_image)
    print(f"Filtered Inverse FFT plot saved to {filtered_ifft_image}")
    plt.close()

def process_files(files=None, initial_guess=[1, 1, 0, 0], output_image='fit_plot.png'):
    """Process CSV files, perform fitting, and plot fits onto a single PNG file."""
    if files is None:
        files = []

    data = load_csv_data(files)
    
    # Create a plot for the sine wave fitting
    plt.figure(figsize=(10, 6))
    
    # Loop through each file, fit the sine wave, and plot the result
    for i, (x_data, y_data) in enumerate(data):
        # Shift the data so the first point is the origin (0, 0)
        x_data_shifted = np.array(x_data) - x_data[0]
        y_data_shifted = np.array(y_data) - y_data[0]
        
        # Non-linear fitting
        amplitude, frequency, phase, offset = nonlinear_fit(x_data_shifted, y_data_shifted, initial_guess)
        print(f"Fitting parameters for file {files[i]}: amplitude={amplitude}, frequency={frequency}, phase={phase}, offset={offset}")
        
        # Generate the fitted data
        y_fit = sine_wave(x_data_shifted, amplitude, frequency, phase, offset)
        
        # Plot the shifted data and the fit
        plt.plot(x_data_shifted, y_data_shifted, label=f'Original Data {i+1}')
        plt.plot(x_data_shifted, y_fit, label=f'Fit {i+1}')
    
    plt.title("Sine Wave Fit for Each CSV File")
    plt.xlabel("X (Shifted to start at 0)")
    plt.ylabel("Y (Shifted to start at 0)")
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(output_image)
    print(f"Plot saved to {output_image}")
    plt.close()

    # Now plot the FFT, IFFT, and filtered IFFT
    plot_fft_and_filtered_ifft(data, output_image_prefix="fft_and_ifft")
