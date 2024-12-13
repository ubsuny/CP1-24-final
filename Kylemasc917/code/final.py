"""This is the algorithm file for Kylemasc917 final project"""

import os
import math

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


def power_law(x, a, b):
    """Model function: y = a * x^b"""
    return a * x**b

def compute_loss(x_data, y_data, a, b):
    """Compute the sum of squared errors (loss function)."""
    loss = 0.0
    for x, y in zip(x_data, y_data):
        y_pred = power_law(x, a, b)
        loss += (y - y_pred)**2
    return loss

def compute_gradients(x_data, y_data, a, b):
    """Compute gradients of loss with respect to a and b."""
    grad_a = 0.0
    grad_b = 0.0

    for x, y in zip(x_data, y_data):
        y_pred = power_law(x, a, b)
        grad_a += -2 * (y - y_pred) * x**b
        grad_b += -2 * (y - y_pred) * a * x**b * math.log(x)

    return grad_a, grad_b

def gradient_descent(x_data, y_data, a_init, b_init, n_steps):
    """Perform gradient descent to fit the power law model."""
    a = a_init
    b = b_init

    # Step number 2^n (step size multiplier)
    step_size = 2 ** n_steps

    for step in range(1000):  # Maximum number of iterations
        grad_a, grad_b = compute_gradients(x_data, y_data, a, b)

        # Update parameters
        a -= step_size * grad_a
        b -= step_size * grad_b

        # Print progress every 100 iterations
        if step % 100 == 0:
            loss = compute_loss(x_data, y_data, a, b)
            print(f"Step {step}, Loss: {loss:.4f}, a: {a:.4f}, b: {b:.4f}")

    return a, b
