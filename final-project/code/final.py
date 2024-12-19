def fahrenheit_to_kelvin(fahrenheit):
    """
    Convert Fahrenheit to Kelvin.

    Parameters:
        fahrenheit (float): Temperature in Fahrenheit.

    Returns:
        float: Temperature in Kelvin.
    """
    kelvin = (fahrenheit - 32) * 5/9 + 273.15
    return kelvin
