"""Reads out date and time form each metafile and converts to unix time"""

def parse_temperature_from_markdown(file_path):
    """
    Extract the environment temperature from a Markdown file.

    Parameters:
    file_path (str): Path to the Markdown file.

    Returns:
    float: Extracted temperature in Fahrenheit.

    Raises:
    ValueError: If the temperature is not found or formatted incorrectly.
    """
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            if "Environment Temperature:" in line:
                try:
                    # Clean Markdown formatting and extract temperature
                    temp_str = line.replace("*", "").split(":")[1].strip().replace("F", "").strip()
                    return float(temp_str)
                except (IndexError, ValueError) as exc:
                    raise ValueError("Invalid temperature format.") from exc

    raise ValueError("Temperature not found in file.")
