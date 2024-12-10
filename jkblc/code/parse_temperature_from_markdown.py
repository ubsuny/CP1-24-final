"""Reads out date and time form each metafile and converts to unix time"""

def parse_temperature_from_markdown(file_path):
    """
    Extract the environment temperature from a markdown file.

    The function searches for the line starting with 'Environment Temperature:'
    followed by a temperature value in Fahrenheit.

    Parameters:
    file_path (str): Path to the markdown file.

    Returns:
    float: Extracted temperature in Fahrenheit.

    Raises:
    ValueError: If temperature data is not found in the file.
    """
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            # Find the line that starts with "Environment Temperature:"
            if "Environment Temperature:" in line:
                try:
                    # Extract the temperature value by splitting the line
                    parts = line.split(":")
                    if len(parts) >= 2:
                        temp_str = parts[1].strip().replace("F", "").strip()
                        return float(temp_str)
                except ValueError as exc:
                    raise ValueError("Invalid temperature format.") from exc
    raise ValueError("Temperature not found in file.")
