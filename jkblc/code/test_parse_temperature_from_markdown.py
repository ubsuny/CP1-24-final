"""Test unix conversions"""

import tempfile
from jkblc.code.final import parse_temperature_from_markdown

def test_parse_temperature_from_markdown():
    """
    Test the temperature parsing function from a markdown file.

    Creates a temporary markdown file with a sample temperature in Fahrenheit.
    Verifies if the function correctly extracts the temperature.
    """
    with tempfile.NamedTemporaryFile(mode='w+', suffix='.md', delete=False) as temp_file:
        temp_file.write(
            """
            # Experiment Data: Sine Wave Walk
            
            ## Metadata
            - **Experiment Name:** Sine Wave Walk
            - **Date and Time:** 2024-12-09 14:30:00
            - **Environment Temperature:** 75 F
            - **Experimenter ID:** LL008

            ## Experiment Details
            This experiment involved walking along a sine wave pattern while
              recording GPS data using the Phyphox app.
            """
        )
        temp_file_path = temp_file.name

    assert parse_temperature_from_markdown(temp_file_path) == 75
