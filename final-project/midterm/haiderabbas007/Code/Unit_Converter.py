"""
This module provides a function to convert lengths in yards or feet to meters.

Function included:

1. convert_to_meters(val, unit):
   - Takes in lengths in yards or feet and converts them to meters.
   - Supports both 'ft' (feet) and 'yd' (yards) units, and is case-insensitive.
"""

def unit_converter(val, unit):
  """
  Takes in lengths in yards or feet and converts to meters.

  Parameters:
      val (float): magnitude of the length.
      unit (str): unit of length ('ft' or 'yd').

  Returns:
      float: Length converted to meters.
  """
  if val < 0:
    raise ValueError("Only non-negative lengths please.")  # Negative lengths ruled out.
  if unit.lower() == 'ft':  # Accounts for 'Ft'.
    val = val * 0.3048  # From Google Unit Converter.
    return val
  elif unit.lower() == 'yd':  # Accounts for 'Yd'.
    val = val * 0.9144  # From Google Unit Converter.
    return val
  else:
    raise ValueError("Only 'ft' or 'yd' please.")
