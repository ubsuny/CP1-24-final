"""
This module is to define preliminary functions to determine the direction of motion
"""

from libraries_module import np

# Function that calculates the unit vector (direction)
def unit_vector(vec):
    """
    Returns the unit vector pointing in the direction of the given vector

    Parameters:
    Inputs:
    vec (list): a list of the cartesian components of a vector
    Outputs:
    direction (list): a list of the components of the desired unit vector
    """
    vector = np.array(list(vec))
    vector_norm = np.linalg.norm(vector)
    direction = vector / vector_norm if vector_norm != 0 else np.zeros_like(vector)
    return direction

# Function that calculates the integrated set of data from another set
# ( assuming starting at zero )
def integrated_data(times, values):
    """
    Returns a list of data points that resembles
    the integration of the initial set of data

    Parameters:
    Inputs:
    times (array), values (array): times is the array of values
    against which the values array is integrated
    Outputs:
    integrated_values (array): the new set of integrated data
    """
    modified_times = np.array(times)
    modified_values = np.array(values)
    time_diffs = np.diff(modified_times)
    integrated_values = np.concatenate(([0], np.cumsum(modified_values[:-1] * time_diffs)))
    return integrated_values

# Function the calculates the average of a list of numbers
def average(ary):
    """
    Returns the average number of a list of numbers

    Parameters:
    Inputs:
    ary (array): the list of numbers desired to be averaged
    Outputs:
    avg (number): the desired average
    """
    arr = np.array(ary)
    avg = np.average(arr)
    return avg
