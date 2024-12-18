'''
Module containing all importable functions
'''
import os

def fahrenheit_to_kelvin(temp_in_f):
    '''
    Function to convert temperature in Fahrenheit to Kelvin scale
    '''
    temp_in_k = ((temp_in_f - 32)*5/9) + 273.15
    return temp_in_k

def read_temperature(meta_file_path):
    '''
    Parser function to read the temperature of one markdown file
    '''
    with open(meta_file_path, 'r', encoding='utf-8') as meta:
        metadata = meta.readlines()

    temp_in_f = -10000 # default unphysical value
    for line in metadata:
        if line.startswith("Temperature ($\\text{\\textdegree}$F):"):
            temp_in_f = line.split(":")[1].strip()

    return temp_in_f

def filename_lister(directory, filename_filter, extension):
    '''
    Function to generate a list of files with the required extension
    containing the filename_filter
    '''
    file_list = []
    
    for filename in os.listdir(directory):
        if filename.endswith(extension) and filename_filter in filename:
            file_list.append(filename)
    
    return file_list

def non_linear_fit(ansatz, x, y, initial_guess, max_iter=1024, tol=1e-6):
    """
    Perform non-linear least squares fitting using a simple gradient-free iterative method.
   
    Parameters:
    - ansatz: Callable function, e.g., lambda x, a, b, c: a * np.exp(-b * x) + c
    - x: Independent variable data
    - y: Observed dependent variable data
    - initial_guess: Dictionary of initial guesses for the parameters
    - max_iter: Maximum number of iterations
    - tol: Tolerance for stopping criterion
   
    Returns:
    - best_params: Dictionary of best-fit parameters
    """
    params = initial_guess.copy()
    param_names = list(initial_guess.keys())
    step_size = 1e-4  # Small step for numerical gradient approximation
   
    for _ in range(max_iter):
        # Compute the residuals
        y_pred = ansatz(x, **params)
        residuals = y - y_pred
        rss = np.sum(residuals ** 2)  # Residual sum of squares
       
        # Approximate numerical gradient
        gradients = {}
        for param in param_names:
            params_step = params.copy()
            params_step[param] += step_size
            y_pred_step = ansatz(x, **params_step)
            rss_step = np.sum((y - y_pred_step) ** 2)
            gradients[param] = (rss_step - rss) / step_size
       
        # Update parameters
        params_new = {param: params[param] - step_size * gradients[param] for param in param_names}
       
        # Check for convergence
        max_change = max(abs(params_new[param] - params[param]) for param in param_names)
        if max_change < tol:
            break
       
        params = params_new
   
    return params
