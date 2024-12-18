"""
the final.py module includes importable functions
designed for different tasks relating to the final
"""
import os
import numpy as np
import distance_calc as dc
import matplotlib.pyplot as plt

def f_k(t):
    """
    f_k takes in a temperature value t and 
    returns that temperature in kelvin
    """
    return (t-32)*5/9+273.15

def parse(path):
    """
    parse takes in a file path as a string and 
    parses through the file. If it finds 
    Temperature: as the first word of a line,
    it will define the next string as the temperature.
    parse then returns this temperature.
    """
    lines=[]
    temp=0
    condition=False
    try:
        with open(path, "r",encoding='utf-8') as file:
            lines=[line.strip() for line in file.readlines()]
    except FileNotFoundError:
        print(f"Error: The file '{path}' does not exist.")
    for line in lines:
        words=line.split()
        if words[0]=="Temperature:":
            temp=int(words[1])
            condition=True
    if condition is True:
        return temp

    return "No temperature in this file."

def file_list(path, filter):
    """
    the md_list function goes through all folders
    within a directory defined by path to 
    produce a list of all the markdown files.
    """
    md=[]
    files=os.listdir(path)
    for file in files:
        if file.endswith(filter):
            md.append(file)
    folders=[item for item in files if os.path.isdir(os.path.join(path, item))]
    for folder in folders:
        new_path=path+"/"+folder
        new_md=md_list(new_path)
        for m in new_md:
            md.append(m)
    return md

def model(x, A, B, C):
    return A * np.sin(B * x + C)

# Define the residuals function: difference between observed y values and predicted y from the model
def residuals(params, x, y):
    A, B, C = params
    return y - model(x, A, B, C)

# Jacobian of the residuals function with respect to A, B, C
def jacobian(x, params):
    A, B, C = params
    jacobian_matrix = np.zeros((len(x), 3))

    # Partial derivatives
    jacobian_matrix[:, 0] = -np.sin(B * x + C)  # Partial derivative w.r.t A
    jacobian_matrix[:, 1] = -A * x * np.cos(B * x + C)  # Partial derivative w.r.t B
    jacobian_matrix[:, 2] = -A * np.cos(B * x + C)  # Partial derivative w.r.t C

    return jacobian_matrix

# Implement Gauss-Newton method for nonlinear least squares
def gauss_newton(x, y, initial_params, n, max_iter=100, tolerance=1e-6):
    step=2**n
    params = np.array(initial_params, dtype=float)
    for iteration in range(max_iter):
        # Calculate residuals and Jacobian
        res = residuals(params, x, y)
        jac = jacobian(x, params)

        # Compute the normal equation: (J^T * J) * delta = J^T * residuals
        JtJ = np.dot(jac.T, jac)  # J^T * J
        Jt_res = np.dot(jac.T, res)  # J^T * residuals
        
        # Solve for delta (change in parameters)
        delta = np.linalg.solve(JtJ, Jt_res)

        # Update parameters
        params -= delta
        
        # Check for convergence: if the change is small enough, stop
        if np.linalg.norm(delta) < tolerance:
            print(f"Converged after {iteration + 1} iterations")
            break

    new_x=np.linspace(x[0], x[len(x)-1], step)
    new_func=model(new_x, params[0], params[1], params[2])

    return params, new_func, new_x

def get_sin_data(path):
    lats,lons=dc.reader(path)
    distances=[]
    xy_coords=[]
    x_pos=[0]
    y_pos=[0]
    for i in range(1,len(lats)-1):
        distances.append(dc.diffm(lats[i], lats[i+1], lons[i], lons[i+1]))

    for i in range(len(distances)):
        xy_coords.append(dc.get_coords(distances, distances[i], i))
        y,x=xy_coords[i]
        x_pos.append(x)
        y_pos.append(y)
    return x_pos,y_pos

def subtract_ave(y):
    sum=0
    for i in y:
        sum+=i
    return np.array(y)-sum/len(y)

def wrap_fft(x,y):
    
    fdata=np.fft.fft(y)
    frequencies=np.fft.fftfreq(len(y), x[len(x)-1]/len(x))
    return fdata, frequencies

def get_params(name):
    fn=name.rstrip(".csv")
    try:
        with open("/workspaces/CP1-24-final/abruns123/data/parameters.md", "r",encoding='utf-8') as file:
            lines=[line.strip() for line in file.readlines()]
    except FileNotFoundError:
        print(f"Error: The file '{path}' does not exist.")
    for line in lines:
        words=line.split()
        if words[0]==(fn+":"):
            p1=float(words[1])
            p2=float(words[2])
            p3=eval(words[3])
    return p1,p2,p3






