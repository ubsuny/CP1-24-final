"""
the final.py module includes importable functions
designed for different tasks relating to the final
"""
import os
import numpy as np
import distance_calc as dc

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

def file_list(path, my_filter):
    """
    the md_list function goes through all folders
    within a directory defined by path to 
    produce a list of all the markdown files.
    """
    md=[]
    files=os.listdir(path)
    for file in files:
        if my_filter in file:
            md.append(file)
    folders=[item for item in files if os.path.isdir(os.path.join(path, item))]
    for folder in folders:
        new_path=path+"/"+folder
        new_md=file_list(new_path, my_filter)
        for m in new_md:
            md.append(m)
    return md

def model(x, a, b, c):
    """
    model creates a model sinwave
    as a function of x with parameters
    a,b,c
    """
    return a * np.sin(b * x + c)


def residuals(params, x, y):
    """
    residuals calculates the difference between 
    observed y values and predicted y from the model
    """
    a, b, c = params
    return y - model(x, a, b, c)


def jacobian(x, params):
    """
    jacobian calculates the Jacobian of the residual 
    function with respect to a,b,c
    """
    a, b, c = params
    jacobian_matrix = np.zeros((len(x), 3))

    # Partial derivatives
    jacobian_matrix[:, 0] = -np.sin(b * x + c)  # Partial derivative w.r.t A
    jacobian_matrix[:, 1] = -a * x * np.cos(b * x + c)  # Partial derivative w.r.t B
    jacobian_matrix[:, 2] = -a * np.cos(b * x + c)  # Partial derivative w.r.t C

    return jacobian_matrix


def gauss_newton(x, y, initial_params, n, max_iter=100, tolerance=1e-6):
    """
    The gauss_newton function implements the
    Gauss-Newton method for nonlinear least 
    squares. 
    """
    step=2**n
    params = np.array(initial_params, dtype=float)
    for iteration in range(max_iter):
        # Calculate residuals and Jacobian
        res = residuals(params, x, y)
        jac = jacobian(x, params)

        # Compute the normal equation: (J^T * J) * delta = J^T * residuals
        jtj = np.dot(jac.T, jac)  # J^T * J
        jt_res = np.dot(jac.T, res)  # J^T * residuals

        # Solve for delta (change in parameters)
        delta = np.linalg.solve(jtj, jt_res)

        # Update parameters
        params -= delta

        # Check for convergence: if the change is small enough, stop
        if np.linalg.norm(delta) < tolerance:
            print(f"Converged after {iteration + 1} iterations")
            break
    new_x=np.linspace(0,40,step, endpoint=False)
    new_func=model(np.array(new_x), params[0], params[1], params[2])

    return params, new_func, new_x

def get_sin_data(path):
    """
    get_sin_data takes in a path to acquire the positional data
    from that file.
    """
    lats,lons=dc.reader(path)
    distances=[]
    xy_coords=[]
    x_pos=[0]
    y_pos=[0]
    for i in range(1,len(lats)-1):
        distances.append(dc.diffm(lats[i], lats[i+1], lons[i], lons[i+1]))

    for i, v in enumerate(distances):
        xy_coords.append(dc.get_coords(distances, v, i))
        y,x=xy_coords[i]
        x_pos.append(x)
        y_pos.append(y)
    return x_pos,y_pos

def subtract_ave(y):
    """
    subtract_ave subtracts the 
    average value from the y-data
    to ensure that it is centered
    about 0.
    """
    my_sum=0
    for i in y:
        my_sum+=i
    return np.array(y)-my_sum/len(y)

def wrap_fft(x,y, inverse):
    """
    wrap_fft either conducts an 
    fft or ifft depending on whether or 
    not the inverse parameter is true.
    Also checks for equidistant data
    """
    condition=False
    dif=round(x[1]-x[0],6)
    for i, val in enumerate(x):
        if i!=len(x)-1:
            print((x[i+1]-val), dif)
            if condition==(np.isclose((x[i+1]-val), dif,rtol=1e-6)):
                condition=True
    if condition is True:
        print("Error: Data is not equidistant")
        return
    if inverse is True:
        fdata=np.fft.ifft(y)

    fdata=np.fft.fft(y)

    return fdata

def get_params(name):
    """
    get_params acquires the needed parameters to conduct
    a good nonlinear fit from the parameters file for the
    sin wave data from the file which has a name defined by the
    name parameter.
    """
    p1,p2,p3=0,0,0
    fn=name.rstrip(".csv")
    try:
        with open("/workspaces/CP1-24-final/abruns123/data/"/
                  "sin_data/plots/parameters.md", "r",encoding='utf-8') as file:
            lines=[line.strip() for line in file.readlines()]
    except FileNotFoundError:
        print(f"Error: The file '{name}' does not exist.")
    for line in lines:
        words=line.split()
        if words[0]==(fn+":"):
            p1=float(words[1])
            p2=float(words[2])
            p3=eval(words[3])

    return p1,p2,p3

def get_frequency_axis(x, n):
    """
    get_frequency_axis manually determines the 
    frequency axis of a given list of x values
    of length 2^n. units are in 1/m such that
    they may be modified outside the function.
    """
    num=2**n
    length=x[len(x)-1]
    sample_rate=num/length

    frequencies=np.linspace(-sample_rate/2, sample_rate/2, num, endpoint=False)
    return frequencies

def get_frequency(x,y):
    """
    get_frequency finds the frequency and magnitude
    from filtered fft data
    """
    maxi=0
    freq=0
    for i, val in enumerate(x):
        if y[i]>maxi:
            maxi=y[i]
            freq=val
    return freq, maxi
