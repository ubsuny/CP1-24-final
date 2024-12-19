"""
final.py
this is the location of the code that is needed for the final
the pylint score has a value of 9.92/10 since it is refrencing
something that will only complicate the function so i decided
it would be bes not to change
"""
import re
import os
import glob
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
import workinh as wrk

def fah_to_kel(f):
    """Converts from Fahrenheit to Kelvin enter the recorded temp as f"""
    if f < -459.66:
        print("The value for Fahrenheit should not be possible try again")
        return None
    temp = (5*(f-32))/9 + 273.15
    return round(temp, 2)

def find_fah(filpath):
    """this function finds the temp recorded for the file you enter in file_name
    ie if you wanted to know the temp recorded in the test file LL13_sinewalktest.md, 
    you would enter 'LL13_sinewalktest.md' for file_name"""
    filpath = os.path.join(filpath.strip())
    with open(filpath, 'r', encoding='utf-8') as f:
        numbs = re.findall(r'\d+', f.read())
    return int(numbs[0])

def filenamelister(exp_name, filetype = '.md'):
    """this function finds and returns all markdown files of an experiment type
    ie if you wanted to find all .md files that relate to the total elevator
    movment experiment enter 'eletot' for exp_name"""
    pattern = os.path.join('/workspaces/CP1-24-final/zbpetersbuf/data/',
                            f"*{exp_name.strip()}*{filetype.strip()}")
    files = glob.glob(pattern)
    return files

def findmdfromcsv(filepath):
    """
    Extracts and returns a portion of the filename from a given CSV file path.

    The function expects the file path to be in the format 'LL<digits>_<file_name>.csv'.
    It extracts the part after 'LL<digits>_' and before the '.csv' extension, 
    returning it prefixed with an underscore ('_').

    Parameters:
    filepath (str): The path to the CSV file.

    Returns:
    str: The extracted portion of the filename prefixed with an underscore.
    """
    match = re.search(r'LL\d+_(.*)\.csv', filepath)
    file_name = match.group(1)
    return f"_{file_name}"

def sinfunk(x, a, b, c, d):
    """this is just the fit function for the fit stuff"""
    return a * np.sin(b*x - b*c) + d*x

def fitsincuve(xax, yax, i=0):
    """
    Fits a sinusoidal function to the provided data points.

    This function attempts to fit a sine wave model to the given data using the 
    `curve_fit` function from `scipy.optimize`. It initializes the parameters 
    with a rough guess and then refines the fit using nonlinear least squares.

    Parameters:
    xax (array-like): The independent variable (x-axis) data points.
    yax (array-like): The dependent variable (y-axis) data points.
    i (int, optional): An optional index parameter (default is 0), which is currently unused 
                       except for printing the input data if the lengths of xax and yax don't match.

    Returns:
    array: The fitted sinusoidal values based on the input data.
    """
    if not len(xax) == len(yax):
        print(xax, yax, i)
    yax = np.array(yax)
    guess_a = (np.abs(np.max(yax) - np.min(yax)))
    guess_b = 4 * np.pi/(xax[-1] - xax[0])
    guess_c = 1
    guess_d = (xax[0]-xax[len(xax)-1])/((yax[0]-yax[len(xax)-1]))*(sum(yax)/sum(xax))
    p0 = [guess_a, guess_b, guess_c, guess_d]
    a, b, c, d = curve_fit(sinfunk, xax, yax, p0=p0, maxfev=50000)[0]
    sin_fit = sinfunk(np.array(xax), a, b, c, d)
    return sin_fit

def stepnumb(datta):
    """
    Shortens the length of the data to the closest power of 2 and adjusts the data accordingly.

    This function ensures that the data (time, x-axis, and y-axis) have matching
    lengths and are evenly spaced. If the lengths are inconsistent or if data points are
    missing, an error message is printed, and the function returns `None`. The data is then
    trimmed to the closest power of 2 (2^n), removing points from both ends as necessary. The
    data is returned after shifting the x-axis and y-axis values to start from 0,
    based on the smallest time value.

    Parameters:
    datta (pandas DataFrame): A DataFrame containing at least the 'Time (s)' column and data for
    x and y axes.

    Returns:
    tuple: A tuple containing the adjusted x-axis, y-axis, and time data after trimming and
    shifting.

    Example:"""

    tim = list(datta.loc[:, 'Time (s)'])
    xax, yax = wrk.gpsloc(datta)
    compare = [len(tim)==len(xax), len(tim)==len(yax), len(xax)==len(yax)]
    alcomp = np.all(compare)
    if not alcomp:
        print("Data is not evenly spaced or data points are missing")
        return None
    n = 0
    while len(tim) > 2**n:
        n+=1
    twon = 2**(n-1)
    remv = (len(tim) - twon) // 2
    remv1 = remv
    if abs(remv - (len(tim) - twon) / 2) == 0.5:
        remv1 = remv + 1
    xax = xax[remv1:-remv]
    yax = yax[remv1:-remv]
    tim = tim[remv1:-remv]
    mintim = np.min([tim[0],tim[len(tim)-1]])
    if mintim == tim[0]:
        return xax-xax[0], yax-yax[0], tim
    return xax-xax[len(tim)-1], yax-yax[len(tim)-1], tim

def adjs_rsqr(yax,ypred):
    """
    Computes the adjusted R-squared (R²) value.

    This function calculates the adjusted R², which is a statistical measure used to assess 
    the goodness of fit of a model. It adjusts the R² value to account for the number of
    predictors in the model, helping to prevent overfitting.

    Parameters:
    yax (array-like): The observed (actual) y-values from the dataset.
    ypred (array-like): The predicted y-values from the model.

    Returns:
    float: The adjusted R² value.
    """
    residuals = yax-ypred
    tss = np.sum((yax-np.mean(yax))**2)
    rss = np.sum(residuals**2)
    r2 = 1-(rss/tss)
    adj_r2 = 1-((1-r2)*(len(yax)-1))/(len(yax)-5)
    return adj_r2

def funfit(file):
    """
    Fits a sinusoidal model to data from a CSV file and computes the adjusted R-squared value.

    This function reads data from a CSV file, extracts and processes the time, x-axis, and
    y-axis data, fits a sinusoidal model to the data, and calculates the adjusted R² value.
    The function also adjusts the data length to the nearest power of 2.

    Parameters:
    file (str): The file path of the CSV containing the data. The CSV must contain columns
    for time and y-values.

    Returns:
    tuple: A tuple containing:
        - `yax`: The original y-axis data.
        - `ynew`: The fitted sinusoidal y-axis data.
        - `xax`: The processed x-axis data.
        - `adr`: The adjusted R² value of the fitted model.
        - `tim`: The processed time data.
    """
    datta = pd.read_csv(file)
    xax, yax, tim = stepnumb(datta)
    ynew = fitsincuve(xax,yax)
    adr = adjs_rsqr(yax,ynew)
    return yax, ynew, xax, adr, tim

def fftfinding(ynew, adr, tim, adtol = 0.1, mag=False):
    """
    Computes the FFT of the fitted data if the adjusted R² is above a specified threshold.

    This function checks if the time intervals in the provided time data are consistent and
    if the adjusted R² value is above a specified threshold. If both conditions are met, it
    computes the Fast Fourier Transform (FFT) of the provided fitted y-values. Optionally,
    it returns the magnitude of the FFT (up to the Nyquist frequency) if specified.

    Parameters:
    ynew (array-like): The fitted y-values obtained from a model (e.g., sinusoidal fitting).
    adr (float): The adjusted R² value, indicating the goodness of the model fit.
    tim (array-like): The time data corresponding to the y-values.
    adtol (float, optional): The minimum acceptable adjusted R² value for performing FFT
    (default is 0.1). mag (bool, optional): If True, the function returns the magnitude
    of the FFT; otherwise, it returns the raw FFT (default is False).

    Returns:
    numpy array or None: The FFT of the fitted data (or its magnitude) if conditions are met,
    otherwise None.
    """
    n = len(tim)
    time_diffs = np.diff(tim)
    avg_time_diff = np.mean(time_diffs)
    compare = np.allclose(time_diffs, avg_time_diff, atol=1e-1)
    if compare:
        if adr > adtol:
            fft = np.fft.fft(ynew)
            if mag:
                return np.abs(fft)[:n // 2]
            return fft
        print(f"adr value {adr} is too low for FFT. It needs to be greater than {adtol}.")
        return None
    print('compair failing')
    print(avg_time_diff)
    return None

def inv_fft(isfft, real='False'):
    """
    Computes the inverse FFT (IFFT) of a given frequency-domain signal.

    This function takes an input FFT or frequency-domain signal and computes its inverse FFT 
    to obtain the time-domain signal. If specified, it returns the magnitude (real part)
    of the IFFT.

    Parameters:
    isfft (array-like): The input FFT or frequency-domain signal to be transformed back to
    the time domain. real (str, optional): If 'True', the function returns the magnitude
    (real part) of the IFFT; if 'False' (default), it returns the full complex-valued IFFT.

    Returns:
    numpy array: The inverse FFT of the input signal, either as a complex-valued or
    real-valued signal.
    """
    if real:
        return np.abs(np.fft.ifft(isfft))
    return np.fft.ifft(isfft)

def freqfinder(xax):
    """
    Computes the frequency values corresponding to a given set of x-axis data points.

    This function assumes that the x-axis data points represent evenly spaced time intervals and 
    computes the corresponding frequency values using the formula for a discrete Fourier transform.

    Parameters:
    xax (array-like): The x-axis data, typically representing time or samples.

    Returns:
    numpy array: The computed frequency values corresponding to the x-axis data.
    """
    n = len(xax)
    d = sum(xax[i+1] - xax[i] for i in range(n-1)) / (n - 1)
    freqs = []
    for k in range(n):
        f = k / (n * d*100)
        freqs.append(f)
    return np.abs(freqs)

def filteredfreq(fft, freq, selec_filter=0.6,selec_filter2 = 0.01):
    """
    Filters the frequency-domain data based on two threshold values.

    This function filters the input FFT data to keep only the frequencies that lie within a
    specified range. Frequencies below a low threshold (defined by `selec_filter`) and
    above a high threshold (defined by `selec_filter2`) are discarded. The function returns
    the filtered FFT values and the corresponding frequencies.

    Parameters:
    fft (array-like): The frequency-domain data (e.g., FFT results).
    freq (array-like): The corresponding frequencies for the FFT data.
    selec_filter (float, optional): The lower threshold multiplier for filtering (default is 0.6).
    selec_filter2 (float, optional): The upper threshold multiplier for filtering (default is 0.01).

    Returns:
    tuple: A tuple containing:
        - `rry`: The filtered FFT values, with values outside the threshold range set to zero.
        - `newfrq`: The corresponding frequencies that fall within the specified threshold range.
    """
    threshold_low = selec_filter * np.max(fft)
    threshold_high = selec_filter2 * np.max(fft)
    mask = (fft >= threshold_high) & (fft <= threshold_low)
    rry = np.zeros(len(fft))
    newfrq = np.zeros(len(fft))
    rry[mask] = fft[mask]
    newfrq[mask] = freq[mask]
    return rry, newfrq

def sumfunk(matrx):
    """
    Averages the values in each column of a 2D matrix, ignoring zeros.

    This function iterates through a 2D matrix (list of lists) and computes the average of the 
    non-zero elements in each column. For each column, the sum of non-zero elements is divided 
    by the number of non-zero elements to obtain the average.

    Parameters:
    matrx (list of lists or 2D array-like): The input matrix, where each sublist represents a row.

    Returns:
    numpy array: A 1D array containing the averaged values of each column, ignoring zeros.
    """
    m2 = 0
    for row in matrx:
        m2 = max(len(row), m2)
    arr1 = np.zeros(m2)
    arr2 = np.zeros(m2)
    arr3 = np.zeros(m2)
    for i in range(len(matrx)):
        for j in range(len(matrx[i])):
            x1 = matrx[i][j]
            if x1 !=0:
                arr1[j] = arr1[j]+x1
                arr2[j] += 1
    for k in range(m2):
        if arr2[k] != 0:
            arr3[k] = arr1[k]/arr2[k]
    return arr3
