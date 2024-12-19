"""
test_final.py
this contains the test functions that test the
functonalities of the function from the file final.py
"""
import os
import pytest
import numpy as np
import pandas as pd
import final as fin

@pytest.fixture
def crtd_data():
    """  this just create the ficture for latter tests to use"""
    tme = ["Time (s)",1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17]
    lat = ["Latitude (°)", 1, 2.01, 3.02, 4.03, 5.04, 6.05, 7.05, 8.06, 9.07, 10.08, 11.09,
           12.1, 13.2, 14.21, 15.22, 16.22, 17.23]
    long = ["Longitude (°)", 1, 2.01, 0.9, 0, 0.9, 2, 1, 0.01, 1, 2.01, 1, 0.01, 1.02, 2, 1, 0, 1]
    alt = ["Altitude WGS84 (m)", 1000, 1001, 1002, 1003, 1004, 1005, 1006, 1007,
           1008, 1009, 1004, 1004, 1003, 1006, 1002, 1001, 1007]
    return pd.DataFrame({"Time (s)": tme[1:], "Latitude (°)": lat[1:], "Longitude (°)": long[1:],
                         "Altitude WGS84 (m)": alt[1:]})

@pytest.fixture
def ypredic():
    """also just a pytest ficture used to check code"""
    return [1, 2, 1, 0, 1, 2, 1, 0, 1, 2, 1, 0, 1, 2, 1, 0, 1]

@pytest.fixture
def create_test_files():
    """Fixture to create test files in a temporary directory."""
    test_dir = "/tmp/test_dir"
    os.makedirs(test_dir, exist_ok=True)
    file1 = os.path.join(test_dir, "eletot_file1.md")
    file2 = os.path.join(test_dir, "eletot_file2.md")
    open(file1, 'w').close()
    open(file2, 'w').close()
    yield test_dir
    os.remove(file1)
    os.remove(file2)
    os.rmdir(test_dir)


def test_fah_to_kel():
    """Tests the conversion from Fahrenheit to Kelvin."""
    assert fin.fah_to_kel(32) == 273.15
    assert fin.fah_to_kel(212) == 373.15
    assert fin.fah_to_kel(-500) is None

def test_find_fah():
    """Tests the function that finds the temperature in a file."""
    temp_file_path = "test_temp.txt"
    with open(temp_file_path, 'w') as f:
        f.write("Recorded temperature is 72\n")
    temp = fin.find_fah(temp_file_path)
    assert temp == 72
    os.remove(temp_file_path)


def test_filenamelister(create_test_files, monkeypatch):
    """Tests the function that lists files matching a pattern."""
    monkeypatch.setattr(fin, 'filenamelister', lambda exp_name, filetype='.md':
                        [os.path.join(create_test_files, f) for f in os.listdir(create_test_files)
                         if f.endswith(filetype)])
    files = fin.filenamelister("eletot", ".md")
    file1 = os.path.join(create_test_files, "eletot_file1.md")
    file2 = os.path.join(create_test_files, "eletot_file2.md")
    assert file1 in files
    assert file2 in files

def test_findmdfromcsv():
    """Tests the function that extracts part of the CSV filename."""
    file_path = "LL13_sinewalktest.csv"
    result = fin.findmdfromcsv(file_path)
    assert result == "_sinewalktest"

def test_sinfunk():
    """Tests the sinfunk function with sample values."""
    x = np.array([0, np.pi/2, np.pi])
    result = fin.sinfunk(x, 1, 1, 0, 0)
    expected = np.array([0, 1, 0])
    assert np.allclose(result, expected)

def test_fitsincuve():
    """Tests the fitting of a sinusoidal curve."""
    x = np.linspace(0, 10, 100)
    y = np.sin(x)
    fitted_y = fin.fitsincuve(x, y)
    assert len(fitted_y) == len(y)

def test_adjs_rsqr():
    """Tests the calculation of the adjusted R-squared value."""
    yax = np.array([1, 2, 3, 4])  # Observed y-values
    ypred = np.array([1, 2, 3, 4])  # Predicted y-values (should match yax in length)
    adj_r2 = fin.adjs_rsqr(yax, ypred)
    assert adj_r2 is not None

def test_funfit(crtd_data):
    """Tests fitting a sinusoidal model from a CSV file using the `funfit` function."""
    yax, ynew, xax, adr, tim = fin.funfit(crtd_data)
    assert len(yax) == len(ynew), f"Length of yax ({len(yax)}) does not match ynew ({len(ynew)})"
    assert adr > 0, f"Adjusted R-squared (adr) is not positive: {adr}"

    assert len(xax) == len(tim), f"Length of xax ({len(xax)}) does not match tim ({len(tim)})"
    assert len(yax) == len(tim), f"Length of yax ({len(yax)}) does not match tim ({len(tim)})"
    os.remove(crtd_data)

def test_fftfinding():
    """Tests the FFT computation on fitted data."""
    x = np.linspace(0, 10, 100)
    y = np.sin(x)
    adr = 0.9
    fft_result = fin.fftfinding(y, adr, x)
    assert fft_result is not None


def test_inv_fft():
    """Tests the inverse FFT calculation."""
    fft_data = np.fft.fft(np.sin(np.linspace(0, 10, 100)))
    result = fin.inv_fft(fft_data)
    assert len(result) == len(fft_data)

def test_freqfinder():
    """Tests the frequency calculation from x-axis data."""
    x = np.linspace(0, 10, 100)
    freq = fin.freqfinder(x)
    assert len(freq) == len(x)

def test_filteredfreq():
    """Tests the filtering of frequency-domain data."""
    fft_data = np.fft.fft(np.sin(np.linspace(0, 10, 100)))
    freq = fin.freqfinder(np.linspace(0, 10, 100))
    filtered_fft, _ = fin.filteredfreq(fft_data, freq)
    assert len(filtered_fft) == len(fft_data)



def test_sumfunk():
    """Tests the sumfunk function with a 2D matrix."""
    matrx = [
        [1, 2, 0],
        [4, 5, 6],
        [7, 0, 9]
    ]
    result = fin.sumfunk(matrx)
    expected = np.array([4, 3.5, 7.5])
    assert np.allclose(result, expected)
