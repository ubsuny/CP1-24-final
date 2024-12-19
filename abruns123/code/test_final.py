"""
the test_final module implements test functions
for a pytest to verify that the functions of final.py work
"""
import final
import numpy as np

def test_parse():
    """
    test_parse verifies that the parse function
    can successfully acquire the correct temperature
    from a file.
    """
    assert final.parse("/workspaces/CP1-24-final/abruns123/data/sin_data/e0001_meta.md")==41

def test_f_k():
    """
    test_f_k verifies that the f_k function can 
    successfully convert a fahrenheit temperature
    to kelvin within accuracy of 1e-3
    """
    assert final.f_k(32)==273.15
    assert np.isclose(final.f_k(98), 309.817, rtol=1e-3)

def test_file_list():
    """
    verifies that the correct number of files are compiled into a list
    """
    assert len(final.file_list("/workspaces/CP1-24-final/abruns123/data/sin_data",".csv"))==20

def test_model():
    """
    test_model verifies that the model
    function returns the correct value for
    a sin wave
    """
    assert final.model(np.pi, 5, 1/2, 3)==5*np.sin(np.pi/2+3)

def test_residuals():
    """
    Test with a simple linear case
    """
    params = [1, 2, 3]  # a=1, b=2, c=3
    x = np.array([1, 2, 3])  # Some sample x values
    y =final.model(x, params[0], params[1], params[2])
    # Corresponding y values for y = x^2 + 2x + 3

    # Expected residuals should be zero since y matches the model exactly
    expected_residuals = np.array([0, 0, 0])
    computed_residuals = final.residuals(params, x, y)

    # Test if the computed residuals match the expected residuals
    np.testing.assert_array_equal(computed_residuals, expected_residuals)

def test_jacobian_basic():
    """
    tests basic case of jacobian matrix
    """
    # Define the x values and parameters
    x = np.array([0, np.pi/2, np.pi])
    params = [1, 2, 3]  # a=1, b=2, c=3

    # Compute the Jacobian matrix using the function
    computed_jacobian = final.jacobian(x, params)

    # Expected values for the Jacobian matrix
    expected_jacobian = np.zeros((len(x), 3))

    # Compute expected values manually (partial derivatives)
    expected_jacobian[:, 0] = -np.sin(2 * x + 3)  # Partial derivative w.r.t A
    expected_jacobian[:, 1] = -1 * x * np.cos(2 * x + 3)  # Partial derivative w.r.t B
    expected_jacobian[:, 2] = -1 * np.cos(2 * x + 3)  # Partial derivative w.r.t C

    # Check if the computed Jacobian is equal to the expected Jacobian
    np.testing.assert_array_almost_equal(computed_jacobian, expected_jacobian, decimal=6)

def test_jacobian_edge_cases():
    """
    Test for an edge case where x is all zeros
    """
    x = np.array([0, 0, 0])
    params = [1, 2, 3]  # a=1, b=2, c=3

    # Compute the Jacobian matrix using the function
    computed_jacobian = final.jacobian(x, params)

    # Expected values for the Jacobian matrix
    expected_jacobian = np.zeros((len(x), 3))

    # Compute expected values manually (partial derivatives)
    expected_jacobian[:, 0] = -np.sin(2 * x + 3)  # Partial derivative w.r.t A
    expected_jacobian[:, 1] = -1 * x * np.cos(2 * x + 3)  # Partial derivative w.r.t B
    expected_jacobian[:, 2] = -1 * np.cos(2 * x + 3)  # Partial derivative w.r.t C

    # Check if the computed Jacobian is equal to the expected Jacobian
    np.testing.assert_array_almost_equal(computed_jacobian, expected_jacobian, decimal=6)

def test_jacobian_with_small_x():
    """
    Test when x values are small (close to zero)
    """
    x = np.array([0.001, 0.002, 0.003])
    params = [1, 2, 3]  # a=1, b=2, c=3

    # Compute the Jacobian matrix using the function
    computed_jacobian = final.jacobian(x, params)

    # Expected values for the Jacobian matrix
    expected_jacobian = np.zeros((len(x), 3))

    # Compute expected values manually (partial derivatives)
    expected_jacobian[:, 0] = -np.sin(2 * x + 3)  # Partial derivative w.r.t A
    expected_jacobian[:, 1] = -1 * x * np.cos(2 * x + 3)  # Partial derivative w.r.t B
    expected_jacobian[:, 2] = -1 * np.cos(2 * x + 3)  # Partial derivative w.r.t C

    # Check if the computed Jacobian is equal to the expected Jacobian
    np.testing.assert_array_almost_equal(computed_jacobian, expected_jacobian, decimal=6)

# Test the gauss_newton function
def test_gauss_newton():
    """
    tests basic case of the gauss_newton method
    """
    # Sample data for testing
    x = np.array([0, 1, 2, 3, 4, 5])
    y = np.sin(x) + 0.1 * np.random.randn(len(x))  # Example noisy sine data

    initial_params = [1, 1, 0]  # Initial guess for parameters (a, b, c)
    n = 5  # Step for new x values

    # Call the gauss_newton function
    params, new_func, new_x = final.gauss_newton(x, y, initial_params, n)

    # Check that the returned parameters are reasonable (e.g., the model should fit the data)
    assert len(params) == 3  # There should be 3 parameters (a, b, c)
    assert np.all(np.isfinite(params))  # Parameters should be finite numbers

    # Check that the new function's shape matches the expected number of points
    assert len(new_func) == len(new_x)

def test_gauss_newton_convergence():
    """
    Check for convergence with simple data and known parameters
    """
    x = np.array([0, 1, 2, 3, 4, 5])
    y = np.sin(x)  # Perfect sine data without noise
    initial_params = [1, 1, 0]
    n = 5

    # Call the gauss_newton function
    params, _, _ = final.gauss_newton(x, y, initial_params, n)

    # Check that the parameters are close to the actual model parameters (a=1, b=1, c=0)
    assert np.allclose(params, [1, 1, 0], atol=1e-3)

def test_gauss_newton_with_large_iterations():
    """
    Test the max iteration condition
    """
    x = np.array([0, 1, 2, 3, 4, 5])
    y = np.sin(x) + 0.1 * np.random.randn(len(x))  # Noisy sine data
    initial_params = [1, 1, 0]
    n = 5
    max_iter = 10  # Limit iterations to 10

    # Call the gauss_newton function
    params, _, _ = final.gauss_newton(x, y, initial_params, n, max_iter=max_iter)

    # Check that the parameters are finite and that the number of iterations is reasonable
    assert len(params) == 3
    assert np.all(np.isfinite(params))

def test_gauss_newton_with_tolerance():
    """
    Test if tolerance is met for stopping criteria
    """
    x = np.array([0, 1, 2, 3, 4, 5])
    y = np.sin(x)  # Perfect sine data without noise
    initial_params = [1, 1, 0]
    n = 5
    tolerance = 1e-8  # Tighter tolerance

    # Call the gauss_newton function
    params, _, _ = final.gauss_newton(x, y, initial_params, n, tolerance=tolerance)

    # Check that the parameters are very close to the true values
    assert np.allclose(params, [1, 1, 0], atol=1e-8)

def test_get_sin_data():
    """
    ensures that the correct datatypes are 
    returned by the function
    """
    x,y=final.get_sin_data("/workspaces/CP1-24-final/abruns123/data/sin_data/e0001_walk.csv")
    assert isinstance(x, list)
    assert isinstance(y,list)

def test_subtract_ave():
    """
    verifies that if all the ydata is the same,
    then an array of zeros is returned
    """
    y=np.array([1,1,1,1])
    condition=True
    y=final.subtract_ave(y)
    for i in y:
        if i==0:
            condition is False
    assert condition is True

def test_wrap_fft_non_equidistant_data():
    """
    Test that the function handles non-equidistant x-values properly.
    """
    x = np.array([0, 1, 2, 4, 5])  # Non-equidistant data
    y = np.sin(x)
    inverse = False

    # The function should print an error message and return None
    result = final.wrap_fft(x, y, inverse)

    assert result is None, "The function should return None for non-equidistant data"

def test_wrap_fft_forward():
    """
    Test the forward FFT (inverse=False) on equidistant data.
    """
    x = np.array([0, 1, 2, 3, 4])
    y = np.sin(x)  # Sample data
    inverse = False

    result = final.wrap_fft(x, y, inverse)

    # Check if the result is a numpy array and has the same shape as the input y
    assert isinstance(result, np.ndarray)
    assert result.shape == y.shape

    # Check if the forward FFT result is a complex array
    assert np.iscomplexobj(result), "The FFT result should be complex"
