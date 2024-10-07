# Midterm 23

This is the midterm for CP1-23 based on Edwin's Hubble paper on the velocity-distance relation of extra galactic nebulae.

*Algorithm (14 points):*
- Make a least-squares fit with uncertainties based on the pseudo-code below to the 9 groups (open circles) in Figure 1 of [_Hubble's 1929 article_](https://www.pnas.org/content/pnas/15/3/168.full.pdf) (2 points).
- Generate a library named `hubble_tools.py` for the above algorithm that also can take lists of velocities and distances and returns a dictionary of the fitting parameters (4 points).
- Add a function to the library that can estimate the age of the universe based on the data in the Hubble paper (2 points).
- Determine the 9 groups of the figure (see hint below) and generate a data file (eg csv) with the velocities and distances (2 points).
- Programmatically read back in this data in your code to do the fit (4 points).

*Documentation (14 points):*
- Generate docstrings for your library (4 points).
- Describe in your documentation how you selected the 9 groups (4 points).
- Compare the slope of the fitted straight line to Hubble's value of K (2 points).
- Describe how to run your code (4 points).

*Tools (14 points):*
- Write unit tests for both algorithms (4 points).
- Add github actions for unit testing (using pytest) and linting (linter of your choice) of your code (4 points)
- Choose an appropiate license for your project (1 point)
- Add a bibliography in your documentation and references to all sources you used (5 points).


### Pseudocode:
```
input data
n = size of data
if n < 2 : 
   print ‘Error! Not enough data!’
   return
for i = 0... N-1 : 
   if abs( sigma_i ) < 0.00001 : 
      return
   S += 1.0 / sigma_i**2
   s_x += x_i / sigma_i**2
   s_y += y_i / sigma_i**2
for i = 0... N-1 : 
   t_i = 1.0 / sigma_i * (x_i-s_x/S)
   s_tt = t_i**2
   b += t_i * y_i / sigma_i
if abs( S ) < 0.000001 : 
   return
a = (s_y - s_x * b) / S
b = b / s_tt
sigma_a2 = (1 + s_x**2/S*s_tt) / S
sigma_b2 = 1.0 / s_tt
for i = 0... N-1 : 
   chi2 += ((y_i - a - b*x_i)/sigma_i)**2
```

**Note:**
Galaxy catalog available at (http://spider.seds.org/ngc/ngc.html). 
