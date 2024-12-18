# README

## Data & Preprocessing
Data was recorded with phyphox, shipped as a csv, and the preprocessed to conform to naming conventions.
Metadata also prepared in a similar manner. Metadata lives in a markdown file named after its experiment.

## Files
- /code/final.py
- /code/test_final.py
- /docs/final.ipynb
- /data/AVG002_gps_sine_walk/experimental_data/AVG002_gps_sine_walk_{trial_number}.csv

### Trial Data
Experimental data lives in the /data/AVG002_gps_sine_walk/experimental_data/ directory.
- Experimental GPS data is stored as a .csv file.
- Each run's metadata is stored as a .md file with the same filename.
### Module: final.py
The final.py module implements the algorithm tasks. It can be found in the /code/ directory.
### Module: test_final.py
The test_final.py module implements the unit tests for the final.py module's functionality. It can also be found in the /code/ directory and can be run with `pytest test_final.py`.


## Project Tasks
### Merging Task
Personal contributions from the midterm project repository have been merged into the final project repository, and the full commit history has been preserved. Midterm project files will live side-by-side with the final project files, however final project files have been specifically named and documented for the sake of clarity.
### Data Task
Data files live in the /data/AVG002_gps_sine_walk/experimental_data/ directory.
### Algorithm Task
#### Function that converts Fahrenheit to Kelvin
Implemented in final.py by the convert_f_to_k() method.
#### Markdown Parser
Implemented in final.py by the parse_markdown() method. Can parse any of the labelled values.
#### Filename Lister
Implemented in final.py by the filter_markdown_files() method, assisted by:  
- extract_number_from_filename()
- sort_filenames()
#### Nonlinear Fitting Curve
Implemented in final.py by the prepare_and_fit() method, assisted by:
- rotate_to_x_axis()
- sine_fit()
- resample_to_2n_segments()
- degrees_to_meters()
#### FFT and IFFT Wrappers
Implemented in final.py by the apply_fft() and apply_ifft() methods, assisted by:
- resample_df_to_2n_segments()
- calculate_frequencies()
- apply_fft_to_arrays()

## Additional Notes
- I had to start using type hints at a certain point to keep the data types straight while passing between functions, but I may have missed a few since I started using them about halfway through. It doesn't seem like the most pythonic practice at first, but I learned to appreciate the hints as the routine grew more complex.

- Plotting methods are usually handled in the code cells, but a prototype was rolled into the codebase for future reference.

# Bibliography
- [Handling GPS with Python](https://florianwilhelm.info/2016/07/handling_gps_data_with_python/)
- [Calculate Distance Between GPS Points in Python](https://janakiev.com/blog/gps-points-distance-python/)
- [Matplotlib List of Colormaps](https://matplotlib.org/stable/gallery/color/named_colors.html)
- [Discrete Fourier Transform ](https://numpy.org/doc/2.1/reference/routines.fft.html)
- [IFFT | Numpy](https://numpy.org/doc/stable/reference/generated/numpy.fft.ifft.html)
- [Understanding FFT Windows](https://www.egr.msu.edu/classes/me451/me451_labs/Fall_2013/Understanding_FFT_Windows.pdf)
- [Understanding FFTs and Windowing](https://download.ni.com/evaluation/pxi/Understanding%20FFTs%20and%20Windowing.pdf)
- [FFT normalisation - Reddit/r/DSP](https://www.reddit.com/r/DSP/comments/htxvn8/fft_normalisation/)
- [Scaling of the DFT and Some More of Its Noteworthy Properties](https://appliedacousticschalmers.github.io/scaling-of-the-dft/AES2020_eBrief/)
- [FFT Normalization](https://www.hep.ucl.ac.uk/~rjn/saltStuff/fftNormalisation.pdf)
- [Numpy Documentation](https://numpy.org/doc/stable/user/)
- [Pandas Documentation](https://pandas.pydata.org/docs/user_guide/index.html#user-guide)
- [Matplotlib Documentation](https://matplotlib.org/stable/users/index.html)
- [Principle Component Analysis & Python](https://builtin.com/machine-learning/pca-in-python)
- [Resampling Data with Python | numpy.interp](https://numpy.org/doc/2.1/reference/generated/numpy.interp.html)
- [Resample and Interpolate Your Time Series Data](https://machinelearningmastery.com/resample-interpolate-time-series-data-python/)
