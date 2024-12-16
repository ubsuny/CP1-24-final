# About
This project consists of two Python modules that represent different types of physical motion paths, or "trips".

Trips can be created either using either acceleration data or GPS location data gathered from PhyFox.


## Modules
The modules are split between the classes which represent trips, and the remaining functionalities which are provided as a utils module.

### trip.py Module
This is where the GpsTrip and AccelTrip (as well as its base class) are implemented.

Shared functionality is provided in the BaseTrip class. The subclasses extended BaseTrip to accommodate the specific needs and nature of the different trip types.

Trip objects are instantiated with the name of the data unzip directory as the first and only positional argument, provided as a string.

### utils.py Module
The utilities module provides functionality like unit conversion functions, csv and metadata importing routines, and several different types of plotting methods.

For more information, see the module's docstrings as well as the example provided below for some basic use cases.

### Testing Modules
Unit testing is provided in two separate files, one to test each of the primary modules.

Unit testing is implemented using the pytest, however the Python built-in unittest module is also used, but solely the purpose of mocking data fixtures.

After much effort to reconcile ongoing linting issues with the pytest data fixtures, the decision was made to utilize unittest for the mocking data instead.

# How to Run
See `Example.md` for a detailed preview.

## Data Preparation
Data exported from PhyFox must be unzipped into the **data directory**, which is located (with respect to the core modules) at the relative path "../data/".

When data is exported from PhyFox as a .zip, place this archive into the **data directory**. Unzip the archive, yielding a new directory which we will call *experiment directory*.

You be able to locate the following:
Experiment Data:
/**data directory**/*experiment directory*/Raw Data.csv
Metadata:
/**data directory**/*experiment directory*/meta/time.csv
/**data directory**/*experiment directory*/meta/device.csv

In order create a Trip object, you must provide the *experiment directory* as a string without any slashes ('/') or further nested directories.

Once the data has been unzipped, it is important to not alter any of the contained files or their metadata. It his however safe to rename the *experiment directory*.

## Trip Creation
To create a trip simply import the class, and create a new object using a set of trip data.

```python
# Import the trip module
import trip

# Get ahold of the GpsTrip class object
GpsTrip = trip.GpsTrip
# Get ahold of the GpsTrip class object
AccelTrip = trip.AccelTrip

# Create some trips!
gps_trip_tri = GpsTrip("AVG001_gps_tri_walk_1")
trip_up = AccelTrip("AVG001_accel_elevator_up_1")
```

## Visualization
We also have several plotting methods available for the different trip types.

```python
# import the utils module
import utils

# Get some plotting methods from the utils module
acceleration_plot = utils.plot_acceltrip_acceleration_with_color  # acceleration
velocity_plot = utils.plot_acceltrip_velocity_with_color          # velocity
trajectory_plot = utils.plot_3d_trajectory                        # 3d trajectory

# Plot a GpsTrip
gps_plot(gps_trip_tri, title="GPS Trip: Triangle Walk")
# Plot an AccelTrip
acceleration_plot(trip_up, component='z', title="Elevator Ascent - Acceleration")
```

### Extra: Thresholding and Compression
The data recorded can have a significant amount of noise or error and as a result, that error can accumulate in calculations that follow.

One way to address this (after data measurement) is to attempt to control the noise through thresholding. Some basic thresholding methods have been provided and can be seen demonstrated in the provided "Example.md" file.

Alternatively, if the issue is that too much of the data falls within the same region of values then differentiating the shape of the plot can become difficult-- this is especially true with heatmaps or colormaps. To address this need, many of the plotting functions offer the option to apply a compression factor when during visualization which is also demonstrated in the "Example.md" file.

# Task Requirements
## Data Task
The collected GPS and accelerometer data for the 4 different experiments lives in the /data/ directory (along with several other sets of testing data). This includes:
- GPS data for the path of a large circle
- GPS data for the math of a large right triangle
- Accelerometer data for ascending elevator
- Accelerometer data for descending elevator

All data is coupled with its generated metadata and a markdown file containing extra information about the experimental testing conditions.
## Algorithm Task
The following functionalities have been implemented:
- Unit conversion, which can be found in util.py
- Distance calculation between adjacent GPS positions, provided by GpsTrip
  - Planar distance calculation is provided (via simple Euclidian distance)
  - Curved distance calculation is provided (via extended Haversine distance)
- Acceleration, Velocity, and Position calculation for each frame of acceleration data is provided by AccelTrip
- Handling of data and metadata is provided by the BaseTrip as well as functions in the utils module.
- Conversion of local timecode time to unix time is provided in the utils module
- All modules and testing files fully lint with a 10/10 score on default pylint settings
- Only the modules specified in requirements.txt and from the python standard library have been used.
## Documentation Task
- All modules have full complements of code comments and docstrings
- 1 image for the GPS motions has been generated in the /documentation/ folder and is named "fig_gps_trip_plots.png"
- 1 image for the accelerometer motions has been generated in the /documentation/ folder and is named "fig_acceleration_trip_plots.png"
- This file as well as "/documentation/Example.md" have been provided to demonstrate how to use the code.
## Tools Task
- Unit tests have been written for both modules and are cleanly labelled with the "test_" prefix.
- A full list of references can be found in "/documentation/project-resources.md"

#### Thank You!
