# Code Explanation

## The Code Flow

### The **libraries_module.py**

- It helps keep track of which libraries are allowed

### The **unit_converter_module.py**

- It defines the functions that convert between different units (if needed)

### The Preliminaries Modules

- Those are used to define the preliminary functions that are to be used by the **main modules**

#### The **time_conversion_preliminaries.py**

- It defines the preliminary functions to be used by the **time_converter_module.py**
- It's to check for the leap years and then calculate the number of seconds passes between two given dates and times

#### The **direction_of_motion_preliminaries.py**

- It defines the preliminary functions functoins to be used by the **direction_of_motion_module.py**
- It's for the calculation of averages and unit vectors, and integrating a given set of data points

### The Functions Modules

- They act as bridges between the **preliminary modules** and the **main modules**

#### The **time_conversion_function.py**

- It defines the function that utilizes the functions already defined in the **time_conversion_preliminaries.py** to calculate the number of seconds passed between the beginning of year 1970 (the unix epoch) and any other given year

#### The **distance_functions.py**

- It defines the two main functions for distance calculations on ellipsoidal surfaces in an iterative approach using the **Vincenty's Formula** [1].
- The first function is to calculate the distance between any two given angular positions on an ellipsoidal surface.
- The second function is to calculate the distance between any two angular positions on the Earth's surface through feeding the Earth's paramters to the first function

#### The **mercator_projection_functions.py**

- In fact, that can be considered a base module for it's the one that defines all that is needed to convert angular positions to planar positions through projecting the spherical surface onto a cylinderical surface using the **Mercator Projector** formuli [2]

#### The **plotting_functions_module.py**

- Similar to the **mercator_projection_functions.py**, it defines all the functions to produce plots of data sets.
- It defines two function. The first function is to plot two dimensional data, and the second function is to be able to show multiple plots on the same grid.

### The Main Modules

#### The **time_converter_module.py**

- It reads all the *meta files* and extracts all the time data for all the runs of each experiment
- It extracts the start and end time of each run for all the experiments
- It uses the **time_conversion_function.py** to convert those times to the *unix time*

#### The **direction_of_motion_module.py**

- It reads all the *raw data files* and extracts all the times and acceleration components
- It uses the **direction_of_motion_preliminaries.py** to integrate the acceleration data against time (to calculate velocities), calculate the directions unit vectors from the velocities' directions, and then averages over all the directions for the journey to get a sense of the overall direction

#### The **distance_calculator_module.py**

- It reads all the *raw data files* and extracts all the times and angular positions
- It uses the **distance_functions.py** to calculate the distance between each two consecutive points in the data set of each run for all the experiments

#### The **motion_tracing_module.py**

- It imports the data read by the **distance_calculator_module.py** to get the angular positions for only the fourth run of each experiment.
- It uses the **mercator_projection_functions.py** to convert those angular positions to planar positions.
- It also imports the data read by the **direction_of_motion_module.py** to get the times and direction vectors for only the fourth run of each experiment.
- It then uses the **plotting_functions_module.py** to plot the directions against time for the *Acceleration* experiments and the xy-coordinates for the *GPS* experiments

## References

[1] "DIRECT AND INVERSE SOLUTIONS OF GEODESICS 0N THE ELLIPSOID WLTH APPLICATION OF NESTED EQUATIONS"
<https://www.ngs.noaa.gov/PUBS_LIB/inverse.pdf>

[2] "Mercator projection"
<https://en.wikipedia.org/wiki/Mercator_projection>
