# How to run the code in the iglesias-cardinale folder


## Step 1: Set up your environment

To use the functions in any of the modules, create a Jupyter Notebook file within the code folder and make sure to import any necessary libraries, such as NumPy.

## Step 2: Import Functions from Modules

Import the desired modules from the code folder

```python
from distance import *
from direction_of_motion import *
from unit_converter import *
from unix_time import *
```

Note that not all mdules are necessary. If, for instance, you need only to calculate velocities from acceleration data, you'll only need to import direction_of_motion.

### Typical use cases

Generally, the modules are useful in the following cases:

#### distance

- Calculating and plotting distance vs time
- Calculating and plotting the distance between two points, both assuming the Earth is Curved and the Earth is Flat.

#### direction_of_motion

- Finding velocities given acceleration and time data
- Finding and plotting the direction of motion from acceleration and time data

#### unit_converter

- Doing simple unit conversions between Feet, Yards, Meters, and Kilometers

#### unix_time

- Calculating the unix time of any date and time in the calendar year 2024. The capabilities are limited to 2024 due to pylint restrictions on the number of positional arguments in a function. 

## Step 3: Test using PyTest

The validity of all modules can be tested using Pytest in the terminal. Simply navigate to the code directory and give the command ```pytest```.

