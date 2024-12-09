Running the Code

We begin the process of running our code by first processing our data. We do this by:

    - Placing the GPS and acceleration data files in the Tolani4/data/ folder.
    - Then run each module individually by calling the functions from the Python command line


We then convert measurements from our data with our unit_converter() module:
# For Example
from Tolani4.Code.unit_converter import feet_to_meters, yards_to_meters

meters = feet_to_meters(500)  # Converts 500 feet to meters

We then calculate the distance between two GPS points or direction of movement using our calculate_direction() and haversine_distance() modules:
# For Example
from Tolani4.Code.distance_calculator import haversine_distance
from Tolani4.Code.direction_calculator import calculate_direction

distance = haversine_distance((lat1, lon1), (lat2, lon2))
direction = calculate_direction(acceleration_data)

Finally we convert date and time to Unix time with our unix_converter() module:
# For Example
from Tolani4.Code.unix_time_converter import convert_to_unix

unix_time = convert_to_unix("2024-10-25", "14:30")

To visually depict our GPS motions in a x,y plane and directions in the elevator over time
we can use our gps_motion_plot.py module and the elevator_direction_plot.py module to illustrate this.