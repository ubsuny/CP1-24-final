# Midterm 24 - data read and process

This is the midterm for CP1-24 based on getting and using research data using the phyphox app (https://phyphox.org).

*Data: 9 points*
- Use the Location (GPS) app to record the spatial locations of how you move along the outline of the  following two geometric elements:
   - a circle with a diameter of at least $d = \frac{500\rm{ft}}{\pi}$
   - a right triangle with at least $a = 80 \rm{ft}$ and $b = 50 \rm{yd}$
- Use the Acceleration (without g) app to record the linear accelaration in $x,y,z$ for the following events:
   - Take the elevator in any UB north campus building from the lowest to the top floor
   - Take the elevator in any UB north campus building from the top to the lowest floor
- After each experiment export all collected data as a csv files and generate an additional markdown file with the following meta information:
   - Date and time of the experiment
   - Current weather conditions
   - Any additional information regarding the experiment

*Algorithm (14 points):*
- write the following python modules that implement general functions:
   - for a unit converter for feet and yard to an SI equivalent (2p)
   - that calculate the distance between two adjacent GPS positions (bonus point if you consider the earth not being flat) (5p)
   - that calculate the direction of motion from the acceleration data (5p)
   - to read out the data and time from each metafile and convert it into unix time (https://en.wikipedia.org/wiki/Unix_time) (2p)

*Documentation (12 points):*
- Generate docstrings for your modules and functions (4 points).
- Generate one figure of your GPS motion in a x,y plane (2p)
- Generate one figure of your direction  in the elevator over time (2p
- Describe how to run your code (4 points).

*Tools (7 points):*
- Write unit tests for all moduls (4 points).
- Choose an appropiate license for your project (1 point)
- Add a bibliography in your documentation and references to all sources you used (2 points).
