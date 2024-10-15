# Midterm 24 - data read and process

This is the midterm for CP1-24 based on getting and using research data using the phyphox app (https://phyphox.org).

*Data: 9 points*
- Use the Location (GPS) app to record the spatial locations of how you move along the outline of the  following two geometric elements:
   - a circle with a diameter of at least $d = \frac{500\rm{ft}}{\pi}$
   - a right triangle with at least $a = 80 \rm{ft}$ and $b = 50 \rm{yd}$
- Use the Acceleration (without g) app to record the linear accelaration in $x,y,z$ for the following events:
   - Take the elevator in any UB north campus building from the lowest to the top floor
   - Take the elevator in any UB north campus building from the top to the lowest floor
- After each experiment export all collected data as csv files and generate an additional markdown file (named following this pattern: `"unique id"_"experiment name".md`, e.g. `LL009_gps_circle_run09.md`) with the following meta information:
   - Date and time of the experiment
   - Current weather conditions
   - Any additional information regarding the experiment

*Algorithm (15 points):*
- write the following python modules that implement importable functions:
   - for a unit converter for feet and yard to an SI equivalent (2p)
   - that calculate the distance between two recorded adjacent GPS positions (bonus point if you consider the earth not being flat) (5p)
   - that calculate the direction of motion from the recorded acceleration data (5p)
   - to read out the data and time from each metafile and convert it into unix time (https://en.wikipedia.org/wiki/Unix_time) (3p)
- All python modules have to be linted using pylint (https://pylint.readthedocs.io/en/stable/) and get a full score using the default settings to get full points.
- Only python libraries listed in requirements.txt can be used

*Documentation (12 points):*
- Generate docstrings for your modules and functions (4 points).
- Generate *one* figure of your GPS motions in a x,y plane and save it as a png file (2 points)
- Generate *one* figure of your directions in the elevator over time and save it as a png file (2 points)
- Describe in a seperate markdown file how to run your code (4 points).

*Tools (6 points):*
- Write unit tests using the pytest framework for all modules. Tests have to be in a seperate file following the pattern `test_*.py` (4 points).
- Add a bibliography in your documentation and references to all sources you used (2 points).

## How to work on your midterm project on github

1. you have to work on a fork of the original ubsuny project
2. you have to submit your work to the original ubsuny project via pull requests
3. all your work has to be in a folder that is called "your github username"
4. inside that folder you have to seperate you work in several subfolders:
     - data
     - code
     - documentation (figures, how to run the code)



## License
This midterm project is under GPLv3 license (see LICENSE file).
If you use any code or other external content that you didn't creat it is your responsibility if that code is compatible to be included in a GPLv3 project. Any code that is used from somewhere else has to be attributed in the NOTES file in the folloowing form:

``` text
date / commit hash:
"Your github Username": filename - Lines of code and Attribution
```
