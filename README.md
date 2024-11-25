# Final 24 - data anlysis and presentation

This is the final project for CP1-24 based on getting and using research data using the phyphox app (https://phyphox.org).
The final is based on the midterm project.

As a first step all content from YOUR midterm project has to be merged into the final repository while preserving all commit history (5 points).

*Data task: (4 points)*
- Use the Location (GPS) app to record the spatial locations of how you move along the outline of a sine wave together with a markdown file (named following this pattern: `"unique id"_"experiment name".md`, e.g. `LL008_sinewalk.md`) which includes the following inforamtion:
   - Date and time of the experiment
   - Envirnment temperature
   - Any additional information regarding the experiment
 
Repeat this 20 times keeping the periodic parameters the same and record each time one csv and one markdown file.

*Algorithm  task: (28 points):*
- write ONE python module that implements importable functions including docstrings and pytest unit tests (seperate file starting with `test_`) for all functions:
   - function that converts Fahrenheit to Kelvin (3)
   - parser that reads out the temperature of one markdown file (4)
   - filename lister that generates programmatically (using the python `os` library) a list of your markdown file based on a filename filter (eg filename contains `experimentname` (4)
   - non-linear fitting in pure python which includes the functionality to specify the step number of $2^n$ (6)
   - numpy wrapper for fft, inverse fft, including functionality that checks for non-equidistant data (6)
   - pure python (no numpy) to calculate the frequency axis in useful units (4)
- All python modules have to be linted using pylint (https://pylint.readthedocs.io/en/stable/) and get a full score using the default settings to get full points.
- Only python libraries listed in requirements.txt can be used

*Documentation task: (10 points):*
Generate one jupyter notebook that includes the following functionality:
- In the first markdown cell describe how to run your code and a bibliography to all sources you used (3 points).
- Generate *one* figure of your GPS motion with the axis in meter and the origin at your starting point, including a non-linear fit using your own function for each of the 20 repetitions including a legend that includes the temperature in Kelvin for each run that was read out from the markdwon file with the functions from the Algorithm task (3 points).
- Generate *one* figure with the FFT (using functions from the algo task) of each of your walks with the frequency in the unit of 1/100m (2 points).
- Generate *one* figure that shows the inverse FFT of the filtered mean value of your sine walk frequency (2 points).

## How to work on your midterm project on github

1. you have to work on a fork of the original ubsuny *final* project
2. you have to submit your work to the original ubsuny *final* project via a *single* pull request
3. all your work has to be in a folder that is called "your github username"
4. inside that folder you have to seperate you work in several subfolders:
     - data
     - code
     - documentation (contains jupyter and figures)

## License
This midterm project is under GPLv3 license (see LICENSE file).
If you use any code or other external content that you didn't creat it is your responsibility if that code is compatible to be included in a GPLv3 project. Any code that is used from somewhere else has to be attributed in the NOTES file in the folloowing form:

``` text
date / commit hash:
"Your github Username": filename - Lines of code and Attribution
```
 make sure if you edit the file to always sync with the base repository first in case there have been changes by others.
