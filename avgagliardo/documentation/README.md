#README

##Data & Preprocessing
Data was recorded with phyphox, shipped as a csv, and the preprocessed to conform to naming conventions.
Metadata also prepared in a similar manner. Metadata lives in a markdown file named after its experiment.

##Files
###Trial Data
Experimental data lives in the /data/AVG002_gps_sine_walk/experimental_data/ directory.
- Experimental GPS data is stored as a .csv file.
- Each run's metadata is stored as a .md file with the same filename.
###Module: final.py
The final.py module implements the algorithm tasks. It can be found in the /code/ directory.
###Module: test_final.py
The test_final.py module implements the unit tests for the final.py module's functionality. It can also be found in the /code/ directory and can be run with `pytest test_final.py`.


##Project Tasks
###Merging Task
Personal contributions from the midterm project repository have been merged into the final project repository, and the full commit history has been preserved. Midterm project files will live side-by-side with the final project files, however final project files have been specifically named and documented for the sake of clarity.
###Data Task
Data files live in the /data/AVG002_gps_sine_walk/experimental_data/ directory.
###Algorithm Task
####Function that converts Fahrenheit to Kelvin
Implemented in final.py, test implemented in test_final.py. 
