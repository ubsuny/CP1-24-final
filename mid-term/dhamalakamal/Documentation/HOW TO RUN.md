# How to run the code 
This is how to run my code guide. Please follow the steps below. 

This guide explains how to run the program, execute tests, and set up your development environment.

## Prerequisites
- Python 3.7+
- Download Python from python.org.
- Virtual Environment (optional but recommended)
- Install virtualenv to manage dependencies:
  `pip install virtualenv`
- Required Python Packages
- Install the following packages (add them to requirements.txt):
    ```
    io
    os
    StringIO
    pytest
    pandas
    numpy
    ```
- To install dependencies:
`pip install -r requirements.txt`

## Setting Up
- Clone the Repository
    - Clone the project to your local machine:
    ```
    git clone https://github.com/dhamalakamal/CP1-24-midterm.git
    cd CP1-24-midterm
    git checkout mid_term
    ```
- Set Up a Virtual Environment 
    - Create a virtual environment:
    ```
    virtualenv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```
- Install Dependencies
    - If you have a requirements.txt, run:
    `pip install -r requirements.txt`

## Running the Program
- Run the Main Program
- To execute the main program (assuming it's in main.py):
    `python main.py`
- Execute Unit Tests
    - Install pytest (if not already installed):
    `pip install pytest`
- Run the tests:
    `pytest file_name.py`
    - Example Tests
    - We have all test into single file test_all.py:
    ```
    pytest test_all.py
    ```

## Common Issues & Troubleshooting
- 

## Notes
Make sure input data (CSV, markdown, etc.) is in the correct format.
Adjust mock data in test files as needed.
For large datasets, optimize functions or reduce test data size.

## Conclusion
Follow these steps to set up, run, and test the program successfully. For questions or contributions, refer to the project’s README or submit an issue on GitHub.
