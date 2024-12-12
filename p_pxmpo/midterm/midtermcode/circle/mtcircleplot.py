"""
This script reads latitude and longitude data from a CSV file and plots the 
coordinates on a scatter plot using Matplotlib. If the specified file is not
found, an error message will be displayed.
"""
import csv
import matplotlib.pyplot as plt

latitude = []
longitude = []

# Ask the user for the CSV file name
file_name = input("Enter the CSV file name (with .csv extension): ")

# Open the CSV file
try:
    with open(file_name, 'r', encoding='utf-8') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        next(csv_reader)  # Skip the header row
        for row in csv_reader:
            latitude.append(float(row[1]))  # Assuming latitude is in the second column
            longitude.append(float(row[2]))  # Assuming longitude is in the third column

    # Create a scatter plot
    plt.figure(figsize=(8, 6))
    plt.scatter(longitude, latitude, c='blue', marker='o')

    # Add labels and title
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.title('Plot of Latitude and Longitude Points')

    # Show the plot
    plt.show()

except FileNotFoundError:
    print(f"Error: The file '{file_name}' was not found. Please check the filename "
           "and try again.")
except ValueError:
    print("Error: Could not convert data to float. Please ensure the CSV file "
          "contains valid numbers.")
except OSError as e:
    print(f"Error: An OS error occurred: {e}")
