from final import process_files
import os

# Define the path to the folder containing your CSV files
data_dir = os.path.join("..", "data")  # Adjust the folder location as needed

# List of CSV files to process
files = [
    os.path.join(data_dir, "KM001_sinewalk.csv"),
    os.path.join(data_dir, "KM002_sinewalk.csv"),
    os.path.join(data_dir, "KM003_sinewalk.csv"),
    os.path.join(data_dir, "KM004_sinewalk.csv"),
    os.path.join(data_dir, "KM005_sinewalk.csv"),
    os.path.join(data_dir, "KM006_sinewalk.csv"),
    os.path.join(data_dir, "KM007_sinewalk.csv"),
    os.path.join(data_dir, "KM008_sinewalk.csv"),
    os.path.join(data_dir, "KM009_sinewalk.csv"),
    os.path.join(data_dir, "KM010_sinewalk.csv"),
    os.path.join(data_dir, "KM011_sinewalk.csv"),
    os.path.join(data_dir, "KM012_sinewalk.csv"),
    os.path.join(data_dir, "KM013_sinewalk.csv"),
    os.path.join(data_dir, "KM014_sinewalk.csv"),
    os.path.join(data_dir, "KM015_sinewalk.csv"),
]

# Call the process_files function to fit the sine waves and plot the result
process_files(files, output_image='fit_plot.png')