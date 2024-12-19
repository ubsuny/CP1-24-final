# Running the Code in This Directory

This guide provides instructions on how to run the code files available in this directory. Each script has specific functionality that can be utilized to process data, perform calculations, and generate visualizations.

## Prerequisites

- Python 3.x
- Required Python libraries, which include:
  - `pandas`
  - `numpy`
  - `matplotlib`

## Overview of Scripts

### 1. `Direction_Calculator.py`

This script calculates the direction of motion based on raw acceleration data in (x, y, z) components. It normalizes these vectors to obtain unit vectors representing the direction of motion.

**How to Run**:

```sh
python Direction_Calculator.py
```

**Inputs**: 
- CSV files containing acceleration data (`Acceleration_Up_to_Down.csv`, `Acceleration_Down_to_Up.csv`).

**Outputs**: 
- Direction vectors saved in a specified output file or printed to the console.

### 2. `Distance_Calculator.py`

This script calculates distances using the Haversine formula and other geometry-based methods.

**How to Run**:

```sh
python Distance_Calculator.py
```

**Inputs**: 
- Coordinates or CSV files containing latitude and longitude data.

**Outputs**: 
- Calculated distances printed to the console or saved to an output file.

### 3. `Unit_Converter.py`

This script converts various units, such as length, temperature, and weight, using Google's unit converter capabilities.

**How to Run**:

```sh
python Unit_Converter.py
```

**Inputs**: 
- Units and values to be converted.

**Outputs**: 
- Converted values printed to the console.

### 4. `Unix_Converter.py`

This script performs conversions of Unix timestamps to human-readable formats.

**How to Run**:

```sh
python Unix_Converter.py
```

**Inputs**: 
- Unix timestamps.

**Outputs**: 
- Human-readable date and time printed to the console.

### 5. `GPS_Image_Generator.py`

This script generates GPS-based visualizations from the provided CSV files (`Circle.csv`, `Triangle.csv`).

**How to Run**:

Ensure you have the relevant CSV files (`Circle.csv`, `Triangle.csv`) in the same directory before running:

```sh
python GPS_Image_Generator.py
```

**Inputs**: 
- `Circle.csv`, `Triangle.csv` containing GPS coordinates.

**Outputs**: 
- Generated images of GPS paths saved as `.png` files.

### 6. `Elevator_Image_Generator.py`

This script generates visualizations of elevator movements based on acceleration data. The data for these movements is provided in `Acceleration_Up_to_Down.csv` and `Acceleration_Down_to_Up.csv`.

**How to Run**:

Ensure you have `Acceleration_Up_to_Down.csv` and `Acceleration_Down_to_Up.csv` in the directory before running:

```sh
python Elevator_Image_Generator.py
```

**Inputs**: 
- `Acceleration_Up_to_Down.csv`, `Acceleration_Down_to_Up.csv` containing elevator acceleration data.

**Outputs**: 
- Visualizations of elevator movement saved as `.png` files.

### 7. Markdown Files (`.md`)

The following Markdown files provide documentation and explanations for various components and datasets used in the project:
- `HA007_Elevator_Up.md`
- `HA007_Elevator_Down.md`
- `HA007_Triangle.md`
- `HA007_Circle.md`

These Markdown files can be viewed in any text editor or Markdown viewer to understand the context and explanations of each dataset.

## Example Application

To calculate the direction vectors for elevator movements and generate 3D visualizations, you can use the following sequence of commands:

1. Run `Direction_Calculator.py` to calculate direction vectors.

   ```sh
   python Direction_Calculator.py
   ```

2. Use `Elevator_Image_Generator.py` to generate visualizations based on these vectors.

   ```sh
   python Elevator_Image_Generator.py
   ```
