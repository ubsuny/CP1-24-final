
# **README: Running the Final Project Code**  

This README provides a **detailed guide** on how to set up, run, and test the project, including the **Final Project**, **Midterm Project Integration**, and **Unit Testing** setup using `pytest`.  

---

## **Project Overview**  

The project is designed for GPS-based experiment analysis using data collected from the Phyphox app. It includes modules for unit conversions, GPS distance calculations, motion direction determination, Unix time conversion, and advanced data analysis functions such as FFT, non-linear fitting, and more.  

---

## **Project Structure**  

```plaintext
/jkblc
    ├── data                # Contains experiment data files (CSV + Markdown)
    ├── code
    │   ├── final.py        # Main code module
    │   ├── test_final.py   # Unit tests using pytest
    │   └── requirements.txt
    └── documentation
        ├── final.ipynb     # Jupyter Notebook with results
        └── figures         # Figures generated by the Notebook
```

---

## **Setup Instructions**  

### **1. Clone the Repository**  
```bash
git clone https://github.com/jkblc/CP1-24-final/dev.git
cd jkblc
```

### **2. Create and Activate a Virtual Environment**  
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### **3. Install Dependencies**  
```bash
pip install -r code/requirements.txt
```

---

## **Running the Project Code**  

### **Running the Jupyter Notebook**
```bash
jupyter notebook documentation/final.ipynb
```

---

### **Running the Tests with Pytest**  

To execute the unit tests:  

```bash
cd code
pytest test_final.py
```

---

# **Explanation of Functions**  

---

## **Midterm Project Functions**  

---

### **1. Unit Conversion Functions**  

#### **`feet_to_meters(feet)`**  
- **Description:** Converts a measurement in feet to meters.  
- **Formula:** `1 foot = 0.3048 meters`  
- **Example Usage:**  
```python
from final import feet_to_meters
print(feet_to_meters(10))  # Output: 3.048
```

#### **`yards_to_meters(yards)`**  
- **Description:** Converts a measurement in yards to meters.  
- **Formula:** `1 yard = 0.9144 meters`  
- **Example Usage:**  
```python
from final import yards_to_meters
print(yards_to_meters(5))  # Output: 4.572
```

---

### **2. GPS Distance Calculation**  

#### **`haversine(coord1, coord2)`**  
- **Description:** Calculates the great-circle distance between two points on Earth using the **Haversine formula**.  
- **Inputs:** `coord1 = (lat1, lon1)`, `coord2 = (lat2, lon2)`  
- **Returns:** Distance in meters.  
- **Example Usage:**  
```python
from final import haversine
paris = (48.8566, 2.3522)
london = (51.5074, -0.1278)
print(haversine(paris, london))  # Output: ~343,774 meters
```

---

### **3. Motion Direction from Acceleration Data**  

#### **`direction_from_acceleration(ax, ay, az)`**  
- **Description:** Calculates motion direction based on 3D acceleration data.  
- **Inputs:** `ax`, `ay`, `az` (acceleration in 3D axes).  
- **Returns:** `(theta, phi)` - motion direction in degrees.  
- **Example Usage:**  
```python
from final import direction_from_acceleration
theta, phi = direction_from_acceleration(1, 0, 0)
print(f"Polar angle: {theta}, Azimuthal angle: {phi}")
```

---

### **4. Unix Time Conversion Functions**  

#### **`convert_to_unix_time(date_string, date_format)`**  
- **Description:** Converts a date string to **Unix time**.  
- **Inputs:**  
  - `date_string` - Date in string format (e.g., `"2024-12-09 14:30:00"`).  
  - `date_format` - Corresponding date format (`"%Y-%m-%d %H:%M:%S"`).  
- **Returns:** Unix timestamp as an integer.  
- **Example Usage:**  
```python
from final import convert_to_unix_time
unix_time = convert_to_unix_time("2024-12-09 14:30:00", "%Y-%m-%d %H:%M:%S")
print(unix_time)  # Example Output: 1731082200
```

---

# **Troubleshooting Tips**  

1. **Pytest Not Found:**  
   Install using:  
   ```bash
   pip install pytest
   ```

2. **Module Not Found:**  
   Ensure the working directory contains all the correct project files. Use `sys.path` adjustments if needed.

---