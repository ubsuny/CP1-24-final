from acceleration_calculate import calculate_directions_from_csv

CSV_FILE = r'D:\Projects\CP1-24-midterm\dhamalakamal\Data\Accl_bottom_top.csv'
calculated_directions = calculate_directions_from_csv(CSV_FILE)
for index, (azimuth, elevation) in enumerate(calculated_directions):
    print(f"Direction for point {index + 1}: Azimuth = {azimuth:.2f}°, Elevation = {elevation:.2f}°")

