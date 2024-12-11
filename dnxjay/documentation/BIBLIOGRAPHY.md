# Bibliography

This document lists the resources and references used in the development of this project.

## General References
- Phyphox App Documentation: https://phyphox.org
  - Used as the source for collecting acceleration and GPS data for analysis.

## Module References

### `unit_converter.py`
- National Institute of Standards and Technology (NIST). "International System of Units (SI) Conversion Factors."
  - For conversion factors between feet, yards, and meters.
  - URL: https://www.nist.gov

### `gps_distance.py`
- Wikipedia: Haversine Formula. https://en.wikipedia.org/wiki/Haversine_formula
  - Used to calculate the distance between two latitude and longitude points, accounting for the curvature of the Earth.

- Stack Overflow: "How do I calculate the distance between two GPS coordinates?" https://stackoverflow.com/questions/19412462/getting-distance-between-two-points-based-on-latitude-longitude
  - Reference for implementing the Haversine formula.

### `motion_direction.py`
- Khan Academy. "Vectors and Motion: Calculating Direction Using Arctan."
  - Reference for calculating motion direction from acceleration data.
  - URL: https://www.khanacademy.org

### `unixtime_converter.py`
- Wikipedia: Unix Time. https://en.wikipedia.org/wiki/Unix_time
  - Used to understand and implement Unix time conversion.

- Python `datetime` Documentation: https://docs.python.org/3/library/datetime.html
  - Reference for date and time parsing, formatting, and conversion to Unix time.

### Testing and Development Tools
- Pytest Documentation: https://docs.pytest.org
  - Used as the testing framework for unit tests.

- Pylint Documentation: https://pylint.pycqa.org
  - Used for linting code to maintain quality and adhere to best practices.

---

This bibliography includes links to documentation and articles used in developing, implementing, and verifying the functionalities in this project.
