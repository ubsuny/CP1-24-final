## Derivation behind angle for arclength connecting any two vectors represented in sphereical coordinates:

You can get the connecting angle between two vectors as follows:

$$|r_1| |r_2| \cos(\alpha) = r_1 \cdot r_2$$

In three dimensional cartesian:

$$r_1 \cdot r_2 = x_1x_2 + y_1y_2 + z_1z_2$$

$$x = \rho ~ \cos(\theta)\sin(\phi), \quad y=\rho ~ \sin(\theta)\sin(\phi),\quad z=\rho ~ \cos(\phi)$$

$$\therefore ~ r_1 \cdot r_2 = \rho^2 ~ \left( \cos(\theta_1)\sin(\phi_1)\cos(\theta_2)\sin(\phi_2) +  \sin(\theta_1)\sin(\phi_1)\sin(\theta_2)\sin(\phi_2)\right) + \cos(\phi_1)\cos(\phi_2)$$

The following trigonometric identities from the second link in references below:

$$ \cos(\theta_1)\cos(\theta_2) = \frac{1}{2}\left( \cos(\theta_1+\theta_2) + \cos(\theta_1-\theta_2) \right) $$

$$ \sin(\theta_1)\sin(\theta_2) = \frac{1}{2}\left( \cos(\theta_1-\theta_2) - \cos(\theta_1+\theta_2) \right) $$

$$ \therefore ~ r_1 \cdot r_2 = \rho^2 ~ \left(\frac{1}{2}\left( \cos(\theta_1+\theta_2) + \cos(\theta_1-\theta_2) + \cos(\theta_1-\theta_2) - \cos(\theta_1+\theta_2) \right)\sin(\phi_1)\sin(\phi_2) + \cos(\phi_1)\cos(\phi_2)\right) $$

$$ \therefore ~ r_1 \cdot r_2 = \rho^2 ~ \left(\frac{1}{2}\left( 2\cos(\theta_1-\theta_2) \right)\sin(\phi_1)\sin(\phi_2) + \cos(\phi_1)\cos(\phi_2)\right) $$

$$ r_1 \cdot r_2 = \rho^2 ~ \left(\cos(\theta_1-\theta_2)\sin(\phi_1)\sin(\phi_2) + \cos(\phi_1)\cos(\phi_2)\right) $$

Relating this back to the first expression with $\alpha$ and knowing $\rho$ is just the radius of Earth ($R$) in this case:

$$ \cos(\alpha) =  \frac{R^2}{R^2}~ \left(\cos(\theta_1-\theta_2)\sin(\phi_1)\sin(\phi_2) + \cos(\phi_1)\cos(\phi_2)\right)$$

$$\boxed{ \therefore ~ \alpha = \arccos\left( \cos(\theta_1-\theta_2)\sin(\phi_1)\sin(\phi_2) + \cos(\phi_1)\cos(\phi_2) \right) }$$

And hence this angle can be used for an arclength between two gps coordinates while acknowledging the (approximately) sphereical geometry of Earth.

## General citations:

1. Cartesian to spherical transformation:
    - [https://math.libretexts.org/Courses/Mount_Royal_University/MATH_2200%3A_Calculus_for_Scientists_II/7%3A_Vector_Spaces/5.7%3A_Cylindrical_and_Spherical_Coordinates#:~:text=To%20convert%20a%20point%20from,and%20z=ρcosφ.](https://math.libretexts.org/Courses/Mount_Royal_University/MATH_2200%3A_Calculus_for_Scientists_II/7%3A_Vector_Spaces/5.7%3A_Cylindrical_and_Spherical_Coordinates#:~:text=To%20convert%20a%20point%20from,and%20z=ρcosφ.)

2. Trigonometric identities used in derivation:
   - [https://andymath.com/product-sum-identities/](https://andymath.com/product-sum-identities/)

3. Website used to establish expected values for gps arclength:
   - [https://www.latlong.net](https://www.latlong.net)

4. Reference for Unix Time:
   - [https://en.wikipedia.org/wiki/Unix_time](https://en.wikipedia.org/wiki/Unix_time)
    
5. Unit converter for expected unit conversions:
   - [https://www.unitconverters.net](https://www.unitconverters.net)

6. Generating and cleaning up mock csv file with pandas:
   - [https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.to_csv.html](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.to_csv.html)

7. Radius of Earth source:
    - [https://nssdc.gsfc.nasa.gov/planetary/factsheet/earthfact.html](https://nssdc.gsfc.nasa.gov/planetary/factsheet/earthfact.html)

8. Numpy angle converter:
    - [https://numpy.org/devdocs/reference/generated/numpy.deg2rad.html](https://numpy.org/devdocs/reference/generated/numpy.deg2rad.html)

9. Default arguments in python:
    - [https://www.geeksforgeeks.org/default-arguments-in-python/](https://www.geeksforgeeks.org/default-arguments-in-python/)

10. Numpy linalg.norm:
    - [https://numpy.org/doc/stable/reference/generated/numpy.linalg.norm.html](https://numpy.org/doc/stable/reference/generated/numpy.linalg.norm.html)

11. Reference for unit testing formatting:
    - [https://www.lambdatest.com/learning-hub/unit-testing](https://www.lambdatest.com/learning-hub/unit-testing)

12. Pandas CSV manipulation (important):
    - [https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.to_csv.html](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.to_csv.html)

13. Figuring out why pylint was yelling at me:
    - [https://stackoverflow.com/questions/14027417/what-does-pylints-too-few-public-methods-message-mean](https://stackoverflow.com/questions/14027417/what-does-pylints-too-few-public-methods-message-mean)

14. Pandas .to_datetime:
    - [https://pandas.pydata.org/docs/reference/api/pandas.to_datetime.html](https://pandas.pydata.org/docs/reference/api/pandas.to_datetime.html)
