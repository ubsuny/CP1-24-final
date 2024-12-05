"""
This module contains functions to find the direction of motion from acceleration data
"""

import numpy as np
import matplotlib.pyplot as plt

def velocity(t, a, v_0):
    '''
    This function calculates velocity given a time range, 
    acceleration at each instance of time, and an initial velocity

    Parameters:
    - t: Time values at which we have a defined acceleration and at which we will find the velocity
    - a: acceleration at each time, t
    - v_0: Initial velocity. Generally zero.

    Returns:
    - v: Velocity at each time t
    '''

    #Calculate the time step differences
    dt = np.diff(t, prepend=t[0])
    #Perform cumulative trapezoidal integration over acceleration to get velocity
    v = v_0 + np.cumsum(a * dt)
    return v

def direction_of_motion(ax,ay,az,t, i):
    '''
    This function plots the direction of motion at a given time, t[i] given acceleration data. 
    The initial velocity is always set to zero. This is to avoid pylint complaining about
    too many positional arguments. 

    Parameters:
    - t: Time values at which we have a defined acceleration and at which we will find the velocity
    - ax: x-acceleration at each time, t
    - ay: y-acceleration at each time, t
    - az: z-acceleration at each time, t
    - i: index of t at which we find the direction of motion. 
          Must be given as a list of all time indices of interest

    Returns:
    - 3d plot of normalized vector v 
    '''

    v_0 = 0

    vx = velocity(t,ax,v_0)
    vy = velocity(t,ay,v_0)
    vz = velocity(t,az,v_0)

    # Create a 3D plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

   # Plot normalized vectors using quiver
    ax.quiver(0, 0, 0, vx[i], vy[i], vz[i], color='b', length=1, normalize=True)

    # Label axes
    ax.set_xlabel(r'$v_x (m/s)$')
    ax.set_ylabel(r'$v_y (m/s)$')
    ax.set_zlabel(r'$v_z (m/s)$')

    ax.set_xlim([-1, 1])  # Set the range for x-axis
    ax.set_ylim([-1, 1])  # Set the range for y-axis
    ax.set_zlim([-1, 1])  # Set the range for z-axis

    ax.set_title(f'Direction of Motion at times t $\\in$ ({t[i[0]]}, {t[i[-1]]})')

    plt.show()
