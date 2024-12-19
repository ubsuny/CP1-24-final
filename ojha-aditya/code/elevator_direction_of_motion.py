'''
This module calculates the velocity three vector for each time slice and its direction
'''
import numpy as np
import pandas as pd

def read_data(fname):
    '''
    Function to read the time and acceleration from data files
    '''
    data_elevator = pd.read_csv(fname)
    time = np.array(data_elevator['Time (s)'])
    ax = np.array(data_elevator['Linear Acceleration x (m/s^2)'])
    ay = np.array(data_elevator['Linear Acceleration y (m/s^2)'])
    az = np.array(data_elevator['Linear Acceleration z (m/s^2)'])
    return time, ax, ay, az

def calculate_velocity(time, ax, ay, az):
    '''
    Function to return the array with velocity unit vectors at each time slice, 
        given accelerations in x, y, z directions and time in s. 
 
            Parameters:
                    time : Time array.
                    ax : Acceleration in x-direction.
                    ay : Acceleration in y-direction.
                    az : Acceleration in z-direction.

            Returns:
                    [vx,vy,vz]: List of arrays of velocities in x, y and z directions respectively.
    '''

    time1 = time[:-1]
    time2 = time[1:]
    delta_time = time2-time1

    vx0 = ax[0]*time[0] # calculating initial x-direction velocity
    vy0 = ay[0]*time[0] # calculating initial y-direction velocity
    vz0 = az[0]*time[0] # calculating initial z-direction velocity

    vx = np.cumsum(ax[1:]*(delta_time[:])) + vx0
    # calculating x-direction velocity
    vy = np.cumsum(ay[1:]*(delta_time[:])) + vy0
    # calculating y-direction velocity
    vz = np.cumsum(az[1:]*(delta_time[:])) + vz0
    # calculating z-direction velocity

    vx = np.insert(vx,0,vx0)
    vy = np.insert(vy,0,vy0)
    vz = np.insert(vz,0,vz0)

    #return [vx,vy,vz]
    velocity = np.array([vx,vy,vz])
    velocity_norm = np.sqrt((velocity**2).sum(axis=0))

    return velocity/velocity_norm
    # returning unit vector in direction of velocity
