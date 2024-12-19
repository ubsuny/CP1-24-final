"""
direction_of_motion is a module that
contains functions for determining the
direction of motion in an elevator going up and down.
"""
import numpy as np
import pandas as pd

def get_del_t(times):
    """
    get_del_t goes through a list of times (times)
    and determines the difference between each 
    sequential time.
    """
    #The later time is subtracted from the initial time
    #for each set of sequential times in the times list
    #to produce the dt list.
    dt=[]
    for i in range(len(times)-1):
        dt.append(times[i+1]-times[i])

    #To keep dt the same length as times, the average dt
    #value is appended as the last value.
    s=0
    for i in dt:
        s+=i
    s=s/len(dt)
    dt.append(s)
    return dt

def adjust_acc(acc):
    """
    adjust_acc() eliminates the 
    smaller random fluctuations
    from the data.
    """
    #Any acceleration value lower than .5 is reduced to zero
    for i in acc:
        if i<.5:
            i=0
    return acc

def get_vel(a, dt):
    """
    get_vel gets a list 
    of velocity values from
    acceleration and dt data
    """
    vel=0
    velocities=[]
    l=len(dt)

    #The change in velocity due to each acceleration * time dt
    #is summed up to produce velocity for each point in time.
    for j in range(l):
        vel+=a[j]*dt[j]
        velocities.append(vel)
    return velocities


def get_direction(times, acc):
    """
    get_direction produces a list of
    direction values that define the 
    direction of motion.
    """
    dt=get_del_t(times)
    acc=adjust_acc(acc)
    vel=get_vel(acc,dt)

    #direction is the list of direction values
    #each value in direction is a velocity value
    #divided by the absolute value of itself.
    direction=[]
    for i in vel:
        if i!=0:
            direction.append(i/(np.abs(i)))
        else:
            direction.append(0)
    direction=make_end_zero(direction)
    return direction

def make_end_zero(direction):
    """
    make_end_zero makes absolutely sure
    that the direction of motion at the end
    of the direction list is zero
    """

    #First, it determines whether the initial motion
    #of the elevator is in the positive or negative
    #z-direction
    negative=False
    positive=False
    for i in direction:
        if i>0:
            positive=True
            break
        if i<0:
            negative=True
            break

    #Now, if there is a jostle as the elevator comes to a stop,
    #it will be detected and the rest of the acceleration values
    #after the direction of motion changes will become zero
    l=len(direction)
    if positive is True:
        for i in range(l):
            if direction[i]<0:
                for j in range(i,l):
                    direction[j]=0

    if negative is True:
        for i in range(l):
            if direction[i]>0:
                for j in range(i,l):
                    direction[j]=0

    #If no jostle is detected, the final direction value is
    #set to zero just to make sure that the behavior of the
    #elevator stopping is conveyed.
    if direction[l-1]!=0:
        direction[l-1]=0
    return direction

def acc_reader(path):
    """
    acc_reader reads a file to isolate
    linear acceleration z data and time data
    """
    file =pd.read_csv(path)

    #condition1 and 2 are used to verify that
    #a file with the correct data is input
    condition2=False
    condition1=False
    for c in file:
        if c=="Linear Acceleration z (m/s^2)":
            condition1=True
        if c=="Time (s)":
            condition2=True
    if condition1 is True and condition2 is True:
        acc=file["Linear Acceleration z (m/s^2)"]
        times=file["Time (s)"]
        return acc, times
    return "Bad file. Does not contain Linear Acceleration z(m/s^2) data or Time (s) data"
