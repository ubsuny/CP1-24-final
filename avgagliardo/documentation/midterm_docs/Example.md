```python
# import the modules
import utils
import trip

# Get ahold of the GpsTrip class object
GpsTrip = trip.GpsTrip

# Get the GPS plotting method to plot the trip's path
gps_plot = utils.plot_gpstrip_segments_with_color

# Create a GPS trip by passing in the name of the experiment. This 
# must be the same as the directory name in which the 'Raw Data.csv' 
# file resides. 
gps_trip_tri = GpsTrip("AVG001_gps_tri_walk_1")

# Inspect some data
print(gps_trip_tri.get_raw_frame)
print(gps_trip_tri.get_raw_frame_meta)
print(gps_trip_tri.data)
print(gps_trip_tri.segments)

# Report on the trip's summary
print(gps_trip_tri.report_trip_summary())

# set a save path (if you want to save the plot)
save_path = "../data/img/"

# plot the trip using a gps plotter, save it, and use the given title
gps_plot(gps_trip_tri, save_path = save_path, title="GPS Trip: Triangle Walk - Displacement in Degrees (°)")
gps_plot(gps_trip_circ, save_path = save_path, title="GPS Trip: Circle Walk - Displacement in Degrees (°)")


```

    
    --------------------
    Trip Summary:
    --------------------
    Experiment Name: AVG001_gps_tri_walk_1
    Type of trip: GPS
    Start time (UTC): 2024-10-24 18:28:32.723 UTC-04:00
    Start time (Unix): 1729808912
    Duration: 234 seconds
    Number of frames: 234
    Total planar distance traveled: 307.12 meters
    Total curved distance traveled: 323.09 meters
    <bound method TripBase.get_raw_frame of <trip.GpsTrip object at 0x000001D2A99724F0>>
    <bound method TripBase.get_raw_frame_meta of <trip.GpsTrip object at 0x000001D2A99724F0>>
           Time (s)  Latitude (°)  Longitude (°)  Altitude (m)  Altitude WGS84 (m)
    0      1.186602     43.001646     -78.791056    191.412668          156.701355
    1      1.193750     43.001653     -78.791057    191.617821          156.906494
    2      2.200982     43.001657     -78.791138    184.022573          149.311218
    3      3.217686     43.001618     -78.791194    176.951898          142.240601
    4      4.216440     43.001631     -78.791175    177.473646          142.762329
    ..          ...           ...            ...           ...                 ...
    229  229.219687     43.001580     -78.791171    175.622845          140.911621
    230  230.223092     43.001577     -78.791185    175.410197          140.698975
    231  231.214768     43.001574     -78.791197    175.888039          141.176819
    232  232.210805     43.001573     -78.791207    175.803627          141.092407
    233  233.213966     43.001572     -78.791212    175.861122          141.149902
    
    [234 rows x 5 columns]
            start_t      stop_t  start_lat  start_long   start_alt    end_lat  \
    0      1.186602    1.193750  43.001646  -78.791056  191.412668  43.001653   
    1      1.193750    2.200982  43.001653  -78.791057  191.617821  43.001657   
    2      2.200982    3.217686  43.001657  -78.791138  184.022573  43.001618   
    3      3.217686    4.216440  43.001618  -78.791194  176.951898  43.001631   
    4      4.216440    5.221717  43.001631  -78.791175  177.473646  43.001644   
    ..          ...         ...        ...         ...         ...        ...   
    228  228.225245  229.219687  43.001581  -78.791156  175.712076  43.001580   
    229  229.219687  230.223092  43.001580  -78.791171  175.622845  43.001577   
    230  230.223092  231.214768  43.001577  -78.791185  175.410197  43.001574   
    231  231.214768  232.210805  43.001574  -78.791197  175.888039  43.001573   
    232  232.210805  233.213966  43.001573  -78.791207  175.803627  43.001572   
    
          end_long     end_alt     lat_delta  long_delta  degree_distance  \
    0   -78.791057  191.617821  7.530000e-06   -0.000001         0.000008   
    1   -78.791138  184.022573  4.050000e-06   -0.000081         0.000081   
    2   -78.791194  176.951898 -3.946000e-05   -0.000055         0.000068   
    3   -78.791175  177.473646  1.338000e-05    0.000018         0.000023   
    4   -78.791169  178.334874  1.300000e-05    0.000007         0.000015   
    ..         ...         ...           ...         ...              ...   
    228 -78.791171  175.622845 -1.270000e-06   -0.000015         0.000015   
    229 -78.791185  175.410197 -2.500000e-06   -0.000014         0.000014   
    230 -78.791197  175.888039 -2.710000e-06   -0.000012         0.000013   
    231 -78.791207  175.803627 -1.850000e-06   -0.000010         0.000010   
    232 -78.791212  175.861122 -9.100000e-07   -0.000005         0.000005   
    
         planar_distance  curved_distance  
    0           0.844493         0.868129  
    1           6.607338        10.062161  
    2           6.297028         9.463465  
    3           2.114184         2.175301  
    4           1.550894         1.772441  
    ..               ...              ...  
    228         1.196188         1.198171  
    229         1.142436         1.160796  
    230         1.045046         1.148043  
    231         0.850820         0.854046  
    232         0.448767         0.451935  
    
    [233 rows x 13 columns]
    
    --------------------
    Trip Summary:
    --------------------
    Experiment Name: AVG001_gps_tri_walk_1
    Type of trip: GPS
    Start time (UTC): 2024-10-24 18:28:32.723 UTC-04:00
    Start time (Unix): 1729808912
    Duration: 234 seconds
    Number of frames: 234
    Total planar distance traveled: 307.12 meters
    Total curved distance traveled: 323.09 meters
    


    
![png](output_0_1.png)
    



    
![png](output_0_2.png)
    



```python
# import the modules
import trip
import utils

# Get ahold of the GpsTrip class object
AccelTrip = trip.AccelTrip

# Get some plotting methods from the utils module
acceleration_plot = utils.plot_acceltrip_acceleration_with_color  # acceleration
velocity_plot = utils.plot_acceltrip_velocity_with_color          # velocity
trajectory_plot = utils.plot_3d_trajectory                        # 3d trajectory

# Create two trips, 1 for each way in an elevator ride
trip_up = AccelTrip("AVG001_accel_elevator_up_1")
trip_dn = AccelTrip("AVG001_accel_elevator_down_1")

# Inspect some data members
print(trip_up.get_raw_frame)
print(trip_up.get_raw_frame_meta)
print(trip_up.data)
print(trip_up.segments)

# Report on the trip summary
print(trip_up.report_trip_summary())

# set a save path (if you want to save the plot)
save_path = "../data/img/"

# Create some plots using our ascending trip data
acceleration_plot(trip_up, component='z', save_path = save_path, title="Elevator Ascent: Acceleration Along Z-Axis")
velocity_plot(trip_up, component='z', save_path = save_path, title="Elevator Ascent: Velocity Along Z-Axis")
trajectory_plot(trip_up, save_path = save_path, title="Elevator Ascent: 3d Trajectory")

# ADDITIONAL FEATURES
# You can select a specific component 
acceleration_plot(trip_up, component='x', title="Elevator Ascent: Acceleration Along X-Axis")
print("Plotted only the x-component of acceleration")

# Or use the total magnitude
acceleration_plot(trip_up, component='total', title="Elevator Ascent: Total Magnitude of Acceleration")
print("Plotted only the total magnitude of acceleration")

# You can use a compression factor while plotting
acceleration_plot(trip_up, component='z', compression_factor=2, title="Elevator Ascent: Acceleration Along Z-Axis")
print("Plotted using compression factor of 2")
      
# you can optionally skip data points
acceleration_plot(trip_up, component='z', step=15, title="Elevator Ascent: Acceleration Along Z-Axis")
print("Plotted only every 5th data point")

# you can toggle the connecting lines
acceleration_plot(trip_up, component='z', connect_points=False, step=15, title="Elevator Ascent: Acceleration Along Z-Axis")
print("Plotted without connecting lines")



```

    <bound method TripBase.get_raw_frame of <trip.AccelTrip object at 0x000001D2A9AFC040>>
    <bound method TripBase.get_raw_frame_meta of <trip.AccelTrip object at 0x000001D2A9AFC040>>
               time   accel_x   accel_y   accel_z  accel_absolute
    0      0.052050  0.000000  0.000000  0.000000        0.000000
    1      0.056799 -0.004799 -0.007201  0.005985        0.010521
    2      0.061547  0.006325  0.001541  0.002396        0.006936
    3      0.066296 -0.009659  0.017289 -0.001793        0.019885
    4      0.071045 -0.013490 -0.001520  0.002396        0.013785
    ...         ...       ...       ...       ...             ...
    6043  28.747989  0.044962  0.047661 -0.096333        0.116504
    6044  28.752738  0.011331 -0.016883 -0.014810        0.025154
    6045  28.757486 -0.031516 -0.023940  0.042295        0.057924
    6046  28.762235 -0.024149 -0.014204 -0.024534        0.037241
    6047  28.766984 -0.016843  0.007264  0.037398        0.041654
    
    [6048 rows x 5 columns]
            start_t     stop_t   delta_t   accel_x   accel_y   accel_z  \
    0      0.052050   0.056799  0.004749  0.000000  0.000000  0.000000   
    1      0.056799   0.061547  0.004749 -0.004799 -0.007201  0.005985   
    2      0.061547   0.066296  0.004749  0.006325  0.001541  0.002396   
    3      0.066296   0.071045  0.004749 -0.009659  0.017289 -0.001793   
    4      0.071045   0.075793  0.004749 -0.013490 -0.001520  0.002396   
    ...         ...        ...       ...       ...       ...       ...   
    6042  28.743241  28.747989  0.004749  0.019080 -0.035743 -0.013350   
    6043  28.747989  28.752738  0.004749  0.044962  0.047661 -0.096333   
    6044  28.752738  28.757486  0.004749  0.011331 -0.016883 -0.014810   
    6045  28.757486  28.762235  0.004749 -0.031516 -0.023940  0.042295   
    6046  28.762235  28.766984  0.004749 -0.024149 -0.014204 -0.024534   
    
          total_acceleration  velocity_x  velocity_y  velocity_z  
    0               0.000000    0.000000    0.000000    0.000000  
    1               0.010521   -0.000023   -0.000034    0.000028  
    2               0.006936    0.000030    0.000007    0.000011  
    3               0.019885   -0.000046    0.000082   -0.000009  
    4               0.013785   -0.000064   -0.000007    0.000011  
    ...                  ...         ...         ...         ...  
    6042            0.042660    0.000091   -0.000170   -0.000063  
    6043            0.116504    0.000214    0.000226   -0.000457  
    6044            0.025154    0.000054   -0.000080   -0.000070  
    6045            0.057924   -0.000150   -0.000114    0.000201  
    6046            0.037241   -0.000115   -0.000067   -0.000117  
    
    [6047 rows x 10 columns]
    
    --------------------
    Trip Summary:
    --------------------
    Experiment Name: AVG001_accel_elevator_up_1
    Type of trip: ACCEL
    Start time (UTC): 2024-10-24 18:15:56.038 UTC-04:00
    Start time (Unix): 1729808156
    Duration: 28 seconds
    Number of frames: 6048
    


    
![png](output_1_1.png)
    



    
![png](output_1_2.png)
    



    
![png](output_1_3.png)
    



    
![png](output_1_4.png)
    


    Plotted only the x-component of acceleration
    


    
![png](output_1_6.png)
    


    Plotted only the total magnitude of acceleration
    


    
![png](output_1_8.png)
    


    Plotted using compression factor of 2
    


    
![png](output_1_10.png)
    


    Plotted only every 5th data point
    


    
![png](output_1_12.png)
    


    Plotted without connecting lines
    


```python

```
