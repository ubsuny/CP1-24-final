# Analysis and Explanation of Plots

## Plot 1: Filtered FFT of Mean Sine Walk Frequency
### Description  
This plot displays the **filtered FFT of the mean sine walk frequency** which shows, from the data across all run, the dominant frequency components.

### Observations  
1. Dominant periodic components in the data becasue of the clear peak observed at low frequency which is a clear indicator of periodic motion.
2. For low frequency dominant signals,as the frequency increases the amplitude decreases slowly.
3. The sharp decline signifies the scarcity of higher-frequency components.

## Plot 2: FFT of GPS Walks
### Description  
This plot shows the **FFT of individual GPS walks** with each run represented by a different color. To know more about the frequency components , the FFT analysis is complulsary.

### Observations  
1. All runs show a **sharp peak at low frequencies**, suggesting the presence of low-frequency periodic motion across multiple runs.  
2. The amplitude decreases rapidly as the frequency increases, which confirms that the signal is dominated by low-frequency components.  
3. Since the individual GPS walk patterns mimic a sine wave, and they can not be 100% accurate, the observations shows that there is experimental noise differences.

## Plot 3: Sine Wave Fits for All Runs
### Description  
This plot displays the **sine wave fits for all runs** as dashed lines. Each line represents the best-fit sine wave corresponding to the GPS data for a given run.

### Observations  
1. From the wave fits, we can see there is a significant difference in amplitude and phase related to different runs. This could be because of the not-so-periodic motion
2. It appears to be noisy which would clearly suggest that there are some deviations from ideal sine motion.
3. It does not show smooth periodic structure given the amount of noise.

### Conclusion  
The sine wave fits partially align with the intended periodic motion. The variability across runs indicates that while periodic behavior exists, the motion was not perfectly sinusoidal due to experimental limitations.
