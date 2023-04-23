import serial
import numpy as np
import matplotlib.pyplot as plt
import os

# Set up the serial port
port = serial.Serial('COM4', baudrate=115200)

# Create the output directories
dir1 = 'plots_1'
dir2 = 'plots_2'
dir3 = 'plots_3'
os.makedirs(dir1, exist_ok=True)
os.makedirs(dir2, exist_ok=True)
os.makedirs(dir3, exist_ok=True)

# Loop to read data from the serial port
while True:
    # Read data from the serial port
    data = port.readline().decode().strip()

    # Parse the data into an array of floats
    data = np.array(data.split(',')).astype(float)

    # Loop over the three sets of data
    for i in range(3):
        # Extract the data for this sensor
        sensor_data = data[i::3]

        # Compute the spectrogram
        spec, freqs, bins, im = plt.specgram(sensor_data, Fs=1000, NFFT=1024, cmap='gray')


        # Save the spectrogram image to the appropriate directory
        if i == 0:
            path = os.path.join(dir1, 'spec.png')
        elif i == 1:
            path = os.path.join(dir2, 'spec.png')
        else:
            path = os.path.join(dir3, 'spec.png')
        plt.imsave(path, spec, cmap='gray')