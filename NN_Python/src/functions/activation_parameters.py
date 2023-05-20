"""
This function returns the number of channels we want for our activation function, defined by us,
as well as the channel array that gives the velocities we find at the centers of the channels.

It takes as input the data csv file.
"""

import numpy as np

def activation_function_parameters(dataset):

    #myCsvData = np.genfromtxt(csvPath, dtype=np.float32, delimiter=',', skip_header=1)
    dataArray = dataset.to_numpy()
    myVelocityArray = dataArray[:, 1]

    maxVel = np.max(myVelocityArray)
    # minVel = np.min(myVelocityArray) # Min vel is 0 so use 0 instead of this

    # How many triangles do I want?
    num_chan = 10

    # Array containing center of triangles superimposed on top of velocity range
    chan_arr = np.linspace(0.0 , maxVel , num_chan)
    chan_arr = chan_arr.reshape(1, num_chan)

    channel_parameters = [num_chan, chan_arr]

    return channel_parameters
