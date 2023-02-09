'''
File to the the ActivLin1D class.

Author: Sebastiano Taddei.
Date:   29/11/2022.
'''

###########
# Imports #
###########

import numpy as np
import matplotlib.pyplot as plt

#############
# Funcitons #

def activ_lin_1d(act_in, chan_id, chan_arr):
    '''
    Copy of the ActivLin1D class as Python function using numpy covering only the actual channels.

    Arguments:
     - act_in:   the input that activates the channels;
     - chan_id:  index of the channel center (i.e., of the chan_arr element);
     - chan_arr: array containing the channel centers.

    Outputs:
     - act_fcn: the activation function.
    '''

    # Compute the number of channels
    chan_num = len(chan_arr)

    # First dimension of activation
    if chan_id == 0:
        if chan_num != 1:
            ampl    = chan_arr[1] - chan_arr[0]
            act_fcn = np.minimum(
                np.maximum(-(act_in - chan_arr[0])/ampl + 1, 0), 1)

        else:
            act_fcn = 1 # in case the user only wants one channel

    elif chan_id != 0 and chan_id == (chan_num - 1):
        ampl    = chan_arr[-1] - chan_arr[-2]
        act_fcn = np.minimum(
            np.maximum((act_in - chan_arr[-2])/ampl, 0), 1)

    else:
        ampl_1  = chan_arr[chan_id] - chan_arr[chan_id - 1]
        ampl_2  = chan_arr[chan_id + 1] - chan_arr[chan_id]
        act_fcn = np.minimum(
            np.maximum((act_in - chan_arr[chan_id - 1])/ampl_1, 0),
            np.maximum(-(act_in - chan_arr[chan_id])/ampl_2 + 1, 0))

    return act_fcn

def main():
    ''' Run the tests. '''

    # Test data
    act_in   = np.linspace(-1, 6, 1000)
    num_chan = 7

    chan_arr = np.linspace(np.min(act_in) + 1, np.max(act_in) - 1, num_chan)
    act_fcn  = np.array([activ_lin_1d(act_in=act_in,
                                      chan_id=i,
                                      chan_arr=chan_arr)
                         for i in range(num_chan)]).T

    # Plot the result
    plt.figure()
    plt.plot(act_in, act_fcn)
    plt.grid(True)
    plt.show()

    # Check that the sum of all activation functions is equal to 1
    check_sum =  np.mean(np.sum(act_fcn, axis=1))
    assert check_sum == 1.0,\
           f'The sum of all activation functions is not 1, it is: {check_sum:.4e}'
    print(f'The sum of all activation functions is: {check_sum:.4e}')

if __name__ == '__main__':
    main()
