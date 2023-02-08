'''
File for the custom layers of the NN model.

Author: Sebastiano Taddei.
Date:   29/11/2022.
'''

###########
# Imports #
###########

import tensorflow as tf

###########
# Classes #
###########

class ActivLin1D(tf.keras.layers.Layer):
    '''
    Linear 1D activation function.

    This function activates at most 2 channels at a time and makes sure that the sum of each
    channel is equal to 1.

    Parameters:
     - chan_id:  index of the channel center (i.e., of the chan_arr element);
     - chan_arr: array containing the channel centers.

    Inputs:
     - inputs: [act_in, f_out] where act_in is the input that activates the channels and f_out is
               the output of the function to be activated.

    Outputs:
     - output: the product between the activation function and the output of the function to be
               activated.

    Usage: act_fun = ActivLin1D(chan_id, chan_arr)([act_in, f_out]).
    '''

    def __init__(self, chan_id, chan_arr, **kwargs):
        self.chan_id  = chan_id
        self.chan_arr = chan_arr

        super().__init__(**kwargs)

    def build(self, input_shape):
        self.built = True

    def call(self, inputs, *args, **kwargs):
        act_in, f_out = inputs

        # Compute the number of channels
        chan_num = len(self.chan_arr)

        # First dimension of activation
        if self.chan_id == 0:
            if chan_num != 1:
                ampl    = self.chan_arr[1] - self.chan_arr[0]
                act_fcn = tf.math.minimum(
                    tf.math.maximum(-(act_in - self.chan_arr[0])/ampl + 1, 0), 1)

            else:
                act_fcn = 1 # in case the user only wants one channel

        elif self.chan_id != 0 and self.chan_id == (chan_num - 1):
            ampl    = self.chan_arr[-1] - self.chan_arr[-2]
            act_fcn = tf.math.minimum(
                tf.math.maximum((act_in - self.chan_arr[-2])/ampl, 0), 1)

        else:
            ampl_1  = self.chan_arr[self.chan_id] - self.chan_arr[self.chan_id - 1]
            ampl_2  = self.chan_arr[self.chan_id + 1] - self.chan_arr[self.chan_id]
            act_fcn = tf.math.minimum(
                tf.math.maximum((act_in - self.chan_arr[self.chan_id - 1])/ampl_1, 0),
                tf.math.maximum(-(act_in - self.chan_arr[self.chan_id])/ampl_2 + 1, 0))

        output = f_out * act_fcn

        return output

    def get_config(self):
        config = super().get_config()

        config.update({
            "chan_id":  self.chan_id,
            "chan_arr": self.chan_arr,
        })

        return config