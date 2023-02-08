"""
File for the direct NN model.

It takes as inputs:
 - pedal "p" with values [-1; 1];
 - current vehicle velocity "v".    ---- CHANNEL ID ?????

It outputs:
 - current vehicle acceleration "a".

Author: Sebastiano Taddei.
Date:   29/11/2022.
"""

###########
# Imports #
###########

import numpy as np
import tensorflow as tf
from src.functions import initializers as ci  # i.e., custom initalizers
from src.functions.layers import ActivLin1D  # import the custom activation function

# Disable eager execution so as not to get compilation errors using symbolic tensors
from tensorflow.python.framework.ops import disable_eager_execution

disable_eager_execution()


#############
# Functions #
#############

def simple_model(batch_size, input_shape, output_shape, weights_init, l2_reg, opt_params):
    """
    Simple NN model for the direct vehicle dynamics.

    It implements the following scheme:

    velocity -> ^2 -> *drag_coeff ->
                                     + -> acceleration
    pedal    -> Fully connected   ->

    Arguments:
     - batch_size:   size of the batch [int];
     - input_shape:  shape of the inputs;
     - output_shape: shape of the output;
     - weights_init: list of weight initializers;
     - l2_reg:       L2 regularization;
     - opt_params:   list of parameters for the optimizer [learning_rate, epsilon].

    Outputs:
     - model: Keras NN model of the aforementioned scheme.
    """

    # -- INPUTS --
    # Input() is used to instantiate a Keras tensor. A Keras tensor is a symbolic tensor - like object, which we augment
    # with certain attributes that allow us to build a Keras model just by knowing the inputs and outputs of the model.
    input_vx = tf.keras.layers.Input(shape=input_shape,
                                     batch_size=batch_size,
                                     name='input_vx')
    input_p = tf.keras.layers.Input(shape=input_shape,
                                    batch_size=batch_size,
                                    name='input_p')

    # -- INITIALIZERS --
    # Initializers define the way to set the initial random weights of Keras layers.
    drag_init = ci.MyInitializer(weights_init[0])
    fc_init = ci.MyInitializer(weights_init[1])

    # -- VELOCITY --
    # Square the input velocity :
    input_vx_squared = tf.keras.layers.Multiply()(
        [input_vx[:, -2:-1], input_vx[:, -2:-1]])  # take last element of the velocity
    # Multiply by the weights:
    vel_weighted = tf.keras.layers.Dense(units=1, kernel_initializer=drag_init)(input_vx_squared)
    # Negative ReLU:
    ReLUed = -tf.keras.layers.ReLU()(vel_weighted)

    # -- PEDAL --
    # Add apply first neural layer:
    neural_layer_1 = tf.keras.layers.Dense(1, kernel_initializer=fc_init)(input_p)
    # Apply second Neural layer
    # neural_layer_2 = tf.keras.layers.Dense(1)(neural_layer_1)

    # -- SUM ACCELERATION AND DRAG --
    # summed = tf.math.add(neural_layer_1 , ReLUed)
    summed = tf.keras.layers.Add()([neural_layer_1, ReLUed])

    # -- OUTPUT OF MODEL --
    output = summed

    # Define the model
    model = tf.keras.Model(inputs=[input_vx, input_p],
                           outputs=output,
                           name='simple_model')

    # Optimizer parameters
    optimizer = tf.keras.optimizers.Adam(learning_rate=opt_params[0],
                                         epsilon=opt_params[1])

    # Compile the model
    model.compile(optimizer=optimizer,
                  loss='mean_squared_error',
                  metrics=[tf.keras.metrics.RootMeanSquaredError()]
                  )

    return model
