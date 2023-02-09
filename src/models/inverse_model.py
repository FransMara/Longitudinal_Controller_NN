
import numpy as np
import tensorflow as tf
from src.functions import initializers as ci  # i.e., custom initalizers
from src.functions.layers import ActivLin1D  # import the custom activation function
from src.functions.activation_parameters import activation_function_parameters as afp

# Disable eager execution so as not to get compilation errors using symbolic tensors
from tensorflow.python.framework.ops import disable_eager_execution
disable_eager_execution()


def inverse_model(batch_size, input_shape, output_shape, weights_init, l2_reg, channel_parameters, opt_params):
    """
    INPUTS:
     - desired acceleration "a" from the motion primitives;
     - desired velocity "v" from the motion primitives.

    OUTPUTS:
     - current pedal position "p" with values [-1; 1].
    """

    # Declare the inputs:
    input_vx = tf.keras.layers.Input(shape=input_shape,
                                     batch_size=batch_size,
                                     name='input_vx')
    input_acc = tf.keras.layers.Input(shape=input_shape,
                                      batch_size=batch_size,
                                      name='input_acc')
    # Declare the activated neural layer that will be used in the for loop
    pedal_layer = [0.]*channel_parameters[0]

    # -- VELOCITY --
    # Square the input velocity :
    input_vx_squared = tf.keras.layers.Multiply(name='velocity_squared')([input_vx[:, -2:-1], input_vx[:, -2:-1]])
    # Multiply by the weights:
    vel_weighted = tf.keras.layers.Dense(units=1,
                                         kernel_initializer='random_normal',
                                         use_bias=False,
                                         name='drag_coeff_layer')(input_vx_squared)
    # Positive ReLU:
    drag = tf.keras.layers.ReLU(name='negative_ReLU')(vel_weighted)

    # -- ACCELERATION --
    # Acceleration activation:
    # Loop through all the channels and add the activation that falls under the specific channel
    for i in range(channel_parameters[0]):
        custom_act_acc = ActivLin1D(chan_id=i,
                                    chan_arr=channel_parameters[1][0],
                                    name='activated_acceleration_' + str(i))([input_vx, input_acc])
        pedal_layer[i] = pedal_layer[i] + tf.keras.layers.Dense(units=1,
                                                                kernel_initializer='random_normal',
                                                                name='pedal_layer_' + str(i))(custom_act_acc)
    pedal = tf.keras.layers.Add(name='neural_layer_out')(pedal_layer)

    # -- SUM ACCELERATION AND DRAG --
    summed = tf.keras.layers.Add(name='pedal_drag_final_sum')([pedal, drag])

    # -- OUTPUT OF MODEL --
    output = summed

    # Define the model
    model = tf.keras.Model(inputs=[input_vx, input_acc],
                           outputs=output,
                           name='inverse_model')

    # Optimizer parameters
    optimizer = tf.keras.optimizers.Adam(learning_rate=opt_params[0],
                                         epsilon=opt_params[1])

    # Compile the model
    model.compile(optimizer=optimizer,
                  loss='mean_squared_error')

    return model
