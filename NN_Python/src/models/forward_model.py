import numpy as np
import tensorflow as tf
from src.functions import initializers as ci  # i.e., custom initalizers
from src.functions.layers import ActivLin1D  # import the custom activation function


def activated_model(batch_size, input_shape, output_shape, weights_init, l2_reg, channel_parameters, opt_params):
    """
    Arguments:
     - batch_size:   size of the batch [int];
     - input_shape:  shape of the inputs;
     - output_shape: shape of the output;
     - weights_init: list of weight initializers;
     - l2_reg:       L2 regularization;
     - opt_params:   list of parameters for the optimizer [learning_rate, epsilon].
     - channel_parameters:   [num_chan , chan_array]
     - nn_layers: list of arrays that will contain number of NN large number of activation triangles we have and the initialization of the sum

    Outputs:
     - model: Keras NN model of the aforementioned scheme.
    """

    # -- INPUTS --
    # Input() is used to instantiate a Keras tensor. A Keras tensor is a symbolic tensor - like object, which we augment
    # with certain attributes that allow us to build a Keras model just by knowing the inputs and outputs of the model.
    input_vx = tf.keras.layers.Input(shape      = input_shape,
                                     batch_size = batch_size,
                                     name       = 'input_vx')
    input_p = tf.keras.layers.Input(shape       = input_shape,
                                    batch_size  = batch_size,
                                    name        = 'input_p')
    
    # Declare list of neural networks number large as size of activation functions
    neural_layer = [0.] * channel_parameters[0]

    # -- VELOCITY --
    # Square the input velocity :
    input_vx_squared = tf.keras.layers.Multiply(name='velocity_squared')([input_vx[:, -2:-1], input_vx[:, -2:-1]])

    # Multiply by the weights:
    vel_weighted     = tf.keras.layers.Dense(units = 14,
                                             name  = 'weighted_vel_network')(input_vx_squared)
    # Negative ReLU:
    drag = -tf.keras.layers.ReLU(name='negative_ReLU')(vel_weighted)

    # -- ACCELERATION --
    # Custom activation function:
    for i in range(channel_parameters[0]):
        # Activation to split the accelerations:
        custom_act_acc  = ActivLin1D(           chan_id            = i,
                                                chan_arr           = channel_parameters[1][0],
                                                name               = 'activated_pedal_' + str(i))([input_vx, input_p])
        # Neural layers:
        neural_layer[i] = neural_layer[i] + tf.keras.layers.Dense(
                                                units              = 64,
                                                kernel_initializer = 'he_normal',
                                                activation         = tf.keras.layers.LeakyReLU(),
                                                name               = 'activated_neural_layer_' + str(i))(custom_act_acc)
    
    # Add neural networks:
    pre_pre_acceleration  = tf.keras.layers.Add(name = 'add_outputs_of_neural_layers')(neural_layer)

    # Acceleration Neural Network:
    acceleration          = tf.keras.layers.Dense(units              = 14,                                         
                                                  kernel_initializer = 'glorot_normal',
                                                  activation         = 'tanh',
                                                  name               = 'acceleration')(pre_pre_acceleration)

    # -- SUM ACCELERATION AND DRAG --
    acc_drag_sum = tf.keras.layers.Add(name='acceleration_drag_final_sum')([acceleration, drag])

    # -- NEURAL LAYER BEFORE OUTPUT --
    ensamble          = tf.keras.layers.Dense(units                = 10,                                       
                                                kernel_initializer = 'glorot_normal',
                                                activation         = 'tanh',
                                                name               = 'acc_drag_layer')(acc_drag_sum)

    # -- OUTPUT --
    output              = tf.keras.layers.Dense(units              = 1,                                       
                                                kernel_initializer = 'glorot_normal',
                                                activation         = 'tanh',
                                                name               = 'output_of_model')(ensamble)

    # Define the model
    model = tf.keras.Model(inputs  = [input_vx, input_p],
                           outputs = output,
                           name    = 'custom_activation_model')

    # Optimizer parameters --> NAdam works better than Adam it appears
    optimizer = tf.keras.optimizers.Nadam(learning_rate = opt_params[0],
                                          epsilon       = opt_params[1])

    # Compile the model
    model.compile(optimizer = optimizer,
                  loss      = 'mean_squared_error')

    return model
