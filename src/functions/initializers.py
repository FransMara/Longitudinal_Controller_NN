'''
File for the custom initializers of the NN model.

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

# Initializers define the way to set the initial random weights of Keras layers.

class MyInitializer(tf.keras.initializers.Initializer):
    """
    Class to create custom initializers for Keras.

    Parameters:
     - init_value: initial value for the initializer with dtype=float32.

    Usage: custom_init = MyInitializer(init_value).
    """

    def __init__(self, init_value, **kwargs):
        self.init_value = init_value

        super().__init__(**kwargs)

    def __call__(self, shape, *args, **kwargs):
        return tf.convert_to_tensor(self.init_value)

    def get_config(self):  # to support serialization
        config = super().get_config()

        config.update({"init_value": self.init_value})

        return config
