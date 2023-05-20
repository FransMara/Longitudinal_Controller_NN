import tensorflow as tf
import numpy as np
from src.functions import initializers as ci # i.e., custom initializers
from src.utils import data_handler as dh
from src.functions.layers import ActivLin1D
from src.functions.activation_parameters import activation_function_parameters as act_fun_params
from src.models import forward_model as fm

# Model Training:
def forward_training(forward_dataset):

    # NN parameters
    nn_name = 'forward_model'
    batch_size    = 100
    epochs        = 200
    valid_split   = 0.1
    input_window  = 10
    output_window = 1
    l2_reg        = 8e-8
    opt_params    = [0.001, # learning rate
                     1e-8]  # epsilon

    # Weights initializers
    drag_init = np.ones(shape=(1, 1),
                        dtype=np.float32)  # to make TF happy
    fc_init = np.ones(shape=(input_window, output_window),
                      dtype=np.float32)

    weights_init = [drag_init,
                    fc_init]


    # Reshape and split data
    train_data, valid_data, time = dh.window_data(dataset=forward_dataset,
                                                  input_labels=['pedal',
                                                                'velocity'],
                                                  output_labels=['acceleration'],
                                                  input_window=input_window,
                                                  output_window=output_window,
                                                  batch_size=batch_size,
                                                  validation_split=valid_split)

    # Create the channel parameters:
    channel_params = act_fun_params(forward_dataset)
    # Create model
    act_model = fm.activated_model(batch_size=batch_size,
                                   input_shape=input_window,
                                   output_shape=output_window,
                                   weights_init=weights_init,
                                   l2_reg=l2_reg,
                                   channel_parameters=channel_params,
                                   opt_params=opt_params)

    # Print model summary
    act_model.summary()

    # Callbacks
    save_path = f'/Users/francescomaraschin/Library/Mobile Documents/com~apple~CloudDocs/Universtiy/IntelligentVehicles/LongitudinalControllerNN/Data/trained_models/{nn_name}/{nn_name}.h5'
    checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath=save_path,
                                                    monitor='loss',
                                                    mode='auto',
                                                    verbose=1,
                                                    save_best_only=True)
    early_stop = tf.keras.callbacks.EarlyStopping(monitor='loss',
                                                  patience=50,
                                                  verbose=1)

    # Fix the random seed for reproducibility
    tf.keras.utils.set_random_seed(314159)
    
    # Fit model
    history = act_model.fit(x=train_data[0],
                            y=train_data[1],
                            batch_size=batch_size,
                            epochs=epochs,
                            verbose=1,
                            callbacks=[checkpoint,
                                       early_stop],
                            validation_data=valid_data,
                            shuffle=False,  # Shuffle shuffels only the training data, dont use if time dependant...
                            use_multiprocessing=True)

    # Load the best model
    best_model = tf.keras.models.load_model(filepath=save_path,
                                            custom_objects={
                                                'MyInitializer': ci.MyInitializer,
                                                'ActivLin1D': ActivLin1D
                                            })

    return [best_model, history, train_data, valid_data, time, batch_size]