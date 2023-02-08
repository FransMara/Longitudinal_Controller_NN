'''
Main file for the longitudinal controller of the course "Intelligent Vehicles"
of prof. Rosati Papini.

This file is responsible for the training of both the direct and inverse NN
models.

Author: Sebastiano Taddei.
Date:   29/11/2022.
'''

###########
# Imports #
###########

import tensorflow as tf
import numpy as np
import pandas as pd
import csv
import matplotlib.pyplot as plt
from src.functions import initializers as ci # i.e., custom initializers
from src.models import simple_direct_model as dm
from src.utils import data_handler as dh
from src.utils import plotter as pl

from train_forward_model import forward_training
from train_inverse_model import inverse_training
from src.functions.layers import ActivLin1D
from src.functions.activation_parameters import activation_function_parameters as act_fun_params
from src.models import forward_model as am

#############
# Functions #
#############

def main():
    '''
    Main function that starts the training process.

    It handles the data loading, the model creation, the training, and the plotting.
    '''

    # Load the forward model training dataset:
    direct_dataset_path = '/Users/francescomaraschin/Desktop/IntelligentVehicles/Project_NN_Conda/data/csv/forward_profiles.csv'
    direct_dataset = dh.load_csv(direct_dataset_path)

    # Plot the dataset
    plt.plot(direct_dataset)
    plt.title('Normalised Forward Model Training Data')
    plt.legend(['Pedal', 'Velocity', 'Acceleration'])
    plt.show()

    # Train the model:
    [forward_model,
     forward_history,
     forward_train_data,
     forward_valid_data,
     time,
     batch_size] = forward_training(direct_dataset = direct_dataset)

    # Plot the forward model results
    pl.results(model=forward_model,
               history=forward_history,
               dataset=[forward_train_data,
                        forward_valid_data],
               time=time,
               batch_size=batch_size)

    # Calculate the inputs for the inverse model
    calculated_accelerations_1 = forward_model.predict(forward_train_data[0])
    calculated_accelerations_2 = forward_model.predict(forward_valid_data[0])
    calculated_accelerations = np.concatenate([calculated_accelerations_1, calculated_accelerations_2])

    # Create the input dataset for the inverse model:
    time = pd.read_csv(direct_dataset_path, usecols=['time'])
    velocity = pd.read_csv(direct_dataset_path, usecols=['velocity'])
    acceleration = pd.DataFrame(calculated_accelerations, columns=['acceleration'])
    pedal = pd.DataFrame(calculated_accelerations, columns=['pedal'])

    inverse_dataset = pd.concat([time, velocity, acceleration, pedal], axis=1)
    inverse_dataset = inverse_dataset.set_index('time')


    # Train the inverse model:
    [inverse_model,
     inv_history,
     inv_train_data,
     inv_valid_data,
     time,
     batch_size] = inverse_training(inverse_dataset=inverse_dataset)

    # Plot the results of the inverse training:
    pl.results(model=inverse_model,
               history=inv_history,
               dataset=[inv_train_data,
                        inv_valid_data],
               time=time,
               batch_size=batch_size)

############
# Run file #
############

if __name__ == '__main__':
    main()
