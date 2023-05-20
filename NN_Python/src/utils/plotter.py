'''
File to handle plotting for the NN model.

Author: Sebastiano Taddei.
Date:   29/11/2022.
'''

###########
# Imports #
###########

import matplotlib.pyplot as plt
import numpy as np

#############
# Functions #
#############

def results(model, history, dataset, time, batch_size, title, location,mode):
    '''
    Function to plot the results of the NN model.

    Arguments:
     - model:      model to predict with;
     - history:    history of the fitted model;
     - dataset:    windowed dataset;
     - time:       time array for of the data;
     - batch_size: size of the batch.
    '''

    train_pred = model.predict(x=dataset[0][0],
                               batch_size=batch_size,
                               verbose=0,
                               use_multiprocessing=True).flatten()

    valid_pred = model.predict(x=dataset[1][0],
                               batch_size=batch_size,
                               verbose=0,
                               use_multiprocessing=True).flatten()

    fig, axes = plt.subplots(1, 3,figsize=(15, 8))

    fig.suptitle(title)
    axes[0].grid(True)
    axes[0].plot(history.history['loss'])
    axes[0].plot(history.history['val_loss'])
    axes[0].set(xlabel="Epoch", ylabel="Loss")
    axes[0].legend(['Train', 'Validation'], loc='best')
    axes[0].set_yscale('log')
    axes[0].set_title('Loss')

    axes[1].grid(True)
    axes[1].plot(time[0], train_pred)
    axes[1].plot(time[0], dataset[0][1][0])
    axes[1].set(xlabel="time", ylabel=mode)
    axes[1].legend(['Train', 'Ground Truth'], loc='best')
    axes[1].set_title('Training prediction')

    axes[2].grid(True)
    axes[2].plot(time[1], valid_pred)
    axes[2].plot(time[1], dataset[1][1][0])
    axes[2].set(xlabel="time", ylabel=mode)
    axes[2].legend(['Train', 'Ground Truth'], loc='best')
    axes[2].set_title('Validation prediction')

    fig.tight_layout()

    plt.savefig(location)
    plt.show()