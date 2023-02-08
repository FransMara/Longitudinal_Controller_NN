'''
File to handle data for the NN model.

Author: Sebastiano Taddei.
Date:   29/11/2022.
'''

###########
# Imports #
###########

import pandas as pd
import numpy as np

#############
# Functions #
#############

def load_csv(filepath):
    '''
    Function to load a CSV dataset.

    Arguments:
     - filepath: path to CSV file;

    Outputs:
     - dataset: pandas dataframe containing the data.
    '''

    dataset = pd.read_csv(filepath_or_buffer=filepath,
                          index_col=0,
                          dtype=np.float32) # to make TF happy

    return dataset

def window_data(dataset, input_labels, output_labels, input_window, output_window, batch_size,
                validation_split=0.3):
    """
    Window data and split it into input/output.

    Arguments:
     - dataset:          pandas dataframe containing both input/output and trainig/validation data;
     - input_labels:     list of labels corresponding to the input columns of the dataset;
     - output_labels:    list of labels corresponding to the input columns of the dataset;
     - input_window:     size of the input window;
     - output_window:    size of the output window;
     - batch_size:       size of the batch;
     - validation_split: percentage of data to reserve for the validation set (taken from the last
                         samples).

    Outputs:
     - train_data: [input training data, target training data];
     - valid_data: [input validation data, target validation data];
     - time:       time array of the dataset.

    Be careful to have the correct input/output relation with your data:
     - [p_k, p_k+1, p_k+2] -> [acc_k] for a future input window;
     - [p_k-2, p_k-1, p_k] -> [acc_k] for a past input window.
    Change the data handler accordingly.
    """

    # Split data into input and output
    pd_input  = dataset[input_labels]
    pd_output = dataset[output_labels]

    # If batch size is provided trim data to be a multiple of it
    if batch_size:
        last_idx = np.floor((len(dataset.index) - input_window)/batch_size).astype(int)*batch_size
    else:
        last_idx = len(dataset.index)

    # Reshape data
    input_data = [None] * len(input_labels)
    for i, (_, values) in enumerate(pd_input.items()):
        np_input      = values.to_numpy().flatten()
        input_data[i] = np.array([np_input[i:i + input_window]
                                  for i in range(last_idx)])

    output_data = [None] * len(output_labels)
    for i, (_, values) in enumerate(pd_output.items()):
        np_output      = values.to_numpy().flatten()
        output_data[i] = np.array([np_output[i:i + output_window]
                                  for i in range(last_idx)])

    # Split data into training and validation
    if batch_size:
        train_idx = np.floor(last_idx*(1 - validation_split)/batch_size).astype(int)*batch_size
    else:
        train_idx = np.floor(last_idx*(1 - validation_split)).astype(int)

    train_data = [[data[:train_idx] for data in input_data],
                  [data[:train_idx] for data in output_data]]
    valid_data = [[data[train_idx:] for data in input_data],
                  [data[train_idx:] for data in output_data]]

    time = [dataset.index[:train_idx],
            dataset.index[train_idx:last_idx],]

    return train_data, valid_data, time



