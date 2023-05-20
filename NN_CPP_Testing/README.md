# C++ Implementtion of the inverse model

This folder contains the definitions and a testing file for the Inverse model implemented in C++.
We need to implement the model in C++ as the autonomous agent sumulation is in C++.

This folder contains the following relevant directories:

<pre> 
├── CMakeLists.txt
├── README.md
├── cmake-build-debug
├── csv_functions.h
├── neural_network.h
├── neural_network_test.cc
├── test_data
│   ├── acceleration.csv
│   └── velocity.csv
└── weights_biases.h
</pre>

## csv_functions.h
Contains the functions necessary to parse the csv files needed in the testing.

## weights_biases.h
Header containing the weights and biases of the network recovered from the .h5 Keras mode.

## neural_network.h
This header file contains the functions used in the Inverse model. As well as the structure of the final inverse model callable as a function.

## neural_network_test.cc
This file tests the model the model and outputs the calculated pedal as a CSV which can then be used in the python testing file to compare the two outputs of the models to verify accuracy. 







