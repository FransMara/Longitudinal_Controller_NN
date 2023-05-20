# Forward and Inverse Model Training

From this folder you can train the forward and inverse networks for the autonomous driving agent. 

This folder contains the following relevant directories:

<pre> 
├── Get_weights_biases_&_chan_arr.ipynb
├── README.md
├── compare_py_and_cpp_models.ipynb
├── longitudianl_controller_MAIN.ipynb
├── requirements.txt
├── src
│   ├── functions
│   │   ├── activation_parameters.py
│   │   ├── initializers.py
│   │   └── layers.py
│   ├── models
│   │   ├── depricated_forward_model_original.py
│   │   ├── forward_model.py
│   │   └── inverse_model.py
│   └── utils
│       ├── data_handler.py
│       └── plotter.py
├── test
│   ├── act_fcn.csv
│   ├── activation_function_test.ipynb
│   └── test_activ_lin_1D.py
├── train_forward_model.py
└── train_inverse_model.py
</pre>

## longitudianl_controller_MAIN

This is the main file that trains first the forward and then the inverse model (using a jupyter notebook (sorry Sebastiano)). This file, on top of loading the CSV's also calles the following two functions:

>### train_forward_model.py & train_inverse_model.py
>> These files define the hyperparameters of the training and in turn also call the definitions of the models found under src/models. 

## Get_weights_biases_&_chan_arr.ipynb
This file exports the weigths and biases of the inverse model to CSV so they can then be implemented in the C++ model of the controller. 

## compare_py_and_cpp_models.ipynb
 This file allows the comparisson between the results of the python performance and the C++ model performance by comparing the plots of the outputs of the models in the two different languages. 






