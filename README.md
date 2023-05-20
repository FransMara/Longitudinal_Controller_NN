# Neural Network for the Longitudinal Control of an Autonomous Driving Agent

    Authors: Francesco Maraschin ; Ashkan Behmanesh 
    Date: May 2023

This project attemps to design and implement an neural network for the control of an autonomous driving agent. 

We train the so called **forward** and **inverse** models. The forward model is trained on the pedal and velocity data of an agent and is tasked with outputting the learnt actual acceleration of the vehicle. 

After this, through transfer learning we use the **forward model** to train the **inverse model** which is traind on the original velocity data and the forward model acceleration data. 

The idea of training through transfer learning is that by splitting the training between two NN's it allows for better feature extraction enabling the Inverse model to start from a better position compared to training it from scratch. 

>NOTE: All directories contain own README files explaining the structure and working of the different operations.

This folder contains the following relevant directories:

<pre> 
├── Data
├── Long_Controller
├── NN_CPP_Testing
└── NN_Python
</pre>

## Data

This directory contains the training data, MatLab data processing files and the final forward and inverse models. 

## Long_Controller
This directory contains the autonomous agent simulation implemented with the final inverse model written in C++.

## NN_CPP_Testing
This directory contains the files relevent to writing and testing/comparing the C++ mode with the original python inverse model. 

## NN_Python
 This directory contins the relevent code to train both the forward and inverse models as well as to analise the performance and save the final Keras .h5 models. 






