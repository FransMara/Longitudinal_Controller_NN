//
// Created by Frans on 04/02/23.
//

#include "weights_biases .h"
#include "csv_functions.h"
#include <iostream>
#include <cstdio>
#include <vector>
using namespace std;

#ifndef NN_CPP_TEST_NEURAL_NETWORK_H
#define NN_CPP_TEST_NEURAL_NETWORK_H

//███████╗ ██████╗    ██╗      █████╗ ██╗   ██╗███████╗██████╗
//██╔════╝██╔════╝    ██║     ██╔══██╗╚██╗ ██╔╝██╔════╝██╔══██╗
//█████╗  ██║         ██║     ███████║ ╚████╔╝ █████╗  ██████╔╝
//██╔══╝  ██║         ██║     ██╔══██║  ╚██╔╝  ██╔══╝  ██╔══██╗
//██║     ╚██████╗    ███████╗██║  ██║   ██║   ███████╗██║  ██║
//╚═╝      ╚═════╝    ╚══════╝╚═╝  ╚═╝   ╚═╝   ╚══════╝╚═╝  ╚═╝

// This function calculates the perceptron function with a number of inputs specified.
// It does not calculate the activation.
// IN: - data_in = the data in input
//     -       w = the vector of the weights of the perceptron
//     -       b = the bias of the perceptron
// OUT: - data_out = the output of the perceptron
template <int Nin>
void linear_layer( const double data_in[Nin][num_channels] , const double w[num_channels][Nin] ,
                     const double b[num_channels] , double data_out[num_channels]){
    // For every channel we have a different NN:
    for (int chan = 0; chan < num_channels; ++chan) {
        data_out[chan] = 0;
        // For every input into that specific NN:
        for (int i = 0; i < Nin; ++i) {
            data_out[chan] += data_in[i][chan] * w[chan][i];
        }
        data_out[chan] += b[chan];
    }
}

template <int Nin>
double linear_layer_deprecated( const double data_in[Nin] , const double w[Nin] , const double b ){
    double data_out = 0.0;
    for(int i=0 ; i<Nin ; i++){
        data_out += data_in[i]*w[i];
    }
    data_out += b;
    return data_out;
}

//██╗   ██╗███████╗██╗      ██████╗  ██████╗██╗████████╗██╗   ██╗    ██╗      █████╗ ██╗   ██╗███████╗██████╗
//██║   ██║██╔════╝██║     ██╔═══██╗██╔════╝██║╚══██╔══╝╚██╗ ██╔╝    ██║     ██╔══██╗╚██╗ ██╔╝██╔════╝██╔══██╗
//██║   ██║█████╗  ██║     ██║   ██║██║     ██║   ██║    ╚████╔╝     ██║     ███████║ ╚████╔╝ █████╗  ██████╔╝
//╚██╗ ██╔╝██╔══╝  ██║     ██║   ██║██║     ██║   ██║     ╚██╔╝      ██║     ██╔══██║  ╚██╔╝  ██╔══╝  ██╔══██╗
// ╚████╔╝ ███████╗███████╗╚██████╔╝╚██████╗██║   ██║      ██║       ███████╗██║  ██║   ██║   ███████╗██║  ██║
//  ╚═══╝  ╚══════╝╚══════╝ ╚═════╝  ╚═════╝╚═╝   ╚═╝      ╚═╝       ╚══════╝╚═╝  ╚═╝   ╚═╝   ╚══════╝╚═╝  ╚═╝

// This function calculates the square of the current velocity and multiplies this by the trained weight to
// get the drag acting on the vehicle.
// This function takes the current velocity as input as well as the single trained weight.
// It calculates and returns the drag.
double velocity_layer( const double data_in , const double w){
        double data_out = 0.0;
        data_out += data_in * data_in;
        data_out += w * data_out;
        return data_out;
}

//███╗   ██╗███████╗ ██████╗     ██████╗ ███████╗██╗     ██╗   ██╗
//████╗  ██║██╔════╝██╔════╝     ██╔══██╗██╔════╝██║     ██║   ██║
//██╔██╗ ██║█████╗  ██║  ███╗    ██████╔╝█████╗  ██║     ██║   ██║
//██║╚██╗██║██╔══╝  ██║   ██║    ██╔══██╗██╔══╝  ██║     ██║   ██║
//██║ ╚████║███████╗╚██████╔╝    ██║  ██║███████╗███████╗╚██████╔╝
//╚═╝  ╚═══╝╚══════╝ ╚═════╝     ╚═╝  ╚═╝╚══════╝╚══════╝ ╚═════╝

// This function implements the negative of the ReLU function, we want the drag to have a negative influence on the
// acceleration of the vehicle, so we send the calculated drag through this layer.
template<int Nin>
void neg_ReLU( const double data_in[Nin] , double data_out[Nin]){
    for(int i = 0; i < Nin ; i++ ){
        if (data_in[i] < 0)
            data_out[i] = 0;
        else if (data_in[i] >= 0)
            data_out[i] = -1.0 * data_in[i];
    }
}

// █████╗ ██████╗ ██████╗ ██╗████████╗██╗ ██████╗ ███╗   ██╗
//██╔══██╗██╔══██╗██╔══██╗██║╚══██╔══╝██║██╔═══██╗████╗  ██║
//███████║██║  ██║██║  ██║██║   ██║   ██║██║   ██║██╔██╗ ██║
//██╔══██║██║  ██║██║  ██║██║   ██║   ██║██║   ██║██║╚██╗██║
//██║  ██║██████╔╝██████╔╝██║   ██║   ██║╚██████╔╝██║ ╚████║
//╚═╝  ╚═╝╚═════╝ ╚═════╝ ╚═╝   ╚═╝   ╚═╝ ╚═════╝ ╚═╝  ╚═══╝

template <int Nin , int Nout>
void addition( const double data_in[Nin] , double data_out[Nout]){
    for(int j=0 ; j<Nout ; j++){
        data_out[j] = 0;
        for(int i=0 ; i<Nin ; i++){
            data_out[j] += data_in[i];
        }
    }
}

// █████╗  ██████╗████████╗██╗██╗   ██╗ █████╗ ████████╗██╗ ██████╗ ███╗   ██╗
//██╔══██╗██╔════╝╚══██╔══╝██║██║   ██║██╔══██╗╚══██╔══╝██║██╔═══██╗████╗  ██║
//███████║██║        ██║   ██║██║   ██║███████║   ██║   ██║██║   ██║██╔██╗ ██║
//██╔══██║██║        ██║   ██║╚██╗ ██╔╝██╔══██║   ██║   ██║██║   ██║██║╚██╗██║
//██║  ██║╚██████╗   ██║   ██║ ╚████╔╝ ██║  ██║   ██║   ██║╚██████╔╝██║ ╚████║
//╚═╝  ╚═╝ ╚═════╝   ╚═╝   ╚═╝  ╚═══╝  ╚═╝  ╚═╝   ╚═╝   ╚═╝ ╚═════╝ ╚═╝  ╚═══╝

template<int Nin>
void custom_activation(const double vel[Nin] , const double acc[Nin] , double act_acc[num_channels][Nin]){
    double ampl , ampl_1 , ampl_2;          // the width of a channels
    double act_fcn[Nin][num_channels];      // activation function the will me multiplied by accelerations

    // For every single velocity of the velocity array in input
    for (int i = 0; i < Nin ; ++i) {
        // For every channel defined (10)
        for (int chan = 0; chan < num_channels ; ++chan) {
            // If first channel
            if (chan == 0){
                // if there are multiple channels
                if (num_channels != 1) {
                    ampl = chan_array[1] - chan_array[0];
                    act_fcn[i][chan] = min(max(-(vel[i] - chan_array[0]) / ampl + 1, 0.0), 1.0);
                }
                // if there is only one channel then the activation function is only 1
                else
                        act_fcn[chan][i] = 1.;
            }
            // If last channel
            else if (chan == num_channels-1){
                ampl    = chan_array[num_channels-1] - chan_array[num_channels-2];
                act_fcn[i][chan] = min( max( ( vel[i] - chan_array[num_channels -2] )/ampl , 0.0) , 1.0);
            }
            // If any other channel
            else {
                ampl_1 = chan_array[chan] - chan_array[chan - 1];
                ampl_2 = chan_array[chan + 1] - chan_array[chan];
                act_fcn[i][chan] = min(max((vel[i] - chan_array[chan - 1]) / ampl_1, 0.0),
                                    max(-(vel[i] - chan_array[chan]) / ampl_2 + 1, 0.0)
                );
            }
            // Activate the acceleration with
            act_acc[i][chan] = act_fcn[i][chan] * acc[i];

            //cout << " " << act_fcn[i][chan];
        }
        //cout << endl;
        //appendRowToCSV(act_fcn[i] , "act_fcn.csv" , num_channels);
    }
}

template <int Nvel>
void custom_activation_dep_1(const double act_in[Nvel] , const int chan_id , double act_fcn[Nvel]){
    double ampl , ampl_1 , ampl_2;      // the width of a channels

    // First dimension of the activation (first channel):
    if (chan_id == 0){
        if (num_channels != 1){ // if there are multiple channels, calculate just the first one
            ampl = chan_array[1] - chan_array[0];
            for (int i = 0; i < Nvel ; i++) {
                act_fcn[i] = min( max( -( act_in[i] - chan_array[0] )/ampl + 1 , 0.0) , 1.0);
            }
        }
        else    // if there is only one channel then the activation function is only 1
            for (int i = 0; i < Nvel ; i++) {
                act_fcn[i] = 1.;
            }
    }
        // Last channel:
    else if (chan_id != 0 && chan_id == num_channels-1){
        ampl    = chan_array[num_channels-1] - chan_array[num_channels-2];
        for (int i = 0; i < Nvel ; i++) {
            act_fcn[i] = min( max( ( act_in[i] - chan_array[Nvel -2] )/ampl , 0.0) , 1.0);
        }
    }
    // All other channels:
    ampl_1  = chan_array[chan_id] - chan_array[chan_id - 1];
    ampl_2  = chan_array[chan_id + 1] - chan_array[chan_id];
    for (int i = 0; i < Nvel ; i++) {
        act_fcn[i] = min( max( ( act_in[i] - chan_array[chan_id -1] )/ampl_1 , 0.0) ,
                          max( -( act_in[i] - chan_array[chan_id] )/ampl_2 + 1 , 0.0)
        );
    }
}

void custom_activation_dep_2(const double act_in , const int chan_id , double act_fcn){
    double ampl , ampl_1 , ampl_2;      // the width of a channels

    // First dimension of the activation (first channel):
    if (chan_id == 0){
        if (num_channels != 1){ // if there are multiple channels, calculate just the first one
            ampl = chan_array[1] - chan_array[0];
            act_fcn = min( max( -( act_in - chan_array[0] )/ampl + 1 , 0.0) , 1.0);
        }
        else    // if there is only one channel then the activation function is only 1
            act_fcn = 1.;
    }
    // Last channel:
    else if (chan_id != 0 && chan_id == num_channels-1){
        ampl    = chan_array[num_channels-1] - chan_array[num_channels-2];
        act_fcn = min( max( ( act_in - chan_array[num_channels -2] )/ampl , 0.0) , 1.0);
    }
    // All other channels:
    else
        ampl_1  = chan_array[chan_id] - chan_array[chan_id - 1];
        ampl_2  = chan_array[chan_id + 1] - chan_array[chan_id];
        act_fcn = min( max( ( act_in - chan_array[chan_id -1] )/ampl_1 , 0.0) ,
                       max( -( act_in - chan_array[chan_id] )/ampl_2 + 1 , 0.0)
                       );
}

//███╗   ██╗███████╗██╗   ██╗██████╗  █████╗ ██╗         ███╗   ██╗███████╗████████╗██╗    ██╗ ██████╗ ██████╗ ██╗  ██╗
//████╗  ██║██╔════╝██║   ██║██╔══██╗██╔══██╗██║         ████╗  ██║██╔════╝╚══██╔══╝██║    ██║██╔═══██╗██╔══██╗██║ ██╔╝
//██╔██╗ ██║█████╗  ██║   ██║██████╔╝███████║██║         ██╔██╗ ██║█████╗     ██║   ██║ █╗ ██║██║   ██║██████╔╝█████╔╝
//██║╚██╗██║██╔══╝  ██║   ██║██╔══██╗██╔══██║██║         ██║╚██╗██║██╔══╝     ██║   ██║███╗██║██║   ██║██╔══██╗██╔═██╗
//██║ ╚████║███████╗╚██████╔╝██║  ██║██║  ██║███████╗    ██║ ╚████║███████╗   ██║   ╚███╔███╔╝╚██████╔╝██║  ██║██║  ██╗
//╚═╝  ╚═══╝╚══════╝ ╚═════╝ ╚═╝  ╚═╝╚═╝  ╚═╝╚══════╝    ╚═╝  ╚═══╝╚══════╝   ╚═╝    ╚══╝╚══╝  ╚═════╝ ╚═╝  ╚═╝╚═╝  ╚═╝

template<int Nin>                                // Nin = size on window of vel & acc (10)
double NN(const double vel_in[Nin], const double acc_in[Nin]){

    // Pedal params:
    double activated_acc[Nin][num_channels];     // array of activated accelerations for every acc in input (10)
    double proto_pedal_out[Nin];                 // pedal output for each neural network
    double final_pedal_out[1] = {0};               // final pedal (sum of proto pedals)

    // Drag params:
    double proto_drag_out[1];                            // drag not ReLUed
    double final_drag_out[1];                            // ReLUed drag

    // Final output of NN:
    double req_pedal;

    // -- DRAG --
    // Drag uses current vel.
    double drag_vel[1] = {vel_in[Nin-1]};
    proto_drag_out[0] = velocity_layer(drag_vel[0] , vel_weight[0][0]);
    // Negative ReLU:
    neg_ReLU<1>(proto_drag_out , final_drag_out);

    // -- PEDAL --
    custom_activation<Nin>(vel_in , acc_in , activated_acc );
    linear_layer<10>(activated_acc , pedal_weights , pedal_bias , proto_pedal_out);

    addition<10 , 1>(proto_pedal_out , final_pedal_out);
    // Add the pedal and drag:
    req_pedal = final_pedal_out[0] + final_drag_out[0];

    return req_pedal;
}

template<int Nin>                                // Nin = size on window of vel & acc (10)
double NN_depricated(const double vel_in[Nin], const double acc_in[Nin]){

    // Pedal params:
    double activated_acc[Nin][num_channels];     // array of activated accelerations for every acc in input (10)
    double act_fcn[Nin][num_channels];           // array of activated velocities for every vel in input (10)
    double proto_pedal_out[Nin];                 // pedal output for each neural network
    double final_pedal_out[1] = {0};               // final pedal (sum of proto pedals)

    // Drag params:
    double proto_drag_out[1];                            // drag not ReLUed
    double final_drag_out[1];                            // ReLUed drag

    // Final output of NN:
    double req_pedal;

    // -- DRAG --
    // Drag uses current vel.
    double drag_vel[1] = {vel_in[Nin-1]};
    proto_drag_out[0] = velocity_layer(drag_vel[0] , vel_weight[0][0]);
    // Negative ReLU:
    neg_ReLU<1>(proto_drag_out , final_drag_out);

    // -- PEDAL --
    for (int j = 0; j < Nin ; j++) { // for every row that the activation function makes
        for (int i = 0; i < num_channels ; i++) { // for every element of the row
            custom_activation<Nin>(vel_in , i , act_fcn[j]);
            activated_acc[j][i] = act_fcn[j][i]*acc_in[j];
            cout << " " << act_fcn[j][i];
        }
        cout << endl;
        // the specific row corresponds to a specific NN:
        proto_pedal_out[j] = linear_layer<Nin>(activated_acc[j] , pedal_weights[j] , pedal_bias[j]);
    }

    addition<10 , 1>(proto_pedal_out , final_pedal_out);
    // Add the pedal and drag:
    req_pedal = final_pedal_out[0] + final_drag_out[0];

    return req_pedal;
}


#endif //NN_CPP_TEST_NEURAL_NETWORK_H
