//
// Created by Frans on 04/02/23.
//

#include <iostream>
#include <cstdio>
#include <vector>

#ifndef NN_CPP_TEST_NEURAL_NETWORK_H
#define NN_CPP_TEST_NEURAL_NETWORK_H

template <int Nin>
double linear_layer( const double data_in[Nin] , const double w[Nin] , const double b ){
    double data_out = 0;
    for(int i=0 ; i<Nin ; i++){
        data_out += data_in[i]*w[i];
    }
    data_out += b;
    return data_out;
}

// Nout = Nin in this case as in general there is no combining of values and only a passing on inputs through a
//            squaring and then through a multiplication of the weight
template <int Nin>
void velocity_layer( const double data_in[Nin] , const double w[Nin][Nin] , double data_out[Nin]){
    for(int j=0 ; j<Nin ; j++){
        data_out[j] = 0;
        for(int i=0 ; i<Nin ; i++) {
            data_out[j] += data_in[i] * data_in[i];
            data_out[j] += data_out[j] * w[j][i];
        }
    }
}

template<int Nin>
void neg_ReLU( const double data_in[Nin] , double data_out[Nin]){
    for(int i = 0; i < Nin ; i++ ){
        if (data_in[i] < 0)
            data_out[i] = 0;
        else if (data_in[i] >= 0)
            data_out[i] = -data_in[i];
    }
}

template <int Nin , int Nout>
void addition( const double data_in[Nin] , double data_out[Nout]){
    for(int j=0 ; j<Nout ; j++){
        data_out[j] = 0;
        for(int i=0 ; i<Nin ; i++){
            data_out[j] += data_in[i];
        }
    }
}

// Custom activation function:
// IN : Nvel       = size of input velocity array
//      Nchan       = number of activation channels
//      act_in     = velocities that activate the channels
//      chan_id    = index of the channel center
//      chan_array = array containing the velocities corresponding to the channel centers
template <int Nvel , int Nchan>
void custom_activation(const double act_in[Nvel] , const int chan_id , const double chan_array[Nchan] , double act_fcn[Nvel]){

    double ampl , ampl_1 , ampl_2;      // the width of a channel
    // shared pointer
    // First dimension of the activation (first channel):
    if (chan_id == 0){
        if (Nchan != 1){ // if there are multiple channels, calculate just the first one
            ampl = chan_array[1] - chan_array[0];
            for (int i = 0; i < Nvel ; i++) {
                act_fcn[i] = std::min( std::max( -( act_in[i] - chan_array[0] )/ampl + 1 , 0.0) , 1.0);
            }
        }
        else    // if there is only one channel then the activation function is only 1
            for (int i = 0; i < Nvel ; i++) {
                act_fcn[i] = 1.;
            }
    }
    // Last channel:
    else if (chan_id != 0 && chan_id == Nchan-1){
        ampl    = chan_array[Nchan-1] - chan_array[Nchan-2];
        for (int i = 0; i < Nvel ; i++) {
            act_fcn[i] = std::min( std::max( -( act_in[i] - chan_array[Nvel -2] )/ampl , 0.0) , 1.0);
        }
    }
    // All other channels:
    ampl_1  = chan_array[chan_id] - chan_array[chan_id - 1];
    ampl_2  = chan_array[chan_id + 1] - chan_array[chan_id];
    for (int i = 0; i < Nvel ; i++) {
        act_fcn[i] = std::min( std::max( -( act_in[i] - chan_array[chan_id -1] )/ampl_1 , 0.0) ,
                               std::max( -( act_in[i] - chan_array[chan_id] )/ampl_2 + 1 , 0.0)
                               );
    }
}

#endif //NN_CPP_TEST_NEURAL_NETWORK_H
