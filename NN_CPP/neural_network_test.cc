//
// Created by Frans on 04/02/23.
//

#include "neural_network.h"

const double vel_weight[1][1] = {
        {0.10332104563713074}
};

const double pedal_bias[10] = {
        0.005141090136021376,
        0.005141090136021376,
        0.005141090136021376,
        0.005141090136021376,
        0.005141090136021376,
        0.005141090136021376,
        0.005141090136021376,
        0.005141090136021376,
        0.005141090136021376,
        0.005141090136021376
};

const double pedal_weights[10][10] = {
        {-0.30136939883232117,	-0.214010551571846,	    -0.16725453734397888,	    -0.04437851533293724,	    0.0029236003756523132,	0.006258087698370218,	    -0.03809134662151337,	    0.04702611267566681,	    0.08145902305841446,	    0.20249931514263153},
        {-0.1438508927822113,	-0.13459661602973938,	    -0.04960713908076286,	    -0.10883837938308716,	    0.01808840036392212,	    -0.02525126375257969,	    -0.08118733018636703,	    0.03148369491100311,	    0.0643184632062912,	    0.14356282353401184},
        {-0.12340202182531357,	0.014678167179226875,	    -0.0030007080640643835,	0.009297288954257965,	    -0.07514441758394241,	    -0.10522723197937012,	    0.02867375686764717,	    0.005200916435569525,	    0.044155675917863846,	    -0.03207032009959221},
        {-0.009801373817026615,	-0.030726730823516846,	0.012429225258529186,	    0.021613532677292824,	    0.010301504284143448,	    -0.055800363421440125,	-0.015940342098474503,	0.0011695045977830887,	0.013756475411355495,	    -0.08429261296987534},
        {-0.08396259695291519,	0.03814958781003952,	    0.039440400898456573,	    0.041117552667856216,	    0.014303631149232388,	    -0.049486346542835236,	-0.04249114915728569,	    -0.051424507051706314,	0.0055778417736291885,	0.023866454139351845},
        {0.03437018021941185,	0.006114837247878313,	    -0.05621477589011192,	    0.023320548236370087,	    0.02356317639350891,	    -0.010998508892953396,	-0.022101979702711105,	0.000839356507640332,	    -0.0018634769367054105,	-0.023783693090081215},
        {0.037523236125707626,	0.0007906569517217577,	-0.004342230502516031,	0.020761389285326004,	    0.03701533377170563,	    -0.06056881695985794,	    -0.001963408663868904,	0.03621814399957657,	    -0.033606141805648804,	0.03177512064576149},
        {0.06005530804395676,	0.017844025045633316,	    -0.0063252635300159454,	-0.014921199530363083,	0.005459195468574762,	    0.006498144473880529,	    0.02150900289416313,	    -0.011614077724516392,	-0.035147953778505325,	0.02336888201534748},
        {0.093944251537323,	    0.11355642229318619,	    -0.032044436782598495,	-0.01861685886979103,	    0.010040344670414925,	    0.0393521673977375,	    -0.02295413613319397,	    0.060194239020347595,	    0.03328411281108856,	    0.018043437972664833},
        {0.07680781185626984,	0.087253138422966,	    -0.01785457693040371,	    -0.012010874226689339,	-0.05588880553841591,	    0.054051171988248825,	    -0.040108777582645416,	-0.023625759407877922,	-0.03405225649476051,	    -0.03506312146782875}
};


const int num_channels = 10;
const double chan_array[num_channels] = {
        -1.0,
        -0.8114791181352403,
        -0.6229582362704806,
        -0.43443735440572107,
        -0.24591647254096138,
        -0.0573955906762017,
        0.13112529118855787,
        0.31964617305331755,
        0.5081670549180772,
        0.6966879367828369
};


double SSNN(const double vel_in[10], const double acc_in[10] ){

    // Pedal params:
    double activated_acc[10][num_channels];         // array of activated accelerations for every acc in input (10)
    double act_fcn[10][num_channels];               // array of activated velocities for every vel in input (10)
    double proto_pedal_out[10];                     // pedal output for each neural network
    double final_pedal_out[1] = {0};           // final pedal (sum of proto pedals)

    // Drag params:
    double proto_drag_out[1];                       // drag not ReLUed
    double final_drag_out[1];                        // ReLUed drag


    // Final pedal output of NN:
    double req_pedal;

    // -- DRAG --
    // Drag uses current vel.
    double drag_vel[1] = {vel_in[9]};
    velocity_layer<1>(drag_vel , vel_weight , proto_drag_out);
    // Negative ReLU:
    neg_ReLU<1>(proto_drag_out , final_drag_out);

    // -- PEDAL --
    // Activation:
    for (int j = 0; j < 10 ; j++) { // for every row that the activation function makes
        for (int i = 0; i < num_channels ; i++) { // for every element of the row
            custom_activation<10,num_channels>(vel_in , i , chan_array , act_fcn[j]);
            activated_acc[j][i] = act_fcn[j][i]*acc_in[j];
        }
        // the specific row corresponds to a specific NN:
        proto_pedal_out[j] = linear_layer<10>(activated_acc[j] , pedal_weights[j] , pedal_bias[j]);
    }

    // Add the calculated pedals:
    for (double i : proto_pedal_out) {
        final_pedal_out[0] = final_pedal_out[0] + i;
    }
    // Add the pedal and drag:
    req_pedal = final_pedal_out[0] + final_drag_out[0];

    return req_pedal;
}

int main() {

    double vel_window[10] = {0,0,0,0,-0.000006,-0.000054,-0.000407,-0.001846,0.000047,0.001719};
    double acc_window[10] = {0.059624,0.059353,0.059656,0.059939,0.060241,0.059911,0.059815,0.059599,0.059458,0.059668};
    double req_pedal;

    req_pedal = SSNN(vel_window , acc_window);
    std::cout << "Requested Pedal: " << req_pedal << std::endl;
    return 0;
}
