//
// Created by Frans on 04/02/23.
//

//██╗███╗   ██╗ ██████╗██╗     ██╗   ██╗██████╗ ███████╗███████╗
//██║████╗  ██║██╔════╝██║     ██║   ██║██╔══██╗██╔════╝██╔════╝
//██║██╔██╗ ██║██║     ██║     ██║   ██║██║  ██║█████╗  ███████╗
//██║██║╚██╗██║██║     ██║     ██║   ██║██║  ██║██╔══╝  ╚════██║
//██║██║ ╚████║╚██████╗███████╗╚██████╔╝██████╔╝███████╗███████║
//╚═╝╚═╝  ╚═══╝ ╚═════╝╚══════╝ ╚═════╝ ╚═════╝ ╚══════╝╚══════╝

#include "neural_network.h"
#include "csv_functions.h"
#include "weights_biases .h"
using namespace std;


//███╗   ███╗ █████╗ ██╗███╗   ██╗
//████╗ ████║██╔══██╗██║████╗  ██║
//██╔████╔██║███████║██║██╔██╗ ██║
//██║╚██╔╝██║██╔══██║██║██║╚██╗██║
//██║ ╚═╝ ██║██║  ██║██║██║ ╚████║
//╚═╝     ╚═╝╚═╝  ╚═╝╚═╝╚═╝  ╚═══╝

int main() {

    // Check CSV loads properly
    if(check_load("acceleration.csv") && check_load("velocity.csv")){
        return 1;
    }

    // Variables used in csv parsing:
    double** vel;
    double** acc;
    int vel_rows, vel_columns;
    int acc_rows , acc_columns;
    // Parse the csvs
    parseCSV("velocity.csv", vel, vel_rows, vel_columns);
    parseCSV("acceleration.csv", acc, acc_rows, acc_columns);

    // Print the parsed data to cout:
    //print_data(vel_rows, vel_columns, vel);
    //print_data(acc_rows , acc_columns , acc);

    // Print summary of data:
    summary(vel_rows , vel_columns , "Velocity");
    summary(acc_rows , acc_columns , "Acceleration");

    // Declare final required pedal:
    double req_pedal[vel_rows];

    for (int i = 0; i < vel_rows ; ++i) {
        req_pedal[i] = NN<10>(vel[i] , acc[i] );
        cout << req_pedal[i] << endl;
    }


    make_csv_no_col(vel_rows ,req_pedal , "pedal_out_cpp.csv");

    // Clean the memory:
    clean_memory(vel_rows , vel);
    clean_memory(acc_rows,acc);

    return 0;
}
