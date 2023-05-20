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
using namespace std;

//███╗   ███╗ █████╗ ██╗███╗   ██╗
//████╗ ████║██╔══██╗██║████╗  ██║
//██╔████╔██║███████║██║██╔██╗ ██║
//██║╚██╔╝██║██╔══██║██║██║╚██╗██║
//██║ ╚═╝ ██║██║  ██║██║██║ ╚████║
//╚═╝     ╚═╝╚═╝  ╚═╝╚═╝╚═╝  ╚═══╝

int main() {

    // ------------------------- SIMPLE TEST ------------------------

    double req_pedal_simple;

    double vel_1[10] = {1.0 , 0.5 , 0.0 , 0.0 , 0.0 , 0.0 , 0.0 , 0.0 , 0.0 ,0.0};
    double acc_1[10] = {1.0 , 0.5 , 0.0 , 0.0 , 0.0 , 0.0 , 0.0 , 0.0 , 0.0 ,0.0};

    req_pedal_simple = NN<10>(vel_1 , acc_1);
    cout << "The pedal is: " << req_pedal_simple << endl << endl;


    // -------------------------- FULL TEST -------------------------

    // Check CSV loads properly --- The code to make the csv's is in "compare_py_and_cpp_models.ipynb"
    if(check_load("acceleration.csv") && check_load("velocity.csv")){
        return 1;
    }

    // Variables used in csv parsing:
    double** vel;
    double** acc;
    int vel_rows, vel_columns;
    int acc_rows , acc_columns;
    // Parse the csvs
    parseCSV("acceleration.csv", acc, acc_rows, acc_columns);
    parseCSV("velocity.csv", vel, vel_rows, vel_columns);


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
