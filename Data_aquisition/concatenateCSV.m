clc
clear all
close all

%% Read individual csv's:

csv_const = readtable("Vehicle_Sim/basic_agent_st/bin/log_internal/Constant_Acc_Profile.csv");
csv_linear = readtable("Vehicle_Sim/basic_agent_st/bin/log_internal/Linear_Acc_Profile.csv");
csv_sine_1 = readtable("Vehicle_Sim/basic_agent_st/bin/log_internal/Sine_Acc_Profile_1.csv");
csv_chirp = readtable("Vehicle_Sim/basic_agent_st/bin/log_internal/Chirp_Acc_Profile.csv");

%% Combine csv's:
all_csv = [csv_linear ; csv_linear ; csv_sine_1 ; csv_chirp];

%% Remove NaNs:
a = table2array(all_csv);
a = rmmissing(a);

%% Normalize data

max_pedal = max(abs(a(:,2)));
max_velocity = max(abs(a(:,3)));
max_acceleration = max(abs(a(:,4)));

pedal = a(:,2)/max_pedal;
velocity = a(:,3)/max_velocity;
acceleration = a(:,4)/max_acceleration;

%% Make a time array consistent with the simulation, that isn't broken up from the data aquisition 

time = zeros(size(pedal,1) , 1);
for i = 1:size(pedal,1)
    time(i) = (i-1) * 0.05;
end

%% Reconvert to table:

a_normalised = [time , pedal , velocity , acceleration];

final_csv = array2table(a_normalised , ...
                        "VariableNames",{'time' , 'pedal' , 'velocity' , 'acceleration'});

%% Save to file:

writetable(final_csv , '/Users/francescomaraschin/Desktop/IntelligentVehicles/Project_NN_Conda/Data_aquisition/forward_profiles.csv');
