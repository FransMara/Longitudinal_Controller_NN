clc
clear
close all

%% Load data
% Constant acceleration profile:
%data = readtable("basic_agent_st/bin/log_internal/Constant_Acc_Profile.csv" , "Delimiter", {',' , ';'});
% Linear acceleration profile:
%data = readtable("basic_agent_st/bin/log_internal/Linear_Acc_Profile.csv" , "Delimiter", {',' , ';'});
% Sinusoidal acceleration profile:
data = readtable("basic_agent_st/bin/log_internal/Sine_Acc_Profile_2.csv" , "Delimiter", {',' , ';'});
% Chirp acceleration profile:
%data = readtable("basic_agent_st/bin/log_internal/Chirp_Acc_Profile.csv" , "Delimiter", {',' , ';'});

%% Read data
time = table2array(data(:, 1));
Requested_Acc = table2array(data(:, 2));
Acutal_Velocity = table2array(data(:, 3));
Acutal_Acc = table2array(data(:, 4));

%% Plot data 

figure
tiledlayout(2,1)
nexttile
hold on
plot(time, Acutal_Velocity)
xlabel 'time (s)'
ylabel 'velocity (km/h)'
title 'Velocity profile'
legend('velocity')

nexttile
hold on
plot(time, Acutal_Acc)
plot(time, Requested_Acc)
xlabel 'time (s)'
ylabel 'Acceleration (m/s^2)'
title 'Acceleration profile'
legend('Actual Acceleration' , 'Requested Acceleration')