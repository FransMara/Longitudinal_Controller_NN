clc
clear
close all

%% Load data
acc_data = readtable("basic_agent_st/bin/log_internal/Accelerations.csv" , "Delimiter", {',' , ';'});


%% Read data
time = table2array(acc_data(:, 1));
PID_output = table2array(acc_data(:, 2));
NN_output = table2array(acc_data(:, 3));

%% Plot data 

figure
hold on
plot(time, PID_output)
plot(time, NN_output)
xlabel 'time (s)'
ylabel 'acc'
title 'PID & NN Output'
legend('PID' , 'NN')

