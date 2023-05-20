clc 
clear all 
close all

%% Load data
data = readtable("/Users/francescomaraschin/Library/Mobile Documents/com~apple~CloudDocs/Universtiy/IntelligentVehicles/LongitudinalControllerNN/Data/csv/filtered_real_world_data1.csv" , ...
                 "Delimiter", {',' , ';'} ...
                );

%% Read data
Requested_Acc = table2array(data(:, 2));
Acutal_Velocity = table2array(data(:, 3));
Acutal_Acc = table2array(data(:, 4));

%% Plot data 

figure
tiledlayout(2,1)
nexttile
hold on
plot(Acutal_Velocity)
xlabel 'samples'
ylabel 'velocity (km/h)'
title 'Velocity profile'
legend('velocity')

nexttile
hold on
plot(Acutal_Acc)
plot(Requested_Acc)
xlabel 'samples'
ylabel 'Acceleration (m/s^2)'
title 'Acceleration profile'
legend('Actual Acceleration' , 'Requested Acceleration')

