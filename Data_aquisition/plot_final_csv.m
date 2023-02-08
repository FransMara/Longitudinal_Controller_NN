clc 
clear all 
close all

%% Load data
data = readtable("forward_profiles.csv" , "Delimiter", {',' , ';'});

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
xlabel 'time (s)'
ylabel 'velocity (km/h)'
title 'Velocity profile'
legend('velocity')

nexttile
hold on
plot(Acutal_Acc)
plot(Requested_Acc)
xlabel 'time (s)'
ylabel 'Acceleration (m/s^2)'
title 'Acceleration profile'
legend('Actual Acceleration' , 'Requested Acceleration')

