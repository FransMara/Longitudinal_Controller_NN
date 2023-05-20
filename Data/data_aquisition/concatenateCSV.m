clc
clear all
close all

%% Read csv:
raw_data = readtable("csv/raw_data.csv");

%% Remove NaNs:
data_no_nan = table2array(raw_data);
data_no_nan = rmmissing(data_no_nan);

%% Split raw data:

raw_pedal = raw_data.pedal;
raw_acc   = raw_data.acceleration;
raw_vel   = raw_data.velocity;

%% Plot raw data:

figure()
plot(raw_data.cycle,raw_data.pedal);
hold on
plot(raw_data.cycle,raw_data.velocity);
plot(raw_data.cycle,raw_data.acceleration);
hold off

%% Find Max of data:
max_pedal = max(abs(raw_pedal));
max_velocity = max(abs(raw_vel));
max_acceleration = max(abs(raw_acc));

%% Filter the data: (Moving average filter)
data_smoothed = data_no_nan;

for J = 1:50
    % Smooth the data:
    for I = 1:4
        data_smoothed(:,I) = smooth(data_smoothed(:,I),'sgolay',0);
    end
end

% Filter the data:
coeff_a = 1;
coeff_b = [1/4 1/4 1/4 1/4];
data_filtered = filter(coeff_b , coeff_a , data_smoothed);


figure 
hold on
plot(data_no_nan(:,4));
plot(data_filtered(:,4));
legend("Unfiltered" , "Filtered")

a = data_filtered;

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

writetable(final_csv , '/Users/francescomaraschin/Documents/Universtiy/IntelligentVehicles/LongitudinalControllerNN/Data/csv/filtered_real_world_data.csv');
