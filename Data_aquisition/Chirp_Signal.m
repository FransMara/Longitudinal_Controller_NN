clear all
close all

% Script that makes a chirp signal manually:

T  = 400;
x = zeros(1,T);

phase_0 = 0;    % initial phase at t=0 
f_0 = 0.01;        % initial frequency at t=0 [Hz]
f_1 = 1;      % final frequency at t=T [Hz]


for i=1:1:T
    x(i) = sin( phase_0 + 2*pi * ( (-f_0 * f_1 * T)/(f_1 - f_0) ) * log(1 - ((f_1 - f_0)/(f_1*T))*i) );
end


plot(x);



