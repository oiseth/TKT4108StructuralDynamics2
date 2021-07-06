clear all
close all
clc
%% Structural properties
M = 100;
K = 500;
C = 2*M*sqrt(K/M)*0.05;
omegaaxis = linspace(-10,10,1001);
%% Auto spectral density of the load
S0 = 1;
Sx = S0*ones(1,length(omegaaxis));
figure
plot(omegaaxis,Sx)
xlabel('$\omega$','Interpreter','latex')
ylabel('$S_x(\omega)$','Interpreter','latex')
%% Frequency response function
H = 1 ./ ((-omegaaxis.^2*M) + (1i*C*omegaaxis) + K);
figure
plot(omegaaxis,real(H),'DisplayName','Re')
hold on
plot(omegaaxis,imag(H),'DisplayName','Im')
xlabel('$\omega$','Interpreter','latex')
ylabel('$H(\omega)$','Interpreter','latex')
legend show
%% Auto spectral density of the response
Sy = zeros(1,length(omegaaxis));
for j = 1:length(omegaaxis)
    Sy(1,j) = H(1,j)*Sx(1,j)*H(1,j)';
end

figure
plot(omegaaxis,real(Sy),'DisplayName','Re')
hold on
plot(omegaaxis,imag(Sy),'DisplayName','Im')
xlabel('$\omega$','Interpreter','latex')
ylabel('$S_y(\omega)$','Interpreter','latex')
legend show

%% Standard deviation of response
Sy_var = trapz(omegaaxis, Sy);
Sy_std = sqrt(Sy_var);
disp(['The standard deviation of the response is ', num2str(Sy_std)])
