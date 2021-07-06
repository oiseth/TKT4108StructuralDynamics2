clc
clear all
close all
%% Monte Carlo simulation of time series (Alternative 1: Sum of cosine terms)
domega = 0.001;
omegaaxis = domega/2:domega:10;
t = linspace(0,600,6000);

Sx = zeros(1,length(omegaaxis));
Sx(omegaaxis<1) = 1;
Sx(omegaaxis<0.9) = 0;

figure
plot(omegaaxis,Sx)
ylim([0 2])
xlabel('$\omega$','Interpreter','latex')
ylabel('$S_x(\omega)$','Interpreter','latex')

%% Monte Carlo simulations 
phi = 2*pi*rand(1,length(omegaaxis));
x = zeros(1,length(t));
for k = 1:length(omegaaxis)
    Ak = sqrt(2*Sx(1,k)*domega);
    x = x + Ak*cos(omegaaxis(1,k)*t+phi(1,k));
end

figure(1)
plot(t,x)
hold on

%% Monte Carlo simulation of time series (Alternative 2: FFT)
NFFT = length(omegaaxis);
%phi = 2*pi*rand(1,length(omegaaxis));
c=sqrt(Sx).*exp(1i.*phi);
x = real(ifft(c))*NFFT*sqrt(2*domega);
t=linspace(0,2*pi/domega,NFFT);
figure(1)
plot(t,x)
xlim([0 600])


