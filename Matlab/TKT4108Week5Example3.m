clc
clear all
close all
%% The auto spectral density of the u-comp according to N400
w = linspace(0,20,1000);
z=50;   %Height above ground,
z1=10;  %Reference height
L1=100;  %Reference integral length scale z1=10; L1=100;
xLu=L1*(z/z1)^0.3; % Integral length scale
Au=6.8/2/pi; % Constant in the auto-spectral density
V=40;   %Mean wind velocity
Iu=0.15;% Turbulence intensity
Su=zeros(1,length(w));
for k=1:length(w)
    Su(1,k)=(Iu*V)^2*Au*xLu/V/((1+1.5*Au*w(1,k)*xLu/V)^(5/3));
end
figure    
plot(w,Su,'-')    
xlabel('\omega')
ylabel('S(\omega)')

%% Monte Carlo simulations of the wind field, see also TKT4108Week2Example3 
t = linspace(0,600,6000);
phi = 2*pi*rand(1,length(w));
dw = w(2)-w(1);
u = zeros(1,length(t));
for k = 1:length(w)
    Ak = sqrt(2*Su(1,k)*dw);
    u = u + Ak*cos(w(1,k)*t+phi(1,k));
end
u = u + V ;

figure(1)
plot(t,u)
ylim([0 80])
grid on
ylabel('V (m/s)')
xlabel('t (s)')
