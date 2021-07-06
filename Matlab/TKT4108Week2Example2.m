clc
clear all
close all
%% Define x(t)
t  =linspace(-10,10,10001);
alpha = 2;
x = exp(-alpha*t); % Auto-correlation function
x(t<0) = 0;
figure('Name','X(t)')
plot(t,x,'LineWidth',2)
ylim([0 2])
ylabel('$x(t)$','Interpreter','latex')
xlabel('$t$','Interpreter','latex')
%% Alternative 1 (Numerical integration)
omegaaxis = linspace(-50,50,1001); % Define frequency axis
X = zeros(1,length(omegaaxis));
for k =1:length(omegaaxis)
    X(1,k) = 1/(2*pi)*trapz(t,x.*exp(-1i*omegaaxis(1,k)*t));
end
F = 1./(alpha+1i*omegaaxis)/2/pi; % Analytical solution
figure('Name','Fourier transform by numerical integration')
plot(omegaaxis,real(X),'LineWidth',2,'DisplayName','Re(X)')
hold on
plot(omegaaxis,imag(X),'LineWidth',2,'DisplayName','Im(X)')
plot(omegaaxis,real(F),'DisplayName','Re(F)')
plot(omegaaxis,imag(F),'DisplayName','Re(F)')
xlim([-1 1]*2*pi*10)
ylabel(['$X(\omega)$' ' ' '$F(\omega)$'] ,'Interpreter','latex')
xlabel('$\omega$','Interpreter','latex')
grid on
legend show

%% Alternative 2: Using FFT
X = fft(x);
dt = t(2)-t(1);
f = linspace(-0.5/dt,0.5/dt,length(t));
omegaaxis = 2*pi*f;
X = fftshift(X);
X = exp(-1i*omegaaxis*-1*t(1)).*X*dt/2/pi; % The fft assumes that the time series start in t=0. (Use first shift teorem to get the correct result)

F = 1./(alpha+1i*omegaaxis)/2/pi;

figure('Name','Fourier transform by numerical integration')
plot(omegaaxis,real(X),'LineWidth',2,'DisplayName','Re(X)')
hold on
plot(omegaaxis,imag(X),'LineWidth',2,'DisplayName','Im(X)')
plot(omegaaxis,real(F),'DisplayName','Re(F)')
plot(omegaaxis,imag(F),'DisplayName','Re(F)')
xlim([-1 1]*2*pi*10)
ylabel(['$X(\omega)$' ' ' '$F(\omega)$'] ,'Interpreter','latex')
xlabel('$\omega$','Interpreter','latex')
grid on
legend show

