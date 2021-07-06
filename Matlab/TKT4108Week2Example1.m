clc
clear all
close all
%% Define square waveform
dt =0.01;
t = 0:dt:10;
x = zeros(1,length(t));
x(t<5) = 1;

figure('Name','Time series')
plot([t-20 t-10 t t+10],[x x x x],'--','DisplayName','x(t)')
hold on
plot(t,x,'LineWidth',2,'DisplayName','One period')
ylim([-2 2])
grid on
ylabel('$X$','Interpreter','latex')
xlabel('$t$','Interpreter','latex')
title('Square wave')
legend show
%% Obtain Fourier coefficients (Alternative 1: Trapizoidial rule)
Nterms = 10;
T = max(t);
a0 = 1/T*trapz(t,x);
tic
ak = zeros(1,Nterms);
bk = zeros(1,Nterms);
for k = 1:Nterms
    ak(1,k) = 1/T*trapz(t,x.*cos(2*pi*k/T*t));
    bk(1,k) = 1/T*trapz(t,x.*sin(2*pi*k/T*t));
end

omegak = 2*pi/T*(1:Nterms);
figure('Name','Real Fourier series')
subplot(1,2,1)
plot(1:Nterms,ak,'o-')
xlabel('$k$','Interpreter','latex')
ylabel('$a_k$','Interpreter','latex')
grid on
ylim([-1 1])
subplot(1,2,2)
plot(1:Nterms,bk,'o-')
xlabel('$k$','Interpreter','latex')
ylabel('$b_k$','Interpreter','latex')
grid on
ylim([-1 1])
% Plot Fourier series approximation
tp = linspace(-40,40,1000);
X_Fourier = ones(1,length(tp))*a0;
for k = 1:Nterms
    X_Fourier = X_Fourier + 2*(ak(1,1)*cos(2*pi*k/t(end)*tp) + bk(1,k)*sin(2*pi*k/t(end)*tp));
end
figure(1)
plot(tp,X_Fourier,'DisplayName',[num2str(Nterms) 'Terms'])
legend show

%% Obtain Fourier coefficients (Alternative 2: Trapezoidal rule, complex Fourier series)
Nterms = length(t);
Xk = zeros(1,Nterms);
for k = 0:Nterms-1
    Xk(1,k+1) = 1/T*trapz(t,x.*exp(-1i*2*pi/T*k*t));
end
figure('Name','rapizoidial rule, complex Fourier series')
subplot(1,2,1)
plot(0:length(Xk)-1,real(Xk),'-o')
xlabel('$k$','Interpreter','latex')
ylabel('$Re(X_k)$','Interpreter','latex')
xlim([0 20])
ylim([-1 1])
grid on
subplot(1,2,2)
plot(0:length(Xk)-1,imag(Xk),'-o')
xlabel('$k$','Interpreter','latex')
ylabel('$Im(X_k)$','Interpreter','latex')
xlim([0 10])
ylim([-1 1])
grid on
%% Obtain Fourier coefficients (Alternative 3: Left Rectangular Rule, complex Fourier series)
% This is what is typically implemented in softwares as the descrete
% Fourier transform
clc
tic
N = length(t);
Xk = zeros(1,N);
for k = 0:length(t)-1
    Xk(1,k+1) = 1/N*x*exp(-1i*2*pi*k*(0:N-1)/N).';
end
toc
figure('Name','Alternative 3: Left Rectangular Rule, complex Fourier series')
subplot(1,2,1)
plot(0:length(Xk)-1,real(Xk),'-o')
xlabel('$k$','Interpreter','latex')
ylabel('$Re(X_k)$','Interpreter','latex')
xlim([0 20])
ylim([-1 1])
grid on
subplot(1,2,2)
plot(0:length(Xk)-1,imag(Xk),'-o')
xlabel('$k$','Interpreter','latex')
ylabel('$Im(X_k)$','Interpreter','latex')
xlim([0 10])
ylim([-1 1])
grid on


%% Obtain Fourier coefficients (Alternative 4: The fast Fourier transform)
clc
tic
Xk = fft(x)/N;
toc
figure('Name','Alternative 4, The fast Fourier transform')
subplot(1,2,1)
plot(0:length(Xk)-1,real(Xk),'-o')
xlabel('$k$','Interpreter','latex')
ylabel('$Re(X_k)$','Interpreter','latex')
xlim([0 20])
ylim([-1 1])
grid on
subplot(1,2,2)
plot(0:length(Xk)-1,imag(Xk),'-o')
xlabel('$k$','Interpreter','latex')
ylabel('$Im(X_k)$','Interpreter','latex')
xlim([0 10])
ylim([-1 1])
grid on



