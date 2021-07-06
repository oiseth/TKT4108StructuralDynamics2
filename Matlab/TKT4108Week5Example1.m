clc
clear all
close all
%% Structural properties
M = 1;
K = 10;
omegan = sqrt(K/M);
zeta = 0.05;
C = 2*zeta*M*omegan;
%% Auto spectral density of the dynamic load X
w = linspace(-20,20,1001);
Sx = 1./(4+w.^2);
figure
plot(w,Sx)
xlabel('$\omega$','Interpreter','latex')
ylabel('$S_x(\omega)$','Interpreter','latex')
%% Frequency response function
H = zeros(1,length(w));
for k = 1:length(w)
    H(1,k) = inv(-w(1,k)^2*M+1i*w(1,k)*C+K);
end
figure 
plot(w,real(H))
hold on
plot(w,imag(H))
xlabel('$\omega$','Interpreter','latex')
ylabel('$H(\omega)$','Interpreter','latex')

%% Auto spectral density of the response
Sy = zeros(1,length(w));
for k = 1:length(w)
    Sy(1,k) = H(1,k)*Sx(1,k)*H(1,k)';
end  
figure
plot(w,real(Sy))
hold on
plot(w,imag(Sy))
var_y = real(trapz(w,Sy));
xlabel('$\omega$','Interpreter','latex')
ylabel('$S_y(\omega)$','Interpreter','latex')
%% Auto spectral density of the time derivative of the response
S_doty= w.^2.*Sy;
var_doty = real(trapz(w,S_doty));
figure
plot(w,real(S_doty))
hold on
plot(w,imag(S_doty))
xlabel('$\omega$','Interpreter','latex')
ylabel('$S_{\dot{y}}(\omega)$','Interpreter','latex')
%% Rayleigh probability density function of peaks
a = linspace(0,2,1000);
p = a./var_y.*exp(-1/2*a.^2/var_y);
figure
plot(a,p)
xlabel('$a$','Interpreter','latex')
ylabel('$p_{a}(a)$','Interpreter','latex')

%% Probability distribution function of the largest peak
T = 100;
v_yp = 1/2/pi*var_doty/var_y.*exp(-1/2*a.^2/var_y);
P_max = exp(-v_yp*T);
figure
plot(a,P_max)
xlabel('$a$','Interpreter','latex')
ylabel('$P_{max}(a)$','Interpreter','latex')
%% Probability density function of the largest peak
p_max = diff(P_max)/(a(2)-a(1));
figure(100)
plot(a(1:end-1),p_max)
hold on
xlabel('$a$','Interpreter','latex')
ylabel('$P_{max}(a)$','Interpreter','latex')






