clc
clear all
close all
%% The auto spectral density of the u-comp according to N400
w = linspace(0,20,5000);
xLu = 100; % Integral length scale
Au = 6.8/2/pi; % Constant in the auto-spectral density
V = 20;   %Mean wind velocity
Iu = 0.15;% Turbulence intensity
Su = zeros(1,length(w));
for k=1:length(w)
    Su(1,k)=(Iu*V)^2*Au*xLu/V/((1+1.5*Au*w(1,k)*xLu/V)^(5/3));
end
figure
plot(w,Su,'-')
xlabel('\omega')
ylabel('S_u(\omega)')

%% Monte Carlo simulations of the wind field, see also TKT4108Week2Example3
t = linspace(0,600,6000);
phi = 2*pi*rand(1,length(w));
dw = w(2)-w(1);
u = zeros(1,length(t));
for k = 1:length(w)
    Ak = sqrt(2*Su(1,k)*dw);
    u = u + Ak*cos(w(1,k)*t+phi(1,k));
end

figure(1)
plot(t,u+V)
ylim([0 40])
grid on
ylabel('V (m/s)')
xlabel('t (s)')
%% Comparison of wind load models
rho = 1.25;
A = 1;
Cd = 2;
Fw_exact = 1/2*rho*A*Cd*(V+u).^2;
Fw_approx = 1/2*rho*A*Cd*V^2*(1+2*u/V);
figure
plot(t,Fw_exact,'DisplayName','Exact')
hold on
plot(t,Fw_approx,'DisplayName','Approx')
grid on
legend show
ylabel('$F_w(t)$','Interpreter','latex')
xlabel('$t$','Interpreter','latex')
%% Frequency response function
M = 100;
fn = 2;
omegan = 2*pi*fn;
K = M*omegan^2;
zeta = 0.005;
C = 2*M*omegan*zeta;

H = 1./(-w.^2*M + 1i*w*C + K);

figure
plot(w,real(H),'-')
hold on
plot(w,imag(H),'-')
xlabel('\omega')
ylabel('H(\omega)')

%% Auto spectral density of the response
Sy = (rho*A*Cd*V)^2*conj(H).*H.*Su;
figure
plot(w,real(Sy),'-')
hold on
plot(w,imag(Sy),'-')
xlabel('\omega')
ylabel('S_y(\omega)')

%% Probability distribution of the response
sigma_y = real(sqrt(trapz(w,Sy)));
mu_y = 1/2*rho*A*Cd*V^2/K;

pd_y = makedist('Normal',mu_y,sigma_y);
yp = linspace(-0.2,0.2,1000);
pdf_y = pdf(pd_y,yp);

figure
plot(yp,pdf_y)
xlabel('$y_1$','Interpreter','latex')

%% Probability distribution of the largest peak in T = 600s (Zero mean assumed)
a = linspace(0,0.2,1000);
sigma_ydot = real(sqrt(trapz(w,w.^2.*Sy)));
vy = 1/2/pi*sigma_ydot/sigma_y*exp(-1/2*(a/sigma_y).^2);
T = 600;
Pmax = exp(-vy*T);

figure
plot(a+mu_y,Pmax)
xlabel('$a$','Interpreter','latex')
ylabel('$P_{max}(a)$','Interpreter','latex')

%% Probability density function of the largest peak in T = 600s (Zero mean assumed)
pmax = diff(Pmax)/(a(2)-a(1));

figure(100)
plot(a(1:end-1)+mu_y,pmax)
hold on
xlabel('$a$','Interpreter','latex')
ylabel('$p_{max}(a)$','Interpreter','latex')

%% Time domain simulations
% Wind
Nsim = 10;
t = 0:0.03:600;
phi = 2*pi*rand(Nsim,length(w));
dw = w(2)-w(1);
u = zeros(Nsim,length(t));
for n = 1 : Nsim
    for k = 1:length(w)
        Ak = sqrt(2*Su(1,k)*dw);
        u(n,:) = u(n,:) + Ak*cos(w(1,k)*t+phi(n,k));
    end
end
% Wind load
Fw_exact = 1/2*rho*A*Cd*(V+u).^2;
Fw_approx = 1/2*rho*A*Cd*V^2*(1+2*u/V);

% Dynamic response
y_exact = zeros(Nsim,length(t));
y_approx = zeros(Nsim,length(t));
for k = 1:Nsim
    y_exact(k,:) = linear_newmark_krenk(M,C,K,Fw_exact(k,:),0,0,t(2)-t(1),1/2,1/4);
    y_approx(k,:) = linear_newmark_krenk(M,C,K,Fw_approx(k,:),0,0,t(2)-t(1),1/2,1/4);
end

figure
for k = 1:Nsim
    subplot(5,2,k)
    plot(t,y_exact(k,:),'displayName','Exact')
    hold on
    plot(t,y_approx(k,:),'--','DisplayName','Approx')
end
legend show
% Comparison of mean extreme value
Expected_max_TD_Exact = mean(max(y_exact,[],2))
Expected_max_TD_approx = mean(max(y_approx,[],2))

%% The newmark algorithm
% Function used to calculate the response
function [u, udot, u2dot] = linear_newmark_krenk(M,C,K,f,u0,udot0,h,gamma,beta)
% Initialize variables
u = zeros(size(M,1),size(f,2));
udot = zeros(size(M,1),size(f,2));
u2dot = zeros(size(M,1),size(f,2));
% Insert initial conditions in response vectors
u(:,1) = u0;
udot(:,1) = udot0;
% Calculate "modified mass"
Mstar = M + gamma*h*C + beta*h^2*K;
% Calculate initial accelerations
u2dot(:,1) = M\(f(:,1)-C*udot(:,1)-K*u(:,1));  % This is a faster version of inv(M)*(P(:,1)-C*udot0-K*u0);
for n = 1:size(f,2)-1
    % Predicion step
    udotstar_np1 = udot(:,n) + (1-gamma)*h*u2dot(:,n);
    ustar_np1 = u(:,n) + h*udot(:,n) + (1/2-beta)*h^2*u2dot(:,n);
    % Correction step
    u2dot(:,n+1) = Mstar\(f(:,n+1)-C*udotstar_np1-K*ustar_np1);
    udot(:,n+1) = udotstar_np1 + gamma*h*u2dot(:,n+1);
    u(:,n+1) = ustar_np1 + beta*h^2*u2dot(:,n+1);
end
end






