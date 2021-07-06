%% Lecture 1 Structural Dynamics 2: What is random vibrations?
% In this example we will study how random vibrations are different from
% deterministic vibrations. We start by considering a portal frame with two
% degrees of freedom $y_1$ and $y_2$. The frame is subjected to two loads
% $X_1$ and $X_2$
%%
close all 
clear all
clc
%% Structural properties
m = 1; % Mass of each story
k = 100; % Stiffness
MM = eye(2)*m; % Mass matrix
KK = [2 -1; -1 1]*k; % Stiffness matrix
% Calculate modes and frequencies
[v, lambda] = eig(KK,MM); % Generalized eigenvalue problem
v(:,1) = v(:,1)/max(abs(v(:,1)));
v(:,2) = v(:,2)/max(abs(v(:,2)));
f = diag(lambda).^0.5/2/pi; % Natural frequencies in Hz
omega = f*2*pi;
zeta = [5 5]/100; % 5% damping for both modes
% Rayleigh damping
alpha1=2*omega(1)*omega(2)*(zeta(2)*omega(1)-zeta(1)*omega(2))/(omega(1)^2-omega(2)^2);
alpha2=2*(zeta(1)*omega(1)-zeta(2)*omega(2))/(omega(1)^2-omega(2)^2);
CC = alpha1*MM + alpha2*KK;
%% Dynamic response due to harmonic load
h = 0.05; % Time step
t = 0:h:600; % time
fl = 2; % load frequency
p0 = 100; % load amplitude
u0 = [0 0]'; % Initial displacements
udot0 = [0 0]'; % Initial velocities
gamma = 1/2; beta = 1/4; % Factors in the Newmark algorithm 
X = [p0*sin(2*pi*fl*t);p0*sin(2*pi*fl*t)]; % load vector
y = linear_newmark_krenk(MM,CC,KK,X,u0,udot0,h,gamma,beta); % Calculate response
% Plot harmonic response
plot_response(t,y,X)

%% Dynamic response due to stochastic load 
rho_X1_X2 = 0.8; % Load correlation coefficient
stdX1 = 100; % Standard deviation X1
stdX2 = 100; % Standard deviation X2

covmX = [stdX1^2 rho_X1_X2*stdX1*stdX2; rho_X1_X2*stdX1*stdX2 stdX2^2];

X = mvnrnd([0 0],covmX,length(t))';

figure
plot(X(1,:),X(2,:),'o')
axis([-1 1 -1 1]*1.5*max(max(abs(X))))
xlabel('$X_1$','Interpreter','Latex')
ylabel('$X_2$','Interpreter','Latex')


y = linear_newmark_krenk(MM,CC,KK,X,u0,udot0,h,gamma,beta); % Calculate response
% Plot response
plot_response(t,y,X);
% Plot probability distributions
mu_y1 = 0;
sigma_y1 = std(y(1,:));
mu_y2 = 0;
sigma_y2 = std(y(2,:));
pd_y1 = makedist('Normal',mu_y1,sigma_y1);
pd_y2 = makedist('Normal',mu_y2,sigma_y2);
yp = linspace(-10,10,1000);
pdf_y1 = pdf(pd_y1,yp);
pdf_y2 = pdf(pd_y2,yp);
figure
plot(yp,pdf_y1,'DisplayName','y_1')
hold on
plot(yp,pdf_y2,'DisplayName','y_1')
legend show





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
%% A function for plotting the response
function plot_response(t,y,X)
hf = figure;
subplot(2,2,1)
plot(t,X(2,:))
ylim([-1 1]*2*max(max(abs(X))));
xlim([0 100])
ylabel('$X_2(t)$','Interpreter','Latex')
grid on

subplot(2,2,3)
plot(t,X(1,:))
ylim([-1 1]*2*max(max(abs(X))));
xlim([0 100])
ylabel('$X_1(t)$','Interpreter','Latex')
grid on

subplot(2,2,2)
plot(t,y(2,:))
ylim([-1 1]*1.5*(max(max(abs(y)))))
xlim([0 100])
ylabel('$y_2(t)$','Interpreter','Latex')
grid on

subplot(2,2,4)
plot(t,y(1,:))
ylim([-1 1]*1.5*(max(max(abs(y)))))
xlim([0 100])
ylabel('$y_1(t)$','Interpreter','Latex')
grid on

hf.Position(3) = hf.Position(3)*2;
end
