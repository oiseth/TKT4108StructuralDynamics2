clc
clear all
close all
%% Correlation
stdX1 = 1;
stdX2 = 1;
rhoX1X2 = [-1 -0.7 -0.5 -0.2 0 0.2 0.7 0.9 1]; % Correlation coefficients
k = 2;
Covm = [stdX1^2 stdX1*stdX2*rhoX1X2(1,k); stdX1*stdX2*rhoX1X2(1,k) stdX2^2]; % Covaraiance matrix
fig1 = figure(1);
pos = 0;
for k = [1 2 8 9]
    Covm = [stdX1^2 stdX1*stdX2*rhoX1X2(1,k); stdX1*stdX2*rhoX1X2(1,k) stdX2^2];
    X = mvnrnd([0 0],Covm,1000);
    pos = pos + 1;
    subplot(4,1,pos)
    plot(1:size(X,1),X(:,1),'-o','MarkerSize',2,'DisplayName','X_1')
    hold on
    plot(1:size(X,1),X(:,2),'-o','MarkerSize',2,'DisplayName','X_2')
    ylabel('$X$','Interpreter','Latex')
    title(['\rho=' num2str(rhoX1X2(1,k))])
    xlim([0 50])
    ylim([-3 3])  
    grid on
end
legend show

fig2 = figure(2);
for k = 1:length(rhoX1X2)
    Covm = [stdX1^2 stdX1*stdX2*rhoX1X2(1,k); stdX1*stdX2*rhoX1X2(1,k) stdX2^2];
    X = mvnrnd([0 0],Covm,1000);
    subplot(3,3,k)
    plot(X(:,1),X(:,2),'o')
    axis([-1 1 -1 1]*3)
    title(['$\rho_{X_1X_2}= ' num2str(rhoX1X2(1,k)) '$'],'Interpreter','Latex')
    xlabel('$X_1$','Interpreter','Latex')
    ylabel('$X_2$','Interpreter','Latex')
    grid on
end
%% Auto correlation
dt = 0.1; % Time step
t = 1e-8:dt:100; % Time

tau = t-max(t)/2; % Tau 
omega_c = 2; % Cut-off frequency
R = 1./(omega_c*tau).*sin(omega_c*tau); % Auto-correlation function
figure
plot(tau,R)
ylim([-2 2])
xlabel('$\tau$','Interpreter','Latex')
ylabel('$R(\tau)$','Interpreter','Latex')
%% Simulate time series based on auto-correlation
tau_mat = abs(t'-t); % Matrix of all possible time delays (\tau values)
tau_mat(tau_mat==0) = eps; % Avoid the singularity when \tau = 0
R_mat = 1./(omega_c*tau_mat).*sin(omega_c*tau_mat); % Covariance matrix for all samples along the time axis
X = mvnrnd(zeros(1,length(t)),R_mat,4); % Draw samples of X
figure3 = figure(3);
% Plot realizations and auto-correlation function
% Auto-correlation function
subplot(5,1,1)
plot(tau,R)
ylim([-1.1 1.1])
xlim([-20 20])
grid on
xlabel('$\tau$','Interpreter','Latex')
ylabel('$R(\tau)$','Interpreter','Latex')
% Realizations of time series
for k = 1:4
    subplot(5,1,k+1)
    plot(t,X(k,:))
    ylim([-3 3])
    xlim([0 40])
    grid on
    ylabel('$X(t)$','Interpreter','Latex')
end
xlabel('$t$','Interpreter','Latex')
%% Make scatter plots at selected time lags 
taui = [0.5 1 2 4];
figure
subplot(3,2,[1 2])
plot(tau,R)
ylim([-1.1 1.1])
xlim([-5 5])
grid on
grid minor
title('Auto-correlation function')
ylabel('$R(\tau)$','Interpreter','latex')
for k = 1:length(taui)
    subplot(3,2,2+k)
    plot(X(:,1),X(:,round(taui(1,k)/dt+1)),'o')
    axis([-4 4 -4 4])
    grid on
    title(['$\tau = ' num2str(t(1,round(taui(1,k)/dt+1))) '$'],'Interpreter','latex')
    ylabel(['$X(t =' num2str(t(1,round(taui(1,k)/dt+1))) ')$'],'Interpreter','latex')
    xlabel(['$X(t =0' ')$'],'Interpreter','latex')
    
end



