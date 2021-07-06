clc
clear all
close all
%% Structural properties
m = 1000;
c = 2000;
EI = 2.1E11*3.36E-6;
L = 4;
k = 2*12*EI/L^3;
M = diag([1 1 1])*m;
K = [2 -1 0; -1 2 -1; 0 -1 1]*k;
C = diag([1 0.5 0])*c;
%% Calculate natural frequencies and damping ratios
s = polyeig(K,C,M);
f = abs(s)/2/pi;
zeta = -real(s)./abs(s);
figure
plot(f,zeta,'-o')
xlim([0 5])
ylim([0 0.05])
xlabel('$f$ (Hz)','Interpreter','latex')
ylabel('Damping ratio $\zeta$','Interpreter','latex')
grid on
%% Frequency response matrix
w = linspace(-60,60,4001);
H = zeros(3,3,length(w));
for k = 1:length(w)
    H(:,:,k) = inv(-w(1,k)^2*M+1i*w(1,k)*C+K);
end
figure
k = 0;
for n = 1:3
    for m = 1:3
        k = k+1;
        subplot(3,3,k)
        plot(w,real(squeeze(H(n,m,:))),'DisplayName','Re')
        hold on
        plot(w,imag(squeeze(H(n,m,:))),'DisplayName','Im')
        xlabel('$\omega$','Interpreter','latex')
        ylabel(['$H_{' num2str(n) num2str(m) '}(\omega)$'],'Interpreter','latex')
    end
end
legend show
%% Cross-spectral density matrix of loads
Sx = zeros(3,3,length(w));
Sx(1,1,:) = 1; Sx(1,1,w>30) = 0; Sx(1,1,w<-30) = 0;
Sx(2,2,:) = 1; Sx(2,2,w>30) = 0; Sx(2,2,w<-30) = 0;
Sx(3,3,:) = 1; Sx(3,3,w>30) = 0; Sx(3,3,w<-30) = 0;
Sx(1,2,:) = -0.5; Sx(1,2,w>30) = 0; Sx(1,2,w<-30) = 0;
Sx(1,3,:) = -0.25; Sx(1,3,w>30) = 0; Sx(1,3,w<-30) = 0;
Sx(2,1,:) = -0.5; Sx(2,1,w>30) = 0; Sx(2,1,w<-30) = 0;
Sx(2,3,:) = -0.5; Sx(2,3,w>30) = 0; Sx(2,3,w<-30) = 0;
Sx(3,1,:) = -0.25; Sx(3,1,w>30) = 0; Sx(3,1,w<-30) = 0;
Sx(3,2,:) = -0.5; Sx(3,2,w>30) = 0; Sx(3,2,w<-30) = 0;

Sx =Sx*10^3;

figure
k = 0;
for n = 1:3
    for m = 1:3
        k = k+1;
        subplot(3,3,k)
        plot(w,real(squeeze(Sx(n,m,:))),'DisplayName','Re')
        hold on
        plot(w,imag(squeeze(Sx(n,m,:))),'DisplayName','Im')
        xlabel('$\omega$','Interpreter','latex')
        ylabel(['$S_{x_{' num2str(n) num2str(m) '}}(\omega)$'],'Interpreter','latex')
        ylim([-1 1]*10^3)
    end
end
legend show

%% Cross-spectral density matrix of the response;
Sy = zeros(3,3,length(w));
for k = 1:length(w)
    Sy(:,:,k) = H(:,:,k)*Sx(:,:,k)*H(:,:,k)';
end

figure
k = 0;
for n = 1:3
    for m = 1:3
        k = k+1;
        subplot(3,3,k)
        plot(w,real(squeeze(Sy(n,m,:))),'DisplayName','Re')
        hold on
        plot(w,imag(squeeze(Sy(n,m,:))),'DisplayName','Im')
        xlabel('$\omega$','Interpreter','latex')
        ylabel(['$S_{y_{' num2str(n) num2str(m) '}}(\omega)$'],'Interpreter','latex')
    end
end
legend show

%% Calclate the probability density functions
mu_y1 = 0;
sigma_y1 = real(sqrt(trapz(w,squeeze(Sy(1,1,:)))));
mu_y2 = 0;
sigma_y2 = real(sqrt(trapz(w,squeeze(Sy(2,2,:)))));
mu_y3 = 0;
sigma_y3 = real(sqrt(trapz(w,squeeze(Sy(3,3,:)))));

pd_y1 = makedist('Normal',mu_y1,sigma_y1);
pd_y2 = makedist('Normal',mu_y2,sigma_y2);
pd_y3 = makedist('Normal',mu_y3,sigma_y3);
yp = linspace(-0.02,0.02,1000);
pdf_y1 = pdf(pd_y1,yp);
pdf_y2 = pdf(pd_y2,yp);
pdf_y3 = pdf(pd_y3,yp);

figure
subplot(2,3,1)
plot(yp,pdf_y1)
xlabel('$y_1$','Interpreter','latex')

subplot(2,3,2)
plot(yp,pdf_y2)
xlabel('$y_2$','Interpreter','latex')

subplot(2,3,3)
plot(yp,pdf_y3)
axis
xlabel('$y_3$','Interpreter','latex')

%% Calculate the cross-spectral density of the load effects
SF = zeros(2,2,length(w));
T = [-12*EI/L^3 12*EI/L^3 0; -6*EI/L^2 6*EI/L^2 0];
for k = 1:length(w)
    SF(:,:,k) = T*Sy(:,:,k)*T.';
end

figure
k = 0;
for n = 1:2
    for m = 1:2
        k = k+1;
        subplot(2,2,k)
        plot(w,real(squeeze(SF(n,m,:))),'DisplayName','Re')
        hold on
        plot(w,imag(squeeze(SF(n,m,:))),'DisplayName','Im')
        xlabel('$\omega$','Interpreter','latex')
        ylabel(['$S_{F_{' num2str(n) num2str(m) '}}(\omega)$'],'Interpreter','latex')
    end
end
legend show

%% Calculate the probability density function of the shear force and bending moment

mu_V = 0;
sigma_V = real(sqrt(trapz(w,squeeze(SF(1,1,:)))));
mu_M = 0;
sigma_M = real(sqrt(trapz(w,squeeze(SF(2,2,:)))));

pd_V = makedist('Normal',mu_V,sigma_V);
pd_M = makedist('Normal',mu_M,sigma_M);
yp = linspace(-20000,20000,1000);
pdf_V = pdf(pd_V,yp);
pdf_M = pdf(pd_M,yp);

figure
subplot(2,2,1)
plot(yp,pdf_V)
xlabel('$V$','Interpreter','latex')
xlim([-2 2]*10^3)

subplot(2,2,2)
plot(yp,pdf_M)
xlabel('$M$','Interpreter','latex')
xlim([-1 1]*10^4)

% Correlation coefficient
cov_VM = real(trapz(w,squeeze(SF(1,2,:))));
rho_VM = cov_VM/(sigma_V*sigma_M);



return
%% Make time series of the loads
Sx = 2*Sx(:,:,w>0);
[XX] = MCCholesky(w(w>0),Sx);
t = XX{1,1};
t = t(t<1000);
X = XX{1,2};
X = X(:,t<1000);

%% Calculate response
y0 = [0 0 0]';
ydot0 = [0 0 0]';
h = t(2)-t(1);
gamma = 1/2;
beta = 1/4;
y = linear_newmark_krenk(M,C,K,X,y0,ydot0,h,gamma,beta);

figure
for k = 1:3
    subplot(3,1,k)
    plot(t,y(k,:))
end

% Calculate the load effects

F = T*y;

corr(F')





%% Monte Carlo Simulation of loads
function [XX]=MCCholesky(omegaaxisinput,SS,varargin)
%% Description
% This function simulates realizations (time series) using the
% cross-spectral density matrix SS(Ndof, Ndof, Nomegaaxisinput) as basis. To improve the performance the
% factorized cross spectral density is interpolated to a denser frequency
% axis befor the time series are generated by means of ifft.

% Ole Øiseth 25.10.2014
%% Define parameters
p=inputParser;
addParameter(p,'Nsim',1,@isnumeric);                     %Number of time series
addParameter(p,'domegasim',0.0001,@isnumeric);           %delta omega used in the simuilations
addParameter(p,'FileName','No',@ischar);                  %File Name. If no file name is given, the data will be exported as a cell array.
parse(p,varargin{:})
Nsim = p.Results.Nsim;
domegasim = p.Results.domegasim;
FileName = p.Results.FileName;
%% Cholesky decomposition of the cross spectral density matrix
GG=zeros(size(SS));
for n=1:length(omegaaxisinput)
    if max(max(abs(SS(:,:,n))))<1.0e-10
    else
        GG(:,:,n)=chol(SS(:,:,n),'lower');
    end
end
%% Simulate time series
omegaaxissim=domegasim:domegasim:max(omegaaxisinput);
NFFT=2^nextpow2(2*length(omegaaxissim));
t=linspace(0,2*pi/domegasim,NFFT);
XX=cell(1,Nsim+1);
if strcmp(FileName,'No');  XX{1,1}=t; end
for z=1:Nsim
    phi=2*pi*rand(size(SS,1),length(omegaaxissim));
    x=zeros(size(SS,1),NFFT);
    for m=1:size(GG,1)
        for n=1:m
            c=interp1(omegaaxisinput,permute((GG(m,n,:)),[1,3,2]),omegaaxissim,'linear','extrap').*exp(1i.*phi(n,:));
            x(m,:)=x(m,:)+real(ifft(c,NFFT))*NFFT*sqrt(2*domegasim);
        end
    end
    if strcmp(FileName,'No'); XX{1,z+1}=x; else save([FileName '_Nr_' num2str(z) '.mat'],'t','x'); end
end

end

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
