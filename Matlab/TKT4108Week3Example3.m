clc
clear all
close all
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
%% Cross sepectra density matrix of loads.
omegaaxis = linspace(-50,50,1000);
SQ = zeros(2,2,length(omegaaxis));

SQ(1,1,:) = 1; SQ(1,1,omegaaxis>25) = 0; SQ(1,1,omegaaxis<-25) = 0;
SQ(2,2,:) = 1; SQ(2,2,omegaaxis>25) = 0; SQ(2,2,omegaaxis<-25) = 0;
SQ(1,2,:) = -0.5; SQ(1,2,omegaaxis>25) = 0; SQ(1,2,omegaaxis<-25) = 0;
SQ(2,1,:) = -0.5; SQ(2,1,omegaaxis>25) = 0; SQ(2,1,omegaaxis<-25) = 0;

figure
subplot(2,2,1)
plot(omegaaxis,real(squeeze(SQ(1,1,:))))
hold on
plot(omegaaxis,imag(squeeze(SQ(1,1,:))))
ylim([0 2])
ylabel('S_Q(\omega)')
xlabel('\omega')
%
subplot(2,2,2)
plot(omegaaxis,real(squeeze(SQ(1,2,:))))
hold on
plot(omegaaxis,imag(squeeze(SQ(1,2,:))))
ylim([-2 2])
ylabel('S_Q(\omega)')
xlabel('\omega')
%
subplot(2,2,3)
plot(omegaaxis,real(squeeze(SQ(2,1,:))))
hold on
plot(omegaaxis,imag(squeeze(SQ(2,1,:))))
ylim([-2 2])
ylabel('S_Q(\omega)')
xlabel('\omega')
%
subplot(2,2,4)
plot(omegaaxis,real(squeeze(SQ(2,2,:))))
hold on
plot(omegaaxis,imag(squeeze(SQ(2,2,:))))
ylim([0 2])
ylabel('S_Q(\omega)')
xlabel('\omega')
%% Frequency response matrix
H = zeros(2,2,length(omegaaxis));

for k = 1:length(omegaaxis)
    H(:,:,k) = inv(-omegaaxis(1,k)^2*MM + 1i*omegaaxis(1,k)*CC + KK);
    
end

figure
subplot(2,2,1)
plot(omegaaxis,real(squeeze(H(1,1,:))))
hold on
plot(omegaaxis,imag(squeeze(H(1,1,:))))
ylabel('H(\omega)')
xlabel('\omega')
%
subplot(2,2,2)
plot(omegaaxis,real(squeeze(H(1,2,:))))
hold on
plot(omegaaxis,imag(squeeze(H(1,2,:))))
ylabel('H(\omega)')
xlabel('\omega')
%
subplot(2,2,3)
plot(omegaaxis,real(squeeze(H(2,1,:))))
hold on
plot(omegaaxis,imag(squeeze(H(2,1,:))))
ylabel('H(\omega)')
xlabel('\omega')
%
subplot(2,2,4)
plot(omegaaxis,real(squeeze(H(2,2,:))))
hold on
plot(omegaaxis,imag(squeeze(H(2,2,:))))
ylabel('H(\omega)')
xlabel('\omega')

%% Cross-spectral density matrix of response
Sy = zeros(2,2,length(omegaaxis));
for k = 1:length(omegaaxis)
    Sy(:,:,k) = H(:,:,k)*SQ(:,:,k)*H(:,:,k)';
end


figure
subplot(2,2,1)
plot(omegaaxis,real(squeeze(Sy(1,1,:))))
hold on
plot(omegaaxis,imag(squeeze(Sy(1,1,:))))
ylabel('S_y(\omega)')
xlabel('\omega')
%
subplot(2,2,2)
plot(omegaaxis,real(squeeze(Sy(1,2,:))))
hold on
plot(omegaaxis,imag(squeeze(Sy(1,2,:))))
ylabel('S_y(\omega)')
xlabel('\omega')
%
subplot(2,2,3)
plot(omegaaxis,real(squeeze(Sy(2,1,:))))
hold on
plot(omegaaxis,imag(squeeze(Sy(2,1,:))))
ylabel('S_y(\omega)')
xlabel('\omega')
%
subplot(2,2,4)
plot(omegaaxis,real(squeeze(Sy(2,2,:))))
hold on
plot(omegaaxis,imag(squeeze(Sy(2,2,:))))
ylabel('S_y(\omega)')
xlabel('\omega')

  