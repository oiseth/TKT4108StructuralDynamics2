clc
clear vars
close all
%% Structural properties
omegaaxis=linspace(0,10,1000);
z=50;   %Height above ground,
z1=10;  %Reference height
L1=100;  %Reference integral length scale z1=10; L1=100;
xLu=L1*(z/z1)^0.3; % Integral length scale
Au=6.8/2/pi; % Constant in the auto-spectral density
V=40;   %Mean wind velocity
Iu=0.15;% Turbulence intensity
Cd=0.63; %Drag coefficient
rho=1.25; % Air density
D = 4.2; % Hight of the section
%% The mode shape
L=500;
y=linspace(0,L,100);
Phi=zeros(1,length(y));
nn=1;
for k=1:length(y)
    Phi(1,k)=sin(nn*pi*y(1,k)/L);
end
figure
plot(y,Phi)
xlabel('x')
ylabel('\Phi(x)')
%% The auto spectral density of the u-comp according to N400
Su=zeros(1,length(omegaaxis));
for k=1:length(omegaaxis)
    Su(1,k)=(Iu*V)^2*Au*xLu/V/((1+1.5*Au*omegaaxis(1,k)*xLu/V)^(5/3));
end
figure
subplot(1,2,1)
plot(omegaaxis,Su)
title('Auto spectral density u component')
ylabel('$S_{u}(\omega)$','Interpreter','Latex')
xlabel('$\omega$','Interpreter','Latex')
xlim([0 3])
subplot(1,2,2)
loglog(omegaaxis*xLu/V,omegaaxis.*Su/((Iu*V)^2))
title('Normalized auto spectral density u component')
ylabel('$\frac{\omega S_{u}(\omega)}{\sigma^2_{u}}$','Interpreter','Latex')
xlabel('$\omega$','Interpreter','Latex')
xlim([0.1 10])
%% Evaluate the integral
SQ=zeros(1,length(omegaaxis));
for k=1:length(omegaaxis)
    Int=zeros(length(y),length(y));
    for m=1:length(y)
        for n=1:length(y)
            dx=abs(y(1,n)-y(1,m));
            Suu=Su(1,k)*exp(-1.6*omegaaxis(1,k)*dx/V);
            Int(m,n)=(rho*V*Cd*D)^2*Phi(1,m)*Suu*Phi(1,n);
        end
    end
    SQ(1,k)=trapz(y,trapz(y,Int));
end
%% Plot generalized buffeting load
figure
plot(omegaaxis,SQ)
xlim([0 3])
title('Auto spectral density generalized buffeting load')
xlabel('$\omega$','Interpreter','latex')
ylabel('$S_{\tilde{Q}_{b}}(\omega)$','Interpreter','latex')

%% Plot integrand for one selected frequency

omega = 0.5;  % Frequency

Int=zeros(length(y),length(y));
for m=1:length(y)
    for n=1:length(y)
        dx=abs(y(1,n)-y(1,m));
        Suu=Su(1,k)*exp(-1.6*omega*dx/V);
        Int(m,n)=(rho*V*Cd*D)^2*Phi(1,m)*Suu*Phi(1,n);
    end
end

figure
surfl(y,y,Int)
xlabel('y_1')
ylabel('y_2')
title('$ (\rho V \bar{C}_D D)^2 \phi(y_1)S_{uu}(\omega,\Delta y)\phi(y_2)$','Interpreter','latex')
lighting phong
shading interp
colormap autumn


