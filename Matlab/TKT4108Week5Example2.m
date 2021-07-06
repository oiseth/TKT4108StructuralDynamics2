clc
clear all
close all
%% Solve double integral
x = linspace(0,10,100);
Integrand = zeros(length(x),length(x));
for n = 1:length(x)
    for m = 1:length(x)
        Integrand(n,m) = x(1,n)*x(1,m)*exp(-abs(x(1,n)-x(1,m)));
    end
end
figure
surfl(x,x,Integrand)
shading interp
ylabel('$x$','Interpreter','latex')
xlabel('$x$','Interpreter','latex')
%% Obtain auto correlation function
tau = linspace(0,10,1000);
R = exp(-tau)*trapz(x,trapz(x,Integrand));

figure
plot(tau,R)
ylabel('$R(\tau)$','Interpreter','latex')
xlabel('$\tau$','Interpreter','latex')

