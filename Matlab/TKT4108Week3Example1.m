clc
clear all
close all
%% Define the dynamic system
% Mass, damping and stiffness
M=1;
C=0.1;
K=1;
%% Dynamic action
td=30;
t=linspace(0,150,5001);
p=zeros(1,length(t));
for n=1:length(t)
    if t(1,n)<td
        p(1,n)=t(1,n);
    else
        p(1,n)=0;
    end
end
%% Discrete Fourier transform of the input
Fs=1/(t(2)-t(1));
NFFT =length(p);
Gp=fftshift(fft(p,NFFT))/length(p);
f = Fs/2*linspace(-1,1,NFFT);
%% Complex frequency response, Chopra Eq. (A.1.7)
H=zeros(1,length(f));
for n=1:length(f)
    omega=2*pi*f(1,n);
    H(1,n)=1/(-omega^2*M+1i*omega*C+K);    
end
%% complex response vector, Chopra Eq. (A.2.8)
U=zeros(1,length(f));
for n=1:length(f)    
    U(1,n)=H(1,n)*Gp(1,n);    
end
%% response in time domain, Chopra Eq. (A.2.9)
U=ifftshift(U);
u=ifft(U)*length(p);
%% Plot results

figure(1)
subplot(3,2,1)
plot(t,p,'-b','DisplayName','Load')
legend show
ylabel('P(t)')
xlabel('t (s)')

subplot(3,2,2)
plot(2*pi*f,imag(Gp),'r','DisplayName','Imag')
hold on
plot(2*pi*f,real(Gp),'b','DisplayName','Real')
ylabel('P(\omega)')
xlabel('\omega (rad/s)')
title('Fourier coefficients')
xlim(2*pi*[-1 1])
legend show

subplot(3,2,4)
plot(2*pi*f,real(H),'b','DisplayName','Real')
hold on
plot(2*pi*f,imag(H),'r','DisplayName','Imag')
ylabel('H(\omega)')
xlabel('\omega (rad/s)')
title('Complex frequency response function')
xlim(2*pi*[-1 1])
legend show

subplot(3,2,6)
plot(2*pi*f,real(fftshift(U)),'b','DisplayName','Real')
hold on
plot(2*pi*f,imag(fftshift(U)),'r','DisplayName','Imag')
ylabel('U(\omega)=H(\omega)\cdotP(\omega)')
xlabel('\omega (rad/s)')
title('Fourier coefficients of response')
xlim(2*pi*[-1 1])
legend show

subplot(3,2,5)
plot(t,real(u(1,1:length(t))),'-b','DisplayName','Real part of response')
hold on
plot(t,imag(u(1,1:length(t))),'-r','DisplayName','Imaginary part of response, =0?')
plot(t,p/K,'-k','DisplayName','Quasi static response')
ylabel('u(t)')
xlabel('t (s)')
legend show
