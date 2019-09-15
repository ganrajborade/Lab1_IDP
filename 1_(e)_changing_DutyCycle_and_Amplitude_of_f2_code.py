import numpy as np  
import matplotlib.pyplot as plt 
from pylab import *
import cmath
t = np.linspace(-4*np.pi,4*np.pi,1000)

def Fourier_cal(N,A,D):
	y = 0	
	z = complex(0,1)
	for n in range(-N,N,1):
		if(n!=0):
			y = y + ((-A/((2*np.pi)**2))*(((4*np.pi*z)/n)+((((2*D)-1)/(D*(1-D)*(n**2)))*(1-np.exp(-z*2*np.pi*n*D)))))*np.exp(z*n*t) #By changing the coefficient of the fourier series,we can easily modify the waveform according to Duty cycle and Amplitude.i.e Cn = ((-A/((2*np.pi)**2))*(((4*np.pi*z)/n)+((((2*D)-1)/(D*(1-D)*(n**2)))*(1-np.exp(-z*2*np.pi*n*D)))))
	return y
y1 = np.sin(t)
y2 = Fourier_cal(2000,1,0.5)# A =1 , D = 0.5(in terms of fraction)
y3 = Fourier_cal(2000,2,0.57)# A =2 , D = 0.57(in terms of fraction)
y4 = Fourier_cal(2000,1,0.7)# A =1 ,D = 0.7(in terms of fraction)
y5 = Fourier_cal(2000,3,0.39)# A =3 ,D = 0.39(in terms of fraction)
subplot(5,1,1)
plt.plot(t,y1)
plt.xlabel('t')
plt.ylabel('sin(t)')
plt.grid()

subplot(5,1,2)
plt.plot(t,y2,'r',label='D = 0.5(in terms of fraction) and A =1,N = 2000  ')
plt.xlabel('t')
plt.ylabel('Fourier approx')
plt.legend()
plt.grid()
plt.ylim(-3.5,3.5)

subplot(5,1,3)
plt.plot(t,y3,'k',label='D = 0.57(in terms of fraction) and A =2,N = 2000  ')
plt.xlabel('t')
plt.ylabel('Fourier approx')
plt.legend()
plt.grid()
plt.ylim(-3.5,3.5)

subplot(5,1,4)
plt.plot(t,y4,'g',label='D = 0.7(in terms of fraction) and A =1,N = 2000  ')
plt.grid()
plt.xlabel('t')
plt.ylabel('Fourier approx')
plt.legend()
plt.ylim(-3.5,3.5)

subplot(5,1,5)
plt.plot(t,y5,'m',label='D = 0.39(in terms of fraction) and A =3 ,N = 2000 ')
plt.grid()
plt.xlabel('t')
plt.ylabel('Fourier approx')
plt.legend()
plt.ylim(-3.5,3.5)
plt.show()
