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
			y = y + ((A/(np.pi*z*n))*(1-np.exp(-z*n*(2*np.pi*D))))*np.exp(z*n*t) #By changing the coefficient of the fourier series,we can easily modify the waveform according to Duty cycle and Amplitude.i.e Cn = ((A/(np.pi*z*n))*(1-np.exp(-z*n*(2*np.pi*D))))
	return y
y1 = np.sin(t)
y2 = Fourier_cal(1000,1,3/4)# A =1 , D = 0.75(in terms of fraction)
y3 = Fourier_cal(1000,2,3/4)# A =2 , D = 0.75(in terms of fraction)
y4 = Fourier_cal(1000,1,1/2)# A =1 ,D = 0.5(in terms of fraction)
y5 = Fourier_cal(1000,3,0.4)# A =3 ,D = 0.4(in terms of fraction)

subplot(5,1,1)
plt.plot(t,y1)
plt.xlabel('t')
plt.ylabel('sin(t)')
plt.grid()



subplot(5,1,2)
plt.plot(t,y2,'r',label='D = 0.75(in terms of fraction) and A =1,N=1000 ')
plt.xlabel('t')
plt.ylabel('Fourier_approx')
plt.legend()
plt.grid()



subplot(5,1,3)
plt.plot(t,y3,'k',label='D = 0.75(in terms of fraction) and A =2,N=1000 ')
plt.xlabel('t')
plt.ylabel('Fourier_approx')
plt.legend()
plt.grid()


subplot(5,1,4)
plt.plot(t,y4,'g',label='D = 0.5(in terms of fraction) and A =1,N=1000 ')
plt.grid()
plt.xlabel('t')
plt.ylabel('Fourier_approx')
plt.legend()



subplot(5,1,5)
plt.plot(t,y5,'m',label='D = 0.4(in terms of fraction) and A =3,N=1000')
plt.grid()
plt.xlabel('t')
plt.ylabel('Fourier_approx')
plt.legend()


plt.show()

####Similarly by changing the fourier coefficients for f2(t) as mentioned above , we can get the waveform according to the required need of Duty cycle and amplitude.