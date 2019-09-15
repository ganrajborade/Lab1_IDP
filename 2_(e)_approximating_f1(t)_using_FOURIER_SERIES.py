import numpy as np  
import matplotlib.pyplot as plt 
from pylab import *
import cmath
t = np.linspace(-2,2,1000)

def Fourier_cal(N):
	y = 0	
	z = complex(0,1)
	for k in range(1,N,2):
		y = y + (4/(np.pi*k))*np.sin(2*np.pi*k*t)
	return y
y1 = np.sin(2*np.pi*t)
y2 = Fourier_cal(20)
y3 = Fourier_cal(1000)

subplot(3,1,1)
plt.plot(t,y1)
plt.xlabel('t')
plt.ylabel('sin(t)')
plt.grid()

subplot(3,1,2)
plt.plot(t,y2,'r')
plt.xlabel('t')
plt.ylabel('Fourier approximation when N = 20')
plt.grid()

subplot(3,1,3)
plt.plot(t,y3,'g')
plt.grid()
plt.xlabel('t')
plt.ylabel('Fourier approximation when N = 1000')
plt.show()
