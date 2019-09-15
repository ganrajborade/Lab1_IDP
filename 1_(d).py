import numpy as np  
import matplotlib.pyplot as plt 
from pylab import *
import cmath
t = np.linspace(-50,50,1000)

def Fourier_cal(N):
	y = 0	
	z = complex(0,1)
	for n in range(1,N,2):
		y = y + (1/(n**2))*np.cos(n*t)
	return y
y1 = np.cos(t)
y2 = Fourier_cal(5)
y3 = Fourier_cal(20)

subplot(3,1,1)
plt.plot(t,y1)
plt.xlabel('t')
plt.ylabel('sin(t)')
plt.grid()

subplot(3,1,2)
plt.plot(t,y2,'r')
plt.xlabel('t')
plt.ylabel('Fourier approximation when N = 5')
plt.grid()

subplot(3,1,3)
plt.plot(t,y3,'g')
plt.grid()
plt.xlabel('t')
plt.ylabel('Fourier approximation when N = 20')
plt.show()
##here we can modify the coefficients of cosine function  (( for n in range(1,N,2):  y = y + (1/(n**2))*np.cos(n*t) return y))   in the summation as 1/(n^2),then only we get the symmetric triangular function. 
