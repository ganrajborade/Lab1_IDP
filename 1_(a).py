import numpy as np  
import matplotlib.pyplot as plt 
from pylab import *
t = np.linspace(0,4*np.pi,1000)

def Fourier_cal(N):
	y = 0	
	for n in range(1,N+1,2):
		y = y + (np.sin(n*t))/(n)
	return y
y1 = Fourier_cal(1)
y2 = Fourier_cal(5)
y3 = Fourier_cal(20)

subplot(3,1,1)
plt.plot(t,y1,'b',label='f1(t) when N=1')
plt.grid()
plt.xlabel('t')
plt.ylabel('f1 (t) when N = 1')
plt.legend()

subplot(3,1,2)
plt.plot(t,y2,'r',label='f1(t) when N=5')
plt.grid()
plt.xlabel('t')
plt.ylabel('f1(t) when N = 5')
plt.legend()

subplot(3,1,3)
plt.plot(t,y3,'g',label='f1(t) when N=20')
plt.grid()
plt.xlabel('t')
plt.ylabel('f1(t) when N = 20')
plt.legend()
plt.show()
