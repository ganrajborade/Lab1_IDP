#!/usr/bin/env python
# coding: utf-8

# In[1]:
#   1_(a).py

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


# In[2]:
#   1_(b).py

import numpy as np  
import matplotlib.pyplot as plt 
from pylab import *
t = np.linspace(0,4*np.pi,1000)
#print(t)
def Fourier_cal(N):
	y = 0	
	for n in range(1,N+1,1):
		y = y + (np.sin(n*t))/(n)
	return y
y1 = Fourier_cal(1)
y2 = Fourier_cal(5)
y3 = Fourier_cal(20)

subplot(3,1,1)
plt.plot(t,y1,'b',label='f2(t) when N=1')
plt.grid()
plt.xlabel('t')
plt.ylabel('f2(t) when N = 1')
plt.legend()

subplot(3,1,2)
plt.plot(t,y2,'r',label='f2(t) when N=5')
plt.grid()
plt.xlabel('t')
plt.ylabel('f2(t) when N = 5')
plt.legend()

subplot(3,1,3)
plt.plot(t,y3,'g',label='f2(t) when N=20')
plt.grid()
plt.xlabel('t')
plt.ylabel('f2(t) when N = 20')
plt.legend()
plt.show()


# In[3]:
#   1_(c).py

import numpy as np  
import matplotlib.pyplot as plt 
import cmath
from pylab import *
##FOR f1:
ak = [0]*101
ak_new = [0]*51
for k in range(0,101):
	if(k%2 == 0):
		ak[k] = 0
	else:
		ak[k] = 1/k
#Above represents the amplitude components of f1.
frequency = [0]*101
for i in range(0,101):
	frequency[i] = i/(2*np.pi)  #Represents the frequency harmonics of f1.
subplot(1,2,1)
for xc in frequency:
	j = np.int(xc*2*pi)
	plt.plot(xc,ak[j],'o')
	plt.vlines(xc, ymin=0, ymax=ak[j])#plotting the amplitude components v/s frequency harmonics.
xposition_new = [0]*51
for i in range(0,51):
	xposition_new[i] = (2*i+1)/(2*np.pi)
	ak_new[i] = 1/(2*i+1)
plt.plot(xposition_new,ak_new,'r',label='plotting curve joining amplitude components corresponding to odd frequency components')#plotting line curve joining amplitude components corresponding to odd frequency components.
plt.ylim(0,1.01)
plt.xlabel('frequency')
plt.ylabel('amplitude component for f1')
plt.legend()
plt.grid()
#########################################################################################

##FOR f2:

ak1 = [0]*101
for k1 in range(1,101):
	ak1[k1] = 1/k1
#Above represents the amplitude components of f2.
frequency1 = [0]*101
for i1 in range(0,101):
	frequency1[i1] = i1/(2*np.pi)  #Represents the frequency harmonics of f2.
subplot(1,2,2)
for xc1 in frequency1:
	j1 = np.int(xc1*2*pi)
	plt.plot(xc1,ak1[j1],'o')
	plt.vlines(xc1, ymin=0, ymax=ak1[j1])#plotting the amplitude components v/s frequency harmonics.
frequency1_new = [0]*100
ak1_new = [0]*100
for i1 in range(0,100):
	frequency1_new[i1] = (i1+1)/(2*np.pi)
	ak1_new[i1] = 1/(i1+1)
plt.plot(frequency1_new,ak1_new,'b',label='plotting curve joining amplitude components corresponding to frequency components')#plotting line curve joining amplitude components corresponding to odd frequency components.
plt.ylim(0,1.01)
plt.xlabel('frequency')
plt.ylabel('amplitude component for f2')
plt.legend()
plt.grid()
plt.show()


# In[4]:
#   1_(d).py

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
plt.plot(t,y2)
plt.xlabel('t')
plt.ylabel('Fourier approximation when N = 5')
plt.grid()

subplot(3,1,3)
plt.plot(t,y3)
plt.grid()
plt.xlabel('t')
plt.ylabel('Fourier approximation when N = 20')
plt.show()
##here we can modify the coefficients of cosine function  ((// for n in range(1,N,2):  y = y + (1/(n**2))*np.cos(n*t) return y//))   in the summation as 1/(n^2),then only we get the symmetric triangular function. 

# In[5]:
#  1_(e)_changing_DutyCycle_and_Amplitude_of_f1_code.py

import numpy as np  
import matplotlib.pyplot as plt 
from pylab import *
import cmath
t = np.linspace(-4*np.pi,4*np.pi,1000)

def Fourier_cal(N,A,D):
	y = 0	
	z = complex(0,1)
	for n in range(1,N,1):
		y = y + ((A/(np.pi*z*n))*(1-np.exp(-z*n*(2*np.pi*D))))*np.exp(z*n*t) #By changing the coefficient of the fourier series,we can easily modify the waveform according to Duty cycle and Amplitude.
	return y
y1 = np.sin(t)
y2 = Fourier_cal(1000,1,3/4)# A =1 , D = 0.75(in terms of fraction)
y3 = Fourier_cal(1000,2,3/4)# A =2 , D = 0.75(in terms of fraction)
y4 = Fourier_cal(1000,1,1/2)# A =1 ,D = 0.5(in terms of fraction)

subplot(4,1,1)
plt.plot(t,y1)
plt.xlabel('t')
plt.ylabel('sin(t)')
plt.grid()

subplot(4,1,2)
plt.plot(t,y2,'r',label='D = 0.75(in terms of fraction) and A =1 ')
plt.xlabel('t')
plt.ylabel('Fourier approximation when N = 1000')
plt.legend()
plt.grid()

subplot(4,1,3)
plt.plot(t,y3,'k',label='D = 0.75(in terms of fraction) and A =2 ')
plt.xlabel('t')
plt.ylabel('Fourier approximation when N = 1000')
plt.legend()
plt.grid()

subplot(4,1,4)
plt.plot(t,y4,'g',label='D = 0.5(in terms of fraction) and A =1 ')
plt.grid()
plt.xlabel('t')
plt.ylabel('Fourier approximation when N = 1000')
plt.legend()
plt.show()

#After running the above code,we observe that we can get desired duty cycle and amplitude of the square waveform by changing the fourier series coefficients of f1.Kindly please check the graph by running the above code :::: 1_(e)_changing_DutyCycle_and_Amplitude_of_f1_code.py 

####Similarly by changing the fourier coefficients for f2 as mentioned above , we can get the waveform according to the required need of Duty cycle and amplitude.




