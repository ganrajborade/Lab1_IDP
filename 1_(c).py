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
#Above represents the amplitude components of f1(t).
frequency = [0]*101
for i in range(0,101):
	frequency[i] = i/(2*np.pi)  #Represents the frequency harmonics of f(t).
subplot(1,2,1)
for xc in frequency:
	j = np.int(xc*2*pi)
	plt.plot(xc,ak[j],'o')
	plt.vlines(xc, ymin=0, ymax=ak[j])#plotting the amplitude components v/s frequency harmonics.

plt.ylim(0,1.01)
plt.xlabel('frequency')
plt.ylabel('amplitude component for f1(t)')
plt.legend()
plt.grid()
#########################################################################################

##FOR f2:

ak1 = [0]*101
for k1 in range(1,101):
	ak1[k1] = 1/k1
#Above represents the amplitude components of f2(t)
frequency1 = [0]*101
for i1 in range(0,101):
	frequency1[i1] = i1/(2*np.pi)  #Represents the frequency harmonics of f2(t)
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
plt.plot(frequency1_new,ak1_new,'b',label='plotting curve joining amplitude components corresponding to frequency components')#plotting line curve joining amplitude components corresponding to all frequency components.
plt.ylim(0,1.01)
plt.xlabel('frequency')
plt.ylabel('amplitude component for f2(t)')
plt.legend()
plt.grid()
plt.show()
