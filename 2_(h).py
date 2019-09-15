import numpy as np  
import matplotlib.pyplot as plt 
import cmath
from pylab import *
##FOR f1:
ak = [0]*16
for k in range(0,16):
	if(k%2 == 0):
		ak[k] = 0
	else:
		ak[k] = 4/(np.pi*k)
#Above represents the amplitude components of f1(t).
frequency = [0]*16
for i in range(0,16):
	frequency[i] = i  #Represents the frequency harmonics of f1(t).
subplot(1,2,1)
for xc in frequency:
	plt.plot(xc,ak[xc],'o')
	plt.vlines(xc, ymin=0, ymax=ak[xc])#plotting the amplitude components v/s frequency harmonics.


plt.xlabel('frequency')
plt.ylabel('amplitude component for f1(t)')
plt.legend()
plt.grid()
# #########################################################################################

##FOR f2:

ak1 = [0]*16
for k in range(0,16):
	if(k%2 == 0):
		ak1[k] = 0
	else:
		ak1[k] = 2/(((np.pi)**2)*(k**2))
ak1[0] = 1/4
#Above represents the amplitude components of f2(t).
frequency1 = [0]*16
for i1 in range(0,16):
	frequency1[i1] = i1 #Represents the frequency harmonics of f2(t).
subplot(1,2,2)
for xc1 in frequency1:
	plt.plot(xc1,ak1[xc1],'o')
	plt.vlines(xc1, ymin=0, ymax=ak1[xc1])#plotting the amplitude components v/s frequency harmonics.

plt.xlabel('frequency')
plt.ylabel('amplitude component for f2(t)')
plt.legend()
plt.grid()
plt.show()
