import numpy as np  
import matplotlib.pyplot as plt 
from pylab import *
import cmath

w, h = 10, 8;
Matrix = [[0 for x in range(w)] for y in range(h)] 
output1 = [[0 for x in range(w)] for y in range(h)] 
output2 = [[0 for x in range(w)] for y in range(h)]
for i in range(h):
	for j in range(w):
		Matrix[i][j] = ((i/2) -2) + j/18
		if(i%2 == 0):
			output1[i][j] = 1
			output2[i][j] = Matrix[i][0] + 1/2 - Matrix[i][j]
		else:
			output1[i][j] = -1
			output2[i][j] = Matrix[i][j] - Matrix[i][0]

subplot(1,2,1)
for i in range(8):
	plt.plot(Matrix[i],output1[i],'b')
List = [-3/2,-1,-1/2,0,1/2,1,3/2]
for i in range(7):
	plt.vlines(x=List[i], ymin=-1, ymax=1.0, color='b')
plt.hlines(y=0,xmin=-2,xmax=2,color='k')
plt.vlines(x=0,ymin=-2,ymax=2,color='k')
plt.text(2,0.004,'x-axis',fontsize=18)
plt.text(0.09,2,'y-axis',fontsize=18)

plt.grid()
plt.xlabel('t')
plt.ylabel('f1(t)')
subplot(1,2,2)
for i in range(8):
	plt.plot(Matrix[i],output2[i],'r')
plt.hlines(y=0,xmin=-2,xmax=2,color='k')
plt.vlines(x=0,ymin=-2,ymax=2,color='k')
plt.text(2,0.004,'x-axis',fontsize=18)
plt.text(0.09,2,'y-axis',fontsize=18)

plt.grid()
plt.xlabel('t')
plt.ylabel('f2(t)')
plt.show()