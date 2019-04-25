import numpy as np
from scipy.signal import get_window
from scipy.fftpack import fft, fftshift
import math
import matplotlib.pyplot as plt
eps = np.finfo(float).eps


M =511
N= M
w = get_window('blackmanharris', M)         # get the window 
### Your code here
hM1 = int(math.floor(M+1)/2)
hM2 = int(math.floor(M/2))

'''
N = 512
hN = int(N/2)
fftbuffer = np.zeros(N)
fftbuffer[:hM1] = w[hM2:]
fftbuffer[N-hM2:] = w[:hM2]
'''

X = fft(w)
absX = abs(X)
#absX[absX<np.finfo(float).eps] = np.finfo(float).eps
#mX = 20*np.log10(absX)



mX1 = np.zeros(N)

#mX1[:hN] = mX[hN:]
#mX1[N-hN:] = mX[:hN]
#mX1[:hM1] = mX[hM2:]
#mX1[N-hM2:] = mX[:hM2]

plt.plot(X)
#plt.plot(np.arange(-hN,hN)/float(N)*M,mX1-max(mX1))
#plt.axis([-20,20,-80,0])
plt.show()
