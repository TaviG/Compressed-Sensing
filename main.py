import numpy as np
from scipy.fft import fft, fftshift, ifft, ifftshift
import matplotlib.pyplot as plt



def soft_thresh(x, lam):
    if ~(isinstance(x[0], complex)):
        return np.zeros(x.shape) + (x + lam) * (x<-lam) + (x - lam) * (x>lam) 
    else:
        return np.zeros(x.shape) + ( abs(x) - lam ) / abs(x) * x * (abs(x)>lam) 

def fftc(x):
    """Computes the centered Fourier transform"""
    return fftshift( fft(x) )

def ifftc(X):
    """Inverses the centered Fourier transform"""
    return ifft( ifftshift(X) )



l = 96
n = 7
sigma = 0.05
np.random.seed(24)

#generate sparse signal
x = np.concatenate( (np.ones(n) / n , np.zeros(l-n)) , axis=0 )
x = np.random.permutation(x)
# add random noise
y = x + sigma * np.random.randn(l)

plt.figure()
plt.plot(x)
plt.show()

plt.figure()
plt.plot(y)
plt.show()


X = fftc(x)
Y = fftc(y)

plt.figure()
plt.plot(abs(X))
plt.show()

plt.figure()
plt.plot(abs(Y))
plt.show()



#uniformly sampled k-space
Xu = 4 * X
for i in range(1,4):
    Xu[i::4] = 0
#reconstructed signal
xu = ifftc(Xu)

#randomly sampled k-space
Xr = 4 * X * np.random.permutation(np.repeat([1,0,0,0], l/4) )
xr = ifftc( Xr )


# undersampled noisy signal in k-space and let this be first order Xhat
Y = 4 * fftc(x) * np.random.permutation(np.repeat([1,0,0,0], l/4) )
Xhat = Y.copy()


# Repeat steps 1-4 until change is below a threshold
eps = 1e-4
lam = 0.05

def distance(x,y):
    return abs(sum(x-y))
diff=[]
err = []
itermax = 10000
while True:
    itermax -= 1
    xhat_old = ifftc(Xhat)
    xhat = soft_thresh(xhat_old, lam)
    diff.append(distance(xhat, xhat_old))
    err.append(distance(xhat.real/4,x))
    if ( diff[-1] < eps ) | ( itermax == 0 ):
        break
    Xhat = fftc(xhat)
    Xhat[Y!=0] = Y[Y!=0]


plt.figure()
plt.plot(xhat.real/4)
plt.show()

plt.figure()
plt.plot(err)
plt.show()