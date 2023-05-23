import pywt
import pywt.data
import matplotlib.pyplot as plt
import cv2
import numpy as np


def flat_wavelet_transform2(x, method='bior1.3'):
    """For a 2D image x, take the wavelet """
    coeffs = pywt.wavedec2( x, method )
    output = coeffs[0].reshape(-1)
    for tups in coeffs[1:]:
        for c in tups:
            output = np.concatenate((output, c.reshape(-1)))
    return output

def inverse_flat_wavelet_transform2(X,  shape, method='bior1.3'):
    shapes = pywt.wavedecn_shapes( shape , method)
    nx = shapes[0][0]
    ny = shapes[0][1]
    n = nx * ny
    coeffs = [X[:n].reshape(nx,ny) ]
    for i, d in enumerate(shapes[1:]):
        vals=list(d.values())
        nx = vals[0][0]
        ny = vals[0][1]
        coeffs.append( (X[ n : n + nx * ny].reshape( nx, ny ), 
                        X[ n + nx * ny : n + 2 * nx * ny ].reshape( nx, ny ), 
                        X[ n + 2 * nx * ny : n + 3 * nx * ny ].reshape( nx, ny ))  )
        n += 3 * nx * ny
    return pywt.waverec2(coeffs, method)




# Wavelet transform of image, and plot approximation and details
titles = ['Approximation', ' Horizontal detail',
          'Vertical detail', 'Diagonal detail']

image = cv2.imread('lena.png')
image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
image = cv2.resize(image, dsize=(512,512),interpolation=cv2.INTER_CUBIC)

# Convert to float for more resolution for use with pywt
image = np.float32(image)

plt.figure()
plt.imshow(image, cmap='gray')
plt.show()

coeffs2 = pywt.dwt2(image, 'bior1.3')
LL, (LH, HL, HH) = coeffs2
fig = plt.figure(figsize=(10, 10*4))
for i, a in enumerate([LL, LH, HL, HH]):
    ax = fig.add_subplot(1, 4, i + 1)
    ax.imshow(a, interpolation="nearest", cmap=plt.cm.gray)
    ax.set_title(titles[i], fontsize=10)
    ax.set_xticks([])
    ax.set_yticks([])

fig.tight_layout()
plt.show()


undersample_rate = .5
n = image.shape[0] * image.shape[1]

original_undersampled = ( image.reshape(-1) \
    * np.random.permutation( 
        np.concatenate( 
            (np.ones( int( n * undersample_rate ) ), 
             np.zeros( int( n * ( 1-undersample_rate )) )) 
        ) 
    ) 
                        ).reshape(512,512)



fig,ax = plt.subplots(1,2,figsize=(20,10))
ax[0].imshow(image, cmap=plt.cm.gray)
ax[1].imshow(original_undersampled,cmap=plt.cm.gray)
plt.show()


plt.figure(figsize=(15,10))
plt.plot(flat_wavelet_transform2(original_undersampled, 'bior1.3')[-100:] )
plt.title('A small sample of the wavelet transform')
plt.show()



plt.figure(figsize=(15,10))
plt.hist(flat_wavelet_transform2(original_undersampled, 'bior1.3') , range=(-300,300), bins=30)
plt.title('A small sample of the wavelet transform')
plt.show()





methods = ['haar','coif1','coif2','coif3','bior1.1','bior1.3','bior3.1','bior3.3','rbio1.1','rbio1.3','rbio3.1','rbio3.3']
def distance(x,y):
    return sum(abs(x.reshape(-1)-y.reshape(-1)))

def soft_thresh(x, lam):
    if ~(isinstance(x[0], complex)):
        return np.zeros(x.shape) + (x + lam) * (x<-lam) + (x - lam) * (x>lam) 
    else:
        return np.zeros(x.shape) + ( abs(x) - lam ) / abs(x) * x * (abs(x)>lam) 

# undersampled noisy signal in image-space and let this be first order Xhat
y = original_undersampled

# Repeat steps 1-4 until change is below a threshold
eps = 1e-2
lam = 100
lam_decay = 0.995
minlam = 1

err2=[]


lam = 100


xhat = y.copy()
for i in range(80):
    method = 'sym3'
    xhat_old = xhat
    Xhat_old = flat_wavelet_transform2(xhat, method)
    Xhat = soft_thresh(Xhat_old, lam)
    xhat = inverse_flat_wavelet_transform2(Xhat, (512,512), method)
    xhat[y!=0] = y[y!=0]   


    xhat = xhat.astype(int)
    xhat[xhat<0] = 0
    xhat[xhat>255] = 255
    err2.append(distance(image, xhat))
    lam *= lam_decay 
#     if (distance(xhat, xhat_old)<eps):
#         break



    
fig = plt.figure(figsize=(10,10))  
plt.loglog(err2)


fig,ax = plt.subplots(1,2,figsize=(20,10))
ax[0].imshow(image, cmap=plt.cm.gray)
ax[1].imshow(xhat,cmap=plt.cm.gray, vmin=0, vmax=255)
plt.show()
