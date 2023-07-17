# -*- coding: utf-8 -*-
"""
Created on Tue Jun 27 10:08:57 2023

@author: Alexander
"""
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import scipy.cluster.vq as scv
#from image_mut_class import imageCloneMutator
from matplotlib.ticker import LinearLocator
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression, Ridge
from functools import lru_cache

#%% IMAGE UTILS

def nan_interpolate(arr_3d:np.ndarray):
    data=arr_3d.copy()
    nanIndex = np.isnan(data).nonzero()

    interpolatedData = data.copy()
    # make an initial guess for the interpolated data using the mean of the non NaN values
    interpolatedData[nanIndex] = np.nanmean(data)


    def sign(x):
        """returns the sign of the neighbor to be averaged for boundary elements"""
        if x == 0:
            return [1, 1]
        elif x == 4-1:
            return [-1, -1]
        else:
            return [-1, 1]

    #calculate kernels for the averages on boundaries/non boundary elements
    for i in range(len(nanIndex)):
        nanIndex = *nanIndex, np.array([sign(x) for x in nanIndex[i]])

    # gauss seidel iteration to interpolate Nan values with neighbors
    # https://en.wikipedia.org/wiki/Gauss%E2%80%93Seidel_method
    for _ in range(100):
        for x, y, z, dx, dy, dz in zip(*nanIndex):
            interpolatedData[x, y, z] = (
                (interpolatedData[x+dx[0], y, z] + interpolatedData[x+dx[1], y, z] +
                 interpolatedData[x, y+dy[0], z] + interpolatedData[x, y+dy[1], z] +
                 interpolatedData[x, y, z+dz[0]] + interpolatedData[x, y, z+dz[1]]) / 6)
    return interpolatedData



def norm(arr:np.ndarray):
    return (arr-np.min(arr))/(np.max(arr)-np.min(arr))

def expand_channels(condition2d,channels=4,data_type=int): 
    return np.asarray(np.concatenate([np.expand_dims(condition2d,axis=2) for _ in range(channels)],axis=2),dtype=data_type)

def mse(im1, im2): 
    return np.square(np.subtract(im1,im2)).mean()

def colormap2arr(arr:np.ndarray,cmap=cm.bwr,resolution:int=256):    
    # http://stackoverflow.com/questions/3720840/how-to-reverse-color-map-image-to-scalar-values/3722674#3722674
    gradient=cmap(np.linspace(0.0,1.0,resolution))

    # Reshape arr to something like (240*240, 4), all the 4-tuples in a long list...
    arr2=np.copy(arr)
    arrflat=arr2.reshape((arr2.shape[0]*arr.shape[1],arr2.shape[2]))

    # Use vector quantization to shift the values in arr2 to the nearest point in
    # the code book (gradient).
    code,dist=scv.vq(arrflat,gradient)

    # code is an array of length arr2 (240*240), holding the code book index for
    # each observation. (arr2 are the "observations".)
    # Scale the values so they are from 0 to 1.
    values=code.astype('float')/gradient.shape[0]

    # Reshape values back to (240,240)
    values=values.reshape(arr2.shape[0],arr2.shape[1])
    values=values[::-1]
    return values

def arr2colormap(arr:np.ndarray,cmap=cm.bwr,resolution:int=256):    
    colorfunc=np.vectorize(lambda x: cmap(x))
    return np.flip(np.flip(np.swapaxes(np.asarray(colorfunc(arr)),0,2),0),1)



#@lru_cache(maxsize=None)

def color2strain(mask_color:np.ndarray,color:np.ndarray,cmap=cm.bwr,deg:int=20,al:int=1,amp:float=.1):
    field  = mask2field(np.copy(mask_color),np.copy(color),cmap=cmap)
    scalar = colormap2arr(field)
    scalar_new = noise2d(scalar,amp)
    fitted,_ = fitPlaneFunc(scalar_new,deg=deg,al=al)
    condition = lambda n1, n2, nb: (
        np.abs(sum(n1 - nb))**2 > np.abs(sum(n2 - nb))**2)[0]
    test, thresh = True, 0
    while test:
        n1=flattenMidBandNoise(fitted, thresh)[::-1] 
        n2=flattenMidBandNoise(fitted, thresh+.01)[::-1]
        test = condition(n1,n2,scalar)
        print(test)
        thresh += .01
    print(thresh)
    return n1


def mask2field(arr3d_mask:np.ndarray,arr3d_cbase:np.ndarray,cmap=cm.bwr):
    '''
    

    Parameters
    ----------
    arr3d_mask : np.ndarray
        DESCRIPTION.
    arr3d_cbase : np.ndarray
        DESCRIPTION.
    cmap : TYPE, optional
        DESCRIPTION. The default is cm.bwr.

    Returns
    -------
    color_field : np.ndarray
        DESCRIPTION.

    '''
    color_only=arr3d_mask*arr3d_cbase
    color_only[color_only[:,:,-1]==0]=cmap(0.5)
    color_field=color_only
    return color_field
    

def fitPlaneFunc(arr:np.ndarray,xr:tuple=(0,1),yr:tuple=(0,1),deg:int=10,al:int=1):
    '''
    

    Parameters
    ----------
    arr : np.ndarray
        Provide a 2d Scalar array.
    xr : tuple, optional
        Range of x values. The default is (0,1).
    yr : tuple, optional
        Range of y values. The default is (0,1).
    deg : int, optional
        Highest polynomial degree of the 2d fitting function. The default is 10.
    al : int, optional
        Value of alpha for the ridge regression. The default is 1.

    Returns
    -------
    arr_poly_fit : np.ndarray
        Return degree {deg} 2D polynomial fit at the original points and resolution given.
    model : TYPE
        return the fitted model.

    '''
    # Create list of point in domain
    arr_shape = np.asarray(arr).shape
    X = np.linspace(xr[0],xr[1],num=arr_shape[0])
    Y = np.linspace(yr[0],yr[1],num=arr_shape[1])
    X,Y = np.meshgrid(X,Y)

    # Process 2D inputs
    poly = PolynomialFeatures(degree=deg)
    input_pts = np.stack([X.flatten(), Y.flatten()]).T
    assert(input_pts.shape == (arr.shape[0]*arr.shape[1], 2))
    in_features = poly.fit_transform(input_pts)
    
    # Linear regression
    model = Ridge(alpha=al)
    model.fit(in_features, arr.flatten())

    # Make predictions
    arr_poly_fit = np.reshape(model.predict(poly.transform(input_pts)),arr_shape)   
    return arr_poly_fit, model



def flattenMidBandNoise(arr: np.ndarray, thresh: float = .15):
    '''
    

    Parameters
    ----------
    arr : np.ndarray
        Provide a 2d Scalar array.
    thresh : float, optional
        Set values with an absolute distance of thresh % from mid values 
        to 50% . The default is 0.15 .

    Returns
    -------
    arr_thresh : np.ndarray
        Returns scalar array normalized on a bound between 0 and 1 with values
        that have an absolute distance of thresh % from mid values 
        to the mid values.

    '''
    # Prevent modification of image in palce
    arr_thresh=np.copy(arr)
    
    # Normalize the image
    arr_thresh -= np.min(arr_thresh)
    arr_thresh /= np.max(arr_thresh)
    
    # Collapse mid-band noise   
    arr_thresh[abs(arr_thresh-.5)<=thresh]=.5
    
    return arr_thresh

def noise2d(arr:np.ndarray,amp:float=.05):
    '''
    

    Parameters
    ----------
    arr : np.ndarray
        2D array to add noise to.
    amp : float, optional
        DESCRIPTION. The default is .05.

    Returns
    -------
    TYPE
        DESCRIPTION.

    '''
    return amp*np.random.rand(*arr.shape)*abs(np.max(arr)-np.min(arr))+np.copy(arr)



#%%  Plotter
def plotSurf(arr:np.ndarray,xr:tuple=(0,1),yr:tuple=(0,1),figs=(4,4),render_null:bool=False):
    arr2=np.copy(arr)
    if ~render_null:
        arr2[arr2==0]=.5
    arr_shape = np.asarray(arr2).shape
    xi=np.linspace(xr[0],xr[1],num=arr_shape[0])
    yi=np.linspace(yr[0],yr[1],num=arr_shape[1])
    XX,YY=np.meshgrid(xi,yi)
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"},figsize=figs)
    surf = ax.plot_surface(XX, YY, arr2, cmap=cm.bwr,
                       linewidth=0.1, antialiased=True)
    ax.set_zlim((0,1))
    return fig

def expandChannels(arr2d:np.ndarray,channels:int=4,data_type=float):
    return np.asarray(np.concatenate([np.expand_dims(arr2d,axis=2) for _ in range(channels)],axis=2),dtype=data_type)


