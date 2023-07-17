# -*- coding: utf-8 -*-
"""
Created on Tue Jun 27 10:04:57 2023

@author: Alexander
"""
#https://stackoverflow.com/questions/3720840/how-to-reverse-a-color-map-image-to-scalar-values

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import scipy.cluster.vq as scv

def colormap2arr(arr,cmap):    
    # http://stackoverflow.com/questions/3720840/how-to-reverse-color-map-image-to-scalar-values/3722674#3722674
    gradient=cmap(np.linspace(0.0,1.0,100))

    # Reshape arr to something like (240*240, 4), all the 4-tuples in a long list...
    arr2=arr.reshape((arr.shape[0]*arr.shape[1],arr.shape[2]))

    # Use vector quantization to shift the values in arr2 to the nearest point in
    # the code book (gradient).
    code,dist=scv.vq(arr2,gradient)

    # code is an array of length arr2 (240*240), holding the code book index for
    # each observation. (arr2 are the "observations".)
    # Scale the values so they are from 0 to 1.
    values=code.astype('float')/gradient.shape[0]

    # Reshape values back to (240,240)
    values=values.reshape(arr.shape[0],arr.shape[1])
    values=values[::-1]
    return values

arr=plt.imread('mri_demo.png')
values=colormap2arr(arr,cm.jet)    
# Proof that it works:
plt.imshow(values,interpolation='bilinear', cmap=cm.jet,
           origin='lower', extent=[-3,3,-3,3])
plt.show()