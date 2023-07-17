# -*- coding: utf-8 -*-
"""
Created on Tue Jun 20 12:39:20 2023

@author: Alexander
"""

### Imports
import matplotlib.pyplot as plt 
import matplotlib as mpl
plt.close('all')
import cv2
import numpy as np
import cm2scalar as c2s

#%% UTILS
def add_channel(image:np.ndarray,channel:np.ndarray, iloc:int):
    if iloc < 0:
        iloc_= image.shape[-1]+1-iloc%(image.shape[-1]+1)
    elif iloc < image.shape[-1]:
        iloc_= iloc
    else:
        raise Exception('IndexError')
    return np.insert(image,iloc_,channel,axis=2)
        
        

expand_channels = lambda condition2d,channels=4,data_type=int: np.asarray(np.concatenate([
    np.expand_dims(condition2d,axis=2) for _ in range(channels)],axis=2),dtype=data_type)


#%% MAIN
tol=0.1
base_image = plt.imread('base.png')
ims=base_image.shape

mask_edge=expand_channels(base_image[:,:,-1]!=0)
plt.figure()
plt.imshow(.5*(1-mask_edge))

mask_color=expand_channels(np.sum(np.abs(base_image[:,:,:-1]-expand_channels(
    np.mean(base_image[:,:,:-1],axis=2),3,np.float32)),axis=2)>=tol)
plt.figure()
plt.imshow(.5*(1-mask_color))

# mutate shape
shapeMutationDepth=5
imageShape=base_image.shape
mse = lambda im1, im2 : np.square(np.subtract(im1,im2)).mean()

kernal = np.ones((3,3),dtype=np.uint8)
erodeMask = cv2.erode(np.float32(mask_color[:,:,:])
                      , kernal, iterations=shapeMutationDepth)
dilateMask = cv2.dilate(np.float32(mask_color[:,:,:])
                      , kernal, iterations=shapeMutationDepth)
seededMask = np.asarray((np.random.rand(*imageShape)*(dilateMask)*
                      (1-erodeMask)-0.9)>0,dtype=np.uint8)
priorMask = cv2.dilate(seededMask, kernal, iterations=2)
nextMask = cv2.erode(priorMask, kernal, 1)
count,test=1,True
while (test)and(count<shapeMutationDepth*1.5):
       nextMask = cv2.erode(priorMask, kernal, iterations=count)
       test=(mse(priorMask,mask_color) > mse(nextMask,mask_color))
       nextMask=priorMask                    
       count+=1
if mse(priorMask,mask_color) >= mse(nextMask,mask_color):
    mutate_mask = priorMask
elif mse(priorMask,mask_color) < mse(nextMask,mask_color):
    mutate_mask = nextMask

test=expand_channels(np.sum(mutate_mask[:,:,:]!=0,axis=2))
plt.figure()
plt.imshow(0.9*mutate_mask)



# mutate color
kernal_size = 6
kernal= np.ones((kernal_size,kernal_size))/kernal_size
color_mutation = cv2.filter2D(np.tanh(np.random.rand(*ims)-.5),-1,kernal)
color_mutation *= (1-np.max(base_image[:,:,:-1]*mask_color[:,:,:-1]))
color_mutation[:,:,-1] = 0 
mutated_colors = color_mutation*mask_color+base_image
#mutated_colors = np.where(mutated_colors>1,mutated_colors,1)

# mutate shape


plt.figure()
plt.imshow(mutated_colors)

plt.figure()
plt.imshow(base_image)


if True:
    pass
else:
    # Cluster stuff
    
    color_mask_vector = np.float32(mask_color.reshape((-1, 4)))[:,:-1]
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    _, labels, (centers) = cv2.kmeans(color_mask_vector, 2, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    
    
    # convert back to 8 bit values
    centers = np.uint8(centers)
    
    # flatten the labels array
    labels = labels.flatten()
    
    # convert all pixels to the color of the centroids
    segmented_image = centers[labels.flatten()]
    
    # reshape back to the original image dimension
    segmented_image_rgb = segmented_image.reshape((ims[0],ims[1],3))
    segmented_image=add_channel(segmented_image_rgb, base_image[:,:,-1], -1)
    
    # show the image
    plt.figure()
    plt.imshow(.5*segmented_image)
    plt.show()



 