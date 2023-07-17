# -*- coding: utf-8 -*-
"""
Created on Tue Jun 20 16:34:35 2023

@author: Alexander
"""

### Imports
import matplotlib.pyplot as plt 
import matplotlib as mpl
import matplotlib.cm as cm
plt.close('all')
import cv2
import numpy as np
import typing

import imageutils as im

class imageCloneMutator:
    def __init__(self, image:str, maskTolerance:float = 0.5, colorMutationScaling:float= 0.3, colorMutationGuassKernal:int = 6, shapeMutationSeedRate:float = 0.1, shapeMutationDepth:int = 20):
        inputs = (image, maskTolerance, colorMutationScaling, colorMutationGuassKernal, shapeMutationSeedRate, shapeMutationDepth)
        self._assertInputs(inputs)
        self.imageFile, self.maskTol, self.colorMutationScaling  = image, maskTolerance, colorMutationScaling
        self.colorMutationGuassKernal, self.shapeMutationSeedRate, self.shapeMutationDepth = colorMutationGuassKernal, shapeMutationSeedRate, shapeMutationDepth
        self._preProcess()
        
    def _assertInputs(self,inputs):
        assert_types=(str,float,float,int,float,int)
        for assert_ , inpt in zip(assert_types,inputs):
            assert type(inpt) == assert_
    
    def _preProcess(self):
        self.baseImage = plt.imread(self.imageFile)
        self.imageShape = self.baseImage.shape
        self.maskEdge = im.expand_channels(self.baseImage [:,:,-1]!=0)
        self.maskColor = im.expand_channels(np.sum(np.abs(self.baseImage[:,:,:-1]-im.expand_channels(
            np.mean(self.baseImage[:,:,:-1],axis=2),3,float)),axis=2)>=self.maskTol)
        self.colorField = cv2.dilate(self.baseImage*self.maskColor,np.ones((3,3),dtype=np.uint8),iterations=40)
        return None        
    
    def _shapeMutate(self):
        kernal = np.ones((3,3),dtype=np.uint8)
        mask = np.float32(self.maskColor[:,:,:])
        erodeMask = cv2.erode(mask, kernal, iterations=self.shapeMutationDepth)
        dilateMask = cv2.dilate(mask, kernal, iterations=self.shapeMutationDepth)
        seededMask = np.asarray((np.random.rand(*self.imageShape)*(dilateMask)*
                                 (1-erodeMask)-0.95)>0,dtype=np.uint8)
        bloat=10
        mutationMask = cv2.filter2D(cv2.erode(cv2.dilate(seededMask, kernal, iterations=bloat)+mask,kernal,iterations=int(bloat)),-1,kernal)
        mutationMask = im.expand_channels(np.prod(mutationMask[:,:,:]!=0,axis=2))*dilateMask
        MSE_con = 1
        mm_p,mm_n = mutationMask,mutationMask
        condition = lambda mask1, mask2 : np.argmin([im.mse(mask1,self.maskColor),im.mse(mask2,self.maskColor)])
        lim=100
        count=0
        while True:
            count+=1
            mm_n = cv2.erode(mm_p, kernal, iterations=1)
            if ~bool(condition(mm_p,mm_n)):
                return mm_p
            elif count > lim:
                return mm_p
            mm_p=mm_n
        return mutationMask
    
    def _colorMutate(self,mask):
        ks=self.colorMutationGuassKernal
        kernal = np.ones((ks,ks))/ks
        colorMutation = cv2.filter2D(np.tanh(np.random.rand(*self.imageShape)-.5),-1,kernal)
        colorMutation *= (1-np.max(self.baseImage[:,:,:-1]*mask[:,:,:-1]))*self.colorMutationScaling
        #colorMutation[:,:,-1] = 0 
        slottedImage = (self.baseImage-10*mask)
        slottedImage[slottedImage<0]=0
        mutantImage = (self.genRandColorField(self.maskColor))*mask+slottedImage
        return mutantImage,mask,self.colorField#colorMutation
    
    def genMutant(self):
       return self._colorMutate(mask=self._shapeMutate())
       

    
    def genRandColorField(self,mask_color:np.ndarray,cmap=cm.bwr,deg:int=20,al:int=1,amp:float=.1):
        mask_edge=self.maskEdge
        color=self.baseImage
        field  = im.mask2field(np.copy(mask_color),np.copy(self.maskEdge),cmap=cmap)
        scalar = im.colormap2arr(field)
        scalar_new = im.noise2d(scalar,amp)
        fitted,_ = im.fitPlaneFunc(scalar_new,deg=deg,al=al)
        condition = lambda n1, n2, nb: (
            np.abs(sum(n1 - nb))**2 > np.abs(sum(n2 - nb))**2)[0]
        test, thresh = True, 0
        while test:
            n1=im.flattenMidBandNoise(fitted, thresh)[::-1] 
            n2=im.flattenMidBandNoise(fitted, thresh+.01)[::-1]
            test = condition(n1,n2,scalar)
            print(test)
            thresh += .01
        print(thresh)
        return n1
        
test=imageCloneMutator('basebwr.png')
this=test.genMutant()
plt.figure()
plt.imshow(this)

# plt.figure()
# plt.imshow(.5*mask)
# plt.figure()
# plt.imshow(cm)

        