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
from matplotlib import animation
import os

import imageutils as im

class imageCloneMutator:
    def __init__(self, image:str, maskTolerance:float = 0.05, colorMutationScaling:float= 0.3, colorMutationGuassKernal:int = 6, shapeMutationSeedRate:float = 0.1, shapeMutationDepth:int = 20):
        '''
        

        Parameters
        ----------
        image : str
            DESCRIPTION.
        maskTolerance : float, optional
            DESCRIPTION. The default is 0.05.
        colorMutationScaling : float, optional
            DESCRIPTION. The default is 0.3.
        colorMutationGuassKernal : int, optional
            DESCRIPTION. The default is 6.
        shapeMutationSeedRate : float, optional
            DESCRIPTION. The default is 0.1.
        shapeMutationDepth : int, optional
            DESCRIPTION. The default is 20.

        Returns
        -------
        None.

        '''
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
        mm_p,mm_n = mutationMask,cv2.erode(mutationMask, kernal, iterations=1)
        l1norm=lambda m1, m2:np.abs(np.sum(m1-m2))
        condition = lambda m1, m2 : np.argmin([l1norm(m1,self.maskColor),l1norm(m2,self.maskColor)])
        erodestep = lambda  m : (m , cv2.erode(m, kernal, iterations=1))
        mm_p,mm_n = erodestep(mutationMask)
        while condition(mm_p,mm_n):
            mm=mm_p
            mm_p,mm_n = erodestep(mm_n)
        return im.norm(mm)#im.norm(cv2.filter2D(mm_p,-1,kernal))
    
    def _mutate(self,mask:np.ndarray,field:np.ndarray):
        slottedImage = (self.baseImage-10*self.maskColor)
        slottedImage[slottedImage<0]=np.nan
        slottedImage = im.nan_interpolate(slottedImage)
        slottedImage = (self.baseImage-10*mask)
        slottedImage[slottedImage<0]=0       
        mutantImage = field*mask+slottedImage
        return mutantImage      

    
    def _genRandColorField(self,cmap=cm.bwr,deg:int=20,al:int=1,amp:float=.1):
        field  = im.mask2field(np.copy(self.maskColor),np.copy(self.baseImage),cmap=cmap)
        scalar = im.colormap2arr(field)[::-1]
        scalar_new = im.noise2d(scalar,amp)
        fitted,_ = im.fitPlaneFunc(scalar_new,deg=deg,al=al)

        field=im.arr2colormap(im.flattenMidBandNoise(fitted, 0))
        return im.norm(field)
    
    def genMutant(self,amp:float=.1):
       return self._mutate(mask=self._shapeMutate(),field=self._genRandColorField(amp=amp))
        




imGen=imageCloneMutator('basebwr.png')
for i in range(1):
    print(i)
    image = im.norm(imGen.genMutant(amp=.5))
    plt.imsave(f'export/{i}_test.png',image)

if False:
    from PIL import Image
    imapp,i,im1 =[],1,Image.open('export/0_test.png')
    while os.path.exists(f'export/{i}_test.png'):
        imapp.append( Image.open(f'export/{i}_test.png'))
        i+=1
    im1.save("export/all_test_images.gif", save_all=True, append_images=imapp, duration=500,loop=0)
  


        