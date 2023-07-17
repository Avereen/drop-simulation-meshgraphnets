# -*- coding: utf-8 -*-
"""
Created on Fri Jun 23 11:51:58 2023

@author: Alexander
"""
import matplotlib.pyplot as plt
import numpy as np
from image_mut_class import imageCloneMutater

def main(image:str,batch:int):
    generator= imageCloneMutater(image)
    for i in range(batch):
        print(i)
        plt.figure(figsize=(3,3))
        plt.imshow(generator.genMutant())
        plt.xticks([])
        plt.yticks([])
        plt.savefig('im/test_image_{}.png'.format(i))
        plt.close("all")
    

if __name__ =='__main__':
    main('base.png',20)