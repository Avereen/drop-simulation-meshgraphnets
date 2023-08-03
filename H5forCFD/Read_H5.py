# -*- coding: utf-8 -*-
"""
Created on Thu Jul 20 12:51:34 2023

@author: caleb
"""

import h5py
import numpy as np
#"H5_file/CFD_60_train.h5"
#"H5_file/test.h5"
path =  "H5_file/test.h5"
file1 = h5py.File(path, "r")
print(len(file1))


z1=file1['13']
print(z1.keys())
cellsDS1 = z1['cells'][:,:,:]
possDS1 = z1['pos'][:,:,:]
nTypeDS1 = z1['node_type'][:,:,:]
velDS1 = z1['velocity'][:,:,:]
pressDS1 = z1['pressure'][:,:,:]

path =  "H5_file/CFD_60_train.h5"
file2 = h5py.File(path, "r")
print(len(file2))

z2=file2['13']
print(z2.keys())
cellsDS2 = z2['cells'][:,:,:]
possDS2 = z2['pos'][:,:,:]
nTypeDS2 = z2['node_type'][:,:,:]
velDS2 = z2['velocity'][:,:,:]
pressDS2 = z2['pressure'][:,:,:]
# datasets = list(file2.keys())
# np.random.shuffle(datasets)
# num=datasets[2]

# keys = datasets[0:3]
# zit = {k:file2[k] for k in keys}
# zin = zit[1]