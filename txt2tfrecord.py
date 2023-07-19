# -*- coding: utf-8 -*-
"""
Created on Fri Jul 14 13:35:57 2023

@author: caleb
"""
"Mainly used to convert CFD txt files from SolidWorks to an h5 file"

import numpy as np
#import os
# import pandas as pd
# import csv
#from scipy.spatial import Delaunay
import matplotlib as mpl
ds=[]
dirPath = "CFD_Results/CFD_60_"
n = [1, 2, 5, 10]
t = np.linspace(0,60, 401)
columns = ['x' , 'y', 'pressure', 'velocity (x)', 'velocity (y)']
for j in n:
    split = str(j)
    path = f'{dirPath}{split}/CFD_60_{split}_'
    Data = np.atleast_3d(np.ones((1, 1280, 5)))
    for i in t:
        step = f'{i:.2f}'
        data = np.loadtxt(path+step+'.txt',
                          delimiter = '\t', 
                          skiprows=1,
                          dtype=str)
        
        slc=data[np.float64(data[:,2])>=np.max(np.float64(data[:,2]))]
        slc=np.delete(slc, (2,3,4,6,9,10), axis=1)
        data =np.atleast_3d(slc)
        data = np.swapaxes(data,0,2)
        data= np.swapaxes(data,2,1)
        Data= np.concatenate((Data,data),axis=0)
    Data = np.delete(Data,0, axis=0)
    #csv.writer(f"CFD_Results/CFD_60_{i}.csv", dialect='excel')
        
    #Cells
    #tri = Delaunay(Data[(0,1),:])
    if j ==1:
        x = np.float64(Data[0,:,0])
        y = np.float64(Data[0,:,1])
        Cells=mpl.tri.Triangulation(x, y).get_masked_triangles()
        c =np.atleast_3d(Cells)
        c= np.swapaxes(c,0,2)
        c= np.swapaxes(c,2,1)
        Cells = np.repeat(c, np.shape(Data)[0], 0)

    #Pos
    Pos = Data[:,:,(0,1)]
    
    #NodeType
    if j == 1:
        def in_boundary(x,y):
            cPx = np.mean((0.2794, -0.1778)) 
            cPy = 0
            center = [cPx,cPy]
            radius = 1/39.37
            if np.sqrt((center[0]-x)**2+(center[1]-y)**2)<=radius:
                return 6 #cylinder nodes
            if x == np.max(np.float64(Pos[0,:,0])):
                return 4 #outlet nodes
            if x == np.min(np.float64(Pos[0,:,0])):
                return 5 #inlet nodes
            
            return 0
        dot=np.vectorize(in_boundary)
        
        radius = 1/39.37
        
        node_type = dot(np.float64(Pos[0,:,0]), np.float64(Pos[0,:,1]))  
        Node_type = np.repeat(np.atleast_3d(node_type), np.shape(Data)[0], 0)
    
    #Velocity (x,y)
    Velocity = Data[:,:,(3,4)]
    
    #Pressure
    Pressure = Data[:,:,2]
    ds_dict = {}
    data = ("pos", "node_type", "velocity", "cells", "pressure")
    ds_dict.update({data[0]: Pos,
                    data[1]: Node_type,
                    data[2]: Velocity,
                    data[3]: Cells,
                    data[4]: Pressure})
    # data_set = np.concatenate((Pos, Node_type, Velocity, Cells, Pressure), axis=2)
    ds.append(ds_dict)
   
#%%
#Convert data to h5 file 
import h5py

for catg in ['train']:
    # ds = load_dataset(tf_datasetPath, split)
    save_path = f'C:/Users/caleb/Practice/Project Data/H5_file/CFD_60_{catg}.h5'
    f = h5py.File(save_path, "w")
    print(save_path)
    data = ("pos", "node_type", "velocity", "cells", "pressure")
    
    for index, d in enumerate(ds):
        pos = np.float64(d[data[0]])
        node_type = np.float64(d[data[1]])
        velocity = np.float64(d[data[2]])
        cells = np.float64(d[data[3]])
        pressure = np.float64(d[data[4]])
        data = ("pos", "node_type", "velocity", "cells", "pressure")
        #d = f.create_dataset(str(index), (len(data), ), dtype=pos.dtype)
        g = f.create_group(str(index))
        for k in data:
            g[k] = eval(k)
        
        print(index)
    f.close()