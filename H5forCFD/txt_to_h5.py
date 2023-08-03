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
# import json
import pickle as pkl
import matplotlib as mpl

copy=lambda dictionary:pkl.loads(pkl.dumps(dictionary)) #this is just a function

ds=[]
dirPath = "CFD_Results/CFD_60_"
n = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
t = np.linspace(0,60, 401)[:351]
columns = ['x' , 'y', 'pressure', 'velocity (x)', 'velocity (y)']
for j in n:
    split = str(j)
    path = f'{dirPath}{split}/CFD_60_{split}_'
    Data = np.atleast_3d(np.ones((1, 1434, 5)))
    for i in t:
        step = f'{i:.2f}'
        data = np.loadtxt(path+step+'.txt',
                          delimiter = '\t', 
                          skiprows=1,
                          dtype=str)
        
        slc=data[np.abs(.004+np.float64(data[:,2]))<=1e-3]#np.median(np.abs(np.float64(data[:,2])))]
        slc=(np.delete(slc, (2,3,4,6,9,10), axis=1))
        slc = slc[~np.any(slc=='',axis=1)]
        data =np.atleast_3d(slc)
        data = np.swapaxes(data,0,2)
        data= np.swapaxes(data,2,1)
        Data= np.append(Data,data,axis=0)
    Data = np.float32(np.delete(Data,0, axis=0))
    Data = np.flip(Data, axis=0)
    
       
    #Cells
    #tri = Delaunay(Data[(0,1),:])
    if j ==1:
        x = (Data[0,:,0])
        y = (Data[0,:,1])
        
        Cells=mpl.tri.Triangulation(x, y).get_masked_triangles()
        c =np.atleast_3d(Cells)
        c= np.swapaxes(c,0,2)
        c= np.swapaxes(c,2,1)
        Cells = np.atleast_3d(np.repeat(c, np.shape(Data)[0], 0))

    #Pos
    
    Pos = (np.atleast_3d(Data[:,:,(0,1)]))

    Pos[:,:,0]-=Pos[:,:,0].min()
    Pos[:,:,1]-=Pos[:,:,1].min()
    
    #NodeType
    if j == 1:
        def in_boundary(x,y):
            cPx = np.mean((0.2794, -0.1778)) 
            cPy = 0
            center = [cPx,cPy]
            radius = 1/39.37
            if np.sqrt((center[0]-x)**2+(center[1]-y)**2)<=radius:
                return 6 #cylinder nodes
                #return 1 or 2 # if it is an obstical or airfoil
            # if y == np.max((Pos[0,:,1])):
            #    return 6 #flow boundary
            # if y == np.min((Pos[0,:,1])):
                # return 6 #flow boundary
            if x == np.max(Pos[0,:,0]):
                return 5 #outlet nodes
            if x == np.min(Pos[0,:,0]):
                return 4 #inlet nodes
            
            return 0
        dot=np.vectorize(in_boundary)
        
        radius = 1/39.37
        
        node_type = dot((Pos[0,:,0]), (Pos[0,:,1]))  
        Node_type = np.int32(np.repeat(np.atleast_3d(node_type), np.shape(Data)[0], 0))
        
    
    #Velocity (x,y)
    Velocity = np.atleast_3d(Data[:,:,(3,4)])
    
    #Pressure
    Pressure = np.atleast_3d(Data[:,:,2])
    Pressure-=Pressure.min()
    Pressure/=Pressure.max()
    Pressure-=0.5
    Pressure*=20
    ds_dict = {}
    data = ("pos", "node_type", "velocity", "cells", "pressure")
    ds_dict.update({data[0]: Pos,
                    data[1]: Node_type,
                    data[2]: Velocity,
                    data[3]: Cells,
                    data[4]: Pressure})
    # data_set = np.concatenate((Pos, Node_type, Velocity, Cells, Pressure), axis=2)
    ds.append(ds_dict)
# generate noise using np.rand and np.std and go from 10 to 100 datasets
#json.loads(json.dumps(ds))
    for k in range(9):
        ds_dictCopy=copy(ds_dict)
        VelstdX=np.std(ds_dictCopy['velocity'][:,:,0])
        VelstdY=np.std(ds_dictCopy['velocity'][:,:,1])
        q = np.size(ds_dictCopy['velocity'][:,:,0])
        ZvelX = 0.5*VelstdX/(q**.5)*100
        ZvelY = 0.5*VelstdY/(q**.5)*100
        Prestd = np.std(ds_dictCopy['pressure'])
        Zpres = 0.5*Prestd/(q**.5)
        mask = Node_type[:,:,0]
        mask[mask==1]=1
        mask[mask!=1]=0
        ds_dictCopy['velocity'][:,:,0]+=ZvelX*mask*np.random.rand(*ds_dictCopy['velocity'][:,:,0].shape)
        ds_dictCopy['velocity'][:,:,1]+=ZvelY*mask*np.random.rand(*ds_dictCopy['velocity'][:,:,1].shape)
        ds_dictCopy['pressure']+=Zpres*np.atleast_3d(mask)*np.random.rand(*ds_dictCopy['pressure'].shape)
        ds.append(ds_dictCopy)


#%%
#Convert data to h5 file 
import h5py
print("it's making the h5 file")
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