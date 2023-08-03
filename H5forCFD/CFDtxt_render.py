# -*- coding: utf-8 -*-
"""
Created on Fri Jul 28 17:40:54 2023

@author: caleb
"""


import matplotlib.pyplot as plt
import matplotlib.tri as tri
import numpy as np
import cv2

# Global figure and axes for the main plot and colorbar
fig_main, ax_main = plt.subplots(figsize=(17, 8))
gs = fig_main.add_gridspec(1, 2, width_ratios=[9, 1])
ax_main = fig_main.add_subplot(gs[0, 0])
cax = fig_main.add_subplot(gs[0, 1])

dirPath = "CFD_Results/CFD_60_"
n = [1]
t = np.linspace(0,60, 401)
columns = ['x' , 'y', 'pressure', 'velocity (x)', 'velocity (y)']
for j in n:
    split = str(j)
    path = f'{dirPath}{split}/CFD_60_{split}_'
    Data = np.atleast_3d(np.ones((1, 1434, 5)))
    
    output_file = f'output_video{j}.mp4'
    fps = 10
    fourcc = cv2.VideoWriter_fourcc(*'acv1')
    video_writer = cv2.VideoWriter(output_file, fourcc, fps, (1700, 800))
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

    for k in range(35):
        k*=10
        print(k)
        x_values = Data[k, :, 0]
        y_values = Data[k, :, 1]
        velocity_valuesX = Data[k, :, 3]
        velocity_valuesY = Data[k, :, 4]
        x = (Data[0,:,0])
        y = (Data[0,:,1])
        
        Cells=tri.Triangulation(x, y).get_masked_triangles()
        c =np.atleast_3d(Cells)
        c= np.swapaxes(c,0,2)
        c= np.swapaxes(c,2,1)
        Cells = (np.repeat(c, np.shape(Data)[0], 0))

        # Create the Triangulation object for the current frame
        triang = tri.Triangulation(x_values, y_values, Cells[0,:,:])
        # Set up the subplots with GridSpec

        # Create the tripcolor plot on the main axis
        vMag = np.sqrt(velocity_valuesX**2 + velocity_valuesY**2)
        plot_trip = ax_main.tripcolor(triang, vMag)
        # Set axis labels and title for the main plot
        ax_main.set_xlabel('X Coordinate')
        ax_main.set_ylabel('Y Coordinate')
        ax_main.set_title('Tripcolor Graph from HDF5 File')
        # Create the colorbar
        cb = plt.colorbar(plot_trip, ax=ax_main, cax=cax)
        cb.set_label('Velocity')  # Set the colorbar label

        # Plot the quiver plot for velocity direction on the same axis
        # ax_main.quiver(x_values, y_values, velocity_valuesX, velocity_valuesY, angles='xy', scale_units='xy', scale=0.1, color='black')

        fig_main.canvas.draw()
        frame = np.array(fig_main.canvas.renderer.buffer_rgba())
        frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)
        video_writer.write(frame)
        
    print(f"video{j} created successfully")
    video_writer.release()
    plt.close()