# -*- coding: utf-8 -*-
"""
Created on Tue Jul 25 20:05:26 2023

@author: caleb
"""

import h5py
import matplotlib.pyplot as plt
import matplotlib.tri as tri
import numpy as np
import cv2

# Global figure and axes for the main plot and colorbar
fig_main, ax_main = plt.subplots(figsize=(17, 8))
gs = fig_main.add_gridspec(1, 2, width_ratios=[9, 1])
ax_main = fig_main.add_subplot(gs[0, 0])
cax = fig_main.add_subplot(gs[0, 1])

# Load data from the HDF5 file
with h5py.File('H5_file/test.h5', 'r') as hf:
    for i in np.int8(np.linspace(0, 99, 6)):
        step = hf[f'{i}']
        output_file = f'output_video{i}.mp4'
        fps = 10
        fourcc = cv2.VideoWriter_fourcc(*'acv1')
        video_writer = cv2.VideoWriter(output_file, fourcc, fps, (1700, 800))

        for j in np.int8(np.linspace(0,len(step['pos'][:, 0, 0]),80)):
            x_values = step['pos'][j, :, 0]
            y_values = step['pos'][j, :, 1]
            velocity_valuesX = step['velocity'][j, :, 0]
            velocity_valuesY = step['velocity'][j, :, 1]
            triangulation_indices = step['cells'][j, :, :]

            # Create the Triangulation object for the current frame
            triang = tri.Triangulation(x_values, y_values, triangulation_indices)
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
            
        print(f"video{i} created successfully")
        video_writer.release()
        plt.close()
       