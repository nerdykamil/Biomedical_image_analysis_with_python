#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  5 10:17:08 2018

@author: yusra
"""

## Image Data

# importing relevant libraries 
import imageio
import matplotlib.pyplot as plt
import numpy as np

## reading image file
img = imageio.imread("check-002.dcm") #imageio can load dicom images (medical images are in dicom format usually) 
# Print image attributes
print('Image type:', type(img))
print('Shape of image array:', img.shape)

# imageio loads metadata along with the image which is then presented as dict

print(img.meta) # meta contains all the patient information
print(img.meta['Modality'])
print(img.meta.keys())

# plotting
plt(img,cmap="gray")
plt.axis("off") # turns off axis lines
plt.show()

# numpy is able to stack images - 
# the stack fn can be used for multi dimensional images and for stacking 2d images
# as follows

im1 = imageio.read("chest-000.dcm") 
im2 = imageio.read("chest-001.dcm")
im3 = imageio.read("chest-002.dcm")

vol = np.stack(im1,im2,im3)

# we can read multiple images from a directory in a numpy stack
vol = imageio.volread('tcia-chest-ct') #here tcia-chest-ct is a folder
# the volread fn checks metadata to make sure images are placed in correct order
# otherwise they are placed in alphabetical order

# sampling rate (in mm) is the amount of physical space covered by each element
# fieldofview is the physical space taken up along each axis 

n0,n1,n2 = vol.shape
d0,d1,d2 = vol.meta['sampling']
fieldofview = n0*d0,n1*d1,n2*d2

## Advanced Plotting
# to gather more information about the image we need to plot its slices 
# for this we will use subplots from pyplot

fig, axes = plt.subplot(nrows =1, ncols =3)

# to plot subplots we will call imshow function directly from axes
axes[0].imshow(vol[0],cmap='gray')
axes[1].imshow(vol[1],cmap = 'gray')
axes[2].imshow(vol[2],cmap='gray')

for ax in axes:
    ax.axis("off")
    
plt.show()

# another way to display 3d images is to select an image in first dimension 
# and plot 2nd and 3rd against each other    

vol = imageio.volread("chest-data")
view_1v2 = vol[pln,:,:] # axial view
view_1v2 = vol[pln]

view_0v2 = vol[:,row,:] # coronal view # in these 3 lines we are plotting head to toe vs left to right
# if selecting row slices we would get a diff view

view_0v1 = vol[:,:,col] # saggital view # plotting 1st and 2nd against each other using a third vield

# for these views it is important to miantain aspect ratios
im = vol[:,:,100]
d0,d1,d2 = vol.meta['sampling']
# to calculate aspect ratio divide sampling rate of first dim to 2nd dim
asp = d0/d1
# pass this asp to plt function
plt.imshow(im,cmap="gray",aspect=asp)


