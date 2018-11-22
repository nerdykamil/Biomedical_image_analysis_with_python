#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  5 23:31:11 2018

@author: yusra
"""
## MASK AND FILTERS

## Intensity Values
# pixels are 2D elements
# voxels are 3D elements
# pixels and voxels have 2 properties - intensity and location
# intensity depends on modality
# uint8 preferred

import imageio
import numpy as np
import scipy.ndimage as ndi # for  histograms
import matplotlib.pyplot as plt

img = imageio.imread("foot-xray.jpg")
print(img.dtype) # retuns uint8
print(img.size)
print('Min. value:', img.min())
print('Max value:', img.max())

img64 = img.astype(np.uint64) 
print(img64.size)

# histograms
# helps us understand the distribution of intensity values in an image
hist = ndi.histogram(img,min=0,max=255,bins=256)
plt.plot(hist) # plotting data as line plot
plt.show()

# skewed distributions are common in medical images because background space is huge 
# and takes up main part of image
# we do hist equalization to reditribute values 
# equalization is done using cdf 
cdf = hist.cumsum() / hist.sum()

# Plot the histogram and CDF
fig, axes = plt.subplots(2, 1, sharex=True)
axes[0].plot(hist, label='Histogram')
axes[1].plot(cdf, label='CDF')
format_and_render_plot()

# MASKS
# a boolean array which serves as a screen to remove undesirable pixels
# should be of same dim as image

mask1 = img > 32 # this would give a pretty good mask incase of xrays
# bone has the highest intensity in xrays because its a translucent obstacle to rays
# hence rays are most absorbed in it
mask2 = img > 64 #this would highlight the bone area pretty nicely
# to find non-bone tissue pixels by subtracting bone mask from whole mask
mask3 = mask1 & ~mask2

# masks can be used to screen or filter image and block the pixelsnot needed 
im_bone = np.where(img > 64,img,0)
plt.imshow(im_bone,cmap="gray")
plt.axis("off")
plt.show()

# to increase size of mask, add pixels on side of edges to make sure you are not missing 
# on important information
# useful when edge are fuzzy
m1 = np.where(img >64,1,0)
m2 = ndi.binary_dilation(m1,iterations=5) # adds pixels to the mask
m3 = ndi.binary_erosion(m1,iterations=5) # opposite of above 
#    binary_opening: Erode then dilate, "opening" areas near edges
#    binary_closing: Dilate then erode, "filling in" holes
m4 = ndi.binary_closing(m1,iterations=5)
m5 = ndi.binary_opening(m1,iterations=5)

## FILTERS

# smoothing takes average with neighboring pixels and provides a blur but blended image
# Sharpening is prominent points stand 

# convolution
img = imageio.imread("foot-xray.jpg")
# smoothing weights
weights = [.11,  .11, .11],
           [.11,  .12, .11],
           [.11,  .11, .11]]

im_filt = ndi.convolve(im,weights)

fig,axes = plt.subplots(2,1)
axes[0].imshow(im,cmap="gray") # original image
axes[1].imshow(im_filt,cmap="gray") #smoothed image
plt.imshow()

# other filters in scipy.ndimage.filters 
# mean_filter()
# median_filter()
# maximum_filter()
# percentile_filter()

# filter kernels donot have to be 3x3, they can be as large as you want

ndi.median_filter(im,size=10) # does nice smoothing
# gaussian filters are used for smoothing larger areas
# gussian is a great way to remove noise but with large sigma values you lose details
ndi.gaussian_filter(im,sigma = 5)
ndi.gaussian_filter(im,sigma = 10)
 

## filters can also be used as feature extracters
# edges are changes in intensity along an axis

im = imageio.imread("foot-xray.jpg")

weights = [[1,1,1],[0,0,0],[-1,-1,-1]]

edges = ndi.convolve(im,weights)
plt.imshow(edges,cmap = "seismic")

# sobel edge detector has weights
# weights = [[1,2,1],[0,0,0],[-1,-2,-1]] for horizontal edges
# weights = [[1,0,-1],[2,0,-2],[-1,0,-1]] for horizontal edges

sobel_fil = ndi.sobel(im,axis = 0) #for horizontal
sobel_fil2 = ndi.sobel(im,axis = 1) #for horizontal

# combine horizontal and vertical edge by finding the distance between them (pythagorus theorem)

edges0 = ndi.sobel(im,axis = 0) #for horizontal
edge1 = ndi.sobel(im,axis = 1) #for horizontal

edge = np.sqrt(edge0**2+edge1**2)

plt.imshow(edge,cmap = "gray")