#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 16 11:19:28 2018

@author: yusra
"""

import scipy.ndimage as ndi
import imageio
import matplotlib.pyplot as plt
import numpy as np

## Segmentation of objects 
# splitting of image into parts
# here we will study how to analyze the resulting objects
# we will be looking at sunnybrook cardiac MRI data
# this data contains 3d timeseries of a person's heart for the course of single heartbeat
# the end goal is to measure ejection fraction: the propotion of blood pumped out of left ventricle


## we have to filter/segment the left ventricle from these mri images 
# fluid filled areas have high intensity values
# 1. Filter an image to remove noise and to smooth it
# 2. create mask of pixels having relatively high values

## OBJECT SEGMENTATION AND LABELING


im = imageio.imread('SCD4201-2d.dcm')
filt = ndi.gaussian_filter(im,sigma=2)
mask = filt > 150

# to remove everything else other then left ventricle which has came into image we 
# use scipy label functionality. It treats 0 pixel values as background.
# find connected non 0 pixels and label them as a separate object and returns
# total number of objects found

labels, nlabels = ndi.label(mask)
plt.imshow(labels,cmap="rainbow")
plt.axis('off')
plt.show()
# this shows that left ventricle has been divided in 2 parts

## you can select individual objects by referencing individual values
left_vent1 = np.where(labels == 1,im,0)
left_vent2 = np.where(labels == 2,im,0)
# to select multiple
left_vent = np.where(labels <3 , im, 0)

## OBJECT EXTRACTION and BOUNDING BOXES
# bounding boxes completely encloses all the pixels of an object
# it is used to extract objects from larger image
# ndi.find_objects() returns bounding box coordinates
boxes = ndi.find_objects(labels)
boxes[0] ## should output 2 slices each for each dimension
plt.imshow(im[boxes[0]],cmap="gray")
plt.show()
#plt.imshow(im[boxes[1]],cmap="gray")
#plt.imshow(im[boxes[2]],cmap="gray")  


## Exercise:
# Smooth intensity values
im = imageio.imread("SCD2001_MR_117.dcm")
im_filt = ndi.median_filter(im,size=3)

# Select high-intensity pixels
mask_start = np.where(im_filt > 60, 1, 0)
mask = ndi.binary_closing(mask_start)

# Label the objects in "mask"
labels, nlabels = ndi.label(mask)

print('Num. Labels:', nlabels) 


#Plot the labels array on top of the original image. 
# To create an overlay, use np.where to convert values of 0 to np.nan. 
# Then, plot the overlay with the rainbow colormap and set alpha=0.75 to 
# make it transparent.

overlay = np.where(labels!=0, labels, np.nan)

# Use imshow to plot the overlay
plt.imshow(overlay, cmap="rainbow", alpha=0.75)

# Label the image "mask"
labels, nlabels = ndi.label(mask)

# For this exercise, create a labeled array from the provided mask. 
# Then, find the label value for the centrally-located left ventricle, 
# and create a mask for it.


# Select left ventricle pixels
lv_val = labels[128, 128]
lv_mask = np.where(labels == lv_val,1,np.nan)

# Overlay selected label
plt.imshow(lv_mask, cmap='rainbow')
plt.show()

# For this exercise, use ndi.find_objects() 
# to create a new image containing only the left ventricle.


# Create left ventricle mask
labels, nlabels = ndi.label(mask)
lv_val = labels[128, 128]
lv_mask = np.where(labels == lv_val, 1, 0)

# Find bounding box of left ventricle
bboxes = ndi.find_objects(lv_mask)
print('Number of objects:', len(bboxes))
print('Indices for first box:', bboxes[0])


## MEASURING INTENSITY

# after segmentation of objects their properties can be identified using scipy tools
# mean, mean standard deviation  and labeled_comprehension
vol = imageio.volread("SCD-3d.npz")
label = imageio.volread("labels.npz")

# all pixel
ndi.mean(vol)

# by doing it with label, you'll restrict the analysis to non zero pixels
ndi.mean(vol,label)

# if provided the index value with label, you can get mean intensity for a single
# label or multiple labels
ndi.mean(vol,label,index=1)
ndi.mean(vol,label,index=[1,2])

# Object Histograms can also use labels to be specific about an object in an image
hist = ndi.histogram(vol,min=0,max=255,bins=256) # for whole image
obj_hist = ndi.histogram(vol,0,255,256,labels,index = [1,2])

## these histograms are very informative at object segmentations
# if it has alot of peaks and variations, it means the object has vrious tissue types
# otherwise if it has a normal distribution then your segmentation is good
# physical properties influencing intesity values should be uniform through out the tissue

# Exercise
# Variance for all pixels
# Variance for all pixels
var_all = ndi.variance(vol, labels=None, index=None)
print('All pixels:', var_all)

# Variance for labeled pixels
var_labels = ndi.variance(vol, labels)
print('Labeled pixels:', var_labels)

# Variance for each object
var_objects = ndi.variance(vol,labels,index=[1,2])
print('Left ventricle:', var_objects[0])
print('Other tissue:', var_objects[1])


# Create histograms for selected pixels
hist1 = ndi.histogram(vol, min=0, max=255, bins=256)
hist2 = ndi.histogram(vol, 0, 255, 256, labels)
hist3 = ndi.histogram(vol, 0, 255, 256, labels, index=1) 

plt.plot(hist1/hist1.sum())
plt.plot(hist2/hist2.sum())
plt.plot(hist3/hist3.sum())
plt.imshow()


# MEASURING OBJECT MORPHOLOGY (SHAPE AND SIZE)

# useful when finding how big is the tumor
# if monitoring it for some time, would want to know, has it grown?
# for this we need space occupied by each element and number of elements

# calculate volumne per voxel
d0,d1,d2 = vol.meta['sampling'] # physical real space by object
dvoxel = d0*d1*d2

# count label voxels
nvoxels = ndi.sum(1,label,index=1) # this shows that for each voxel in left ventricle we have assigned the weight 1 and summed them

# total volumne of object 
volume = nvoxels * dvoxel

# Distance transformation (tells distance of each voxel from background)
# left ventricle mask
mask = np.where(labels==1,1,0)

# in terms of voxels
d= ndi.distance_transform_edt(mask)
d.max() # shows how many pixels from the edge the most embedded point is 
d = ndi.distance_transform_edt(mask,sampling = vol.meta['sampling'])
d.max()

# center of mass ( mass refers to intensity values, with larger values pulling the center towards them)
com = ndi.center_of_mass(vol,labels,index=1) # returns tuple coordinates for each object specified

plt.imshow(vol[5],cmap='gray')
plt.scatter(com[2],com[1])
plt.show()

## Exercise 

# Calculate left ventricle distances
lv = np.where(labels==1, 1, 0)
dists = ndi.distance_transform_edt(lv,sampling= vol.meta['sampling'])

# Report on distances
print('Max distance (mm):', dists.max())
print('Max location:',ndi.maximum_position(dists) )

# Plot overlay of distances
overlay = np.where(dists[5] > 0, dists[5], np.nan) 
plt.imshow(overlay, cmap='hot')

# Extract centers of mass for objects 1 and 2
coms = ndi.center_of_mass(vol,labels,index=[1,2])
print('Label 1 center:', coms[0])
print('Label 2 center:', coms[1])

# Add marks to plot
for c0, c1, c2 in coms:
    plt.scatter(c2, c1, s=100, marker='o')
plt.show()

# ---- xxxx -------

# Ejection fraction  = (leftventricale(max) - leftventricle(min))/leftventricle(max)
# leftventricle (max) is volume of ventricle when it is fully relaxed, min is when fully squeezed
# this gives us fraction of blood pumped
# procedure for this would be;
# 1. calculate volume at every time stamp of the image series
# 2. find min and max from this 1 d series of volumes and put in equation and put in equation

## assummed that you have access to volumetric time series
# and segmented left ventricle stored in (t,x,y,z) format
# vol_ts.shape = (20,12,256,256,256)
# labels.shape = (20,12,256,256)

d0,d1,d2,d3=vol_ts.meta['sampling']
dvoxel = d1 * d2 * d3
# Instantiate empty list
ts = np.zeros(20)
# Loop through volume time series
for t in range(20):
    nvoxels=ndi.sum(1,
                    labels[t], 
                    index=1)
    ts[t] = nvoxels * dvoxel

plt.plot(ts)
plt.show()

min_vol = ts.min()
max_vol = ts.max()

ejec_frac = (max_vol - min_vol)/max_vol

# Get index of max and min volumes
tmax = np.argmax(ts)
tmin = np.argmin(ts)

# Plot the largest and smallest volumes
fig, axes = plt.subplots(2,1)
axes[0].imshow(vol_ts[tmax,4], vmax=160)
axes[1].imshow(vol_ts[tmin,4], vmax=160)

