#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 19 11:38:17 2018

@author: yusra
"""

# dealing

## SPATIAL TRANSFORMATIONS

# Brain Data from OASIS (Open Access Series of Imaging Studies)
# 400 MRIs of adults (Age: 18 - 80) and have mild to severe alzheimers disease)
# with multiple subject data there will be differences in intensity scales, 
#object orientation and object placement 
# register images to a predefined position and coordinate system
# align all images with atlas or template image
# the process of aligning two images together is called registration
# registration requires multiple operations - scaling, shifting, resizing, rotating etc
# affine tranformations rotate and shift preserving points, lines and planes 
# operation in affine transformations are translate, rotate, scale, shear

## Translation
import imageio
import scipy.ndimage as  ndi
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind

im = imageio.imread('OAS1036-2d.dcm')
print(im.shape) # find center if 256,256 then 128,128 is te center
com = ndi.center_of_mass(im)
#finding diff in actual center and image center in both column and rows 
d0 = 128 - com[0]
d1 = 128 - com[1] 

xfm = ndi.shift(im,shift=[d0,d1])

fig, axes = plt.subplots(2,1)
axes[0].imshow(im)
axes[1].imshow(xfm)

## Rotation 
xfm = ndi.rotate(im,angle=25) # func by default rotates all pixels in image
print(xfm.shape) # 297,297
xfm = ndi.rotate(im,angle=25,reshape=False) # this will preserve the shape of original image
print(xfm.shape) # 256,256


# we can use transformation matrix to simplify registration process

# Identity matrix
mat = [[1, 0, 0],
       [0, 1, 0],
       [0, 0, 1]]
xfm = ndi.affine_transform(im, mat)

# Translate and rescale
mat = [[0.8, 0,   -20],
       [0,   0.8, -10],
       [0,   0,   1]]

xfm = ndi.affine_transform(im, mat)

# Resampling and Interpolation

# when comparing images differences and array shape and sampling rate can pose hurdles
# resampling: slicing data onto a different array, it is different from cropping 
# because you don't lose the field of view instead the amount of space sampled by each pixel
# is increased or decreased


# in downsampling, information in multiple pixels is merged to reduce the image size

vol = imageio.volread("OAS1_0255")
print(vol.shape)

vol_dn = ndi.zoom(vol,zoom=0.5) # the resultant image will have size 128,128 - loses info but descreases memory consumed by half
print(vol_dn.shape) 

# in upsampling would make the size large but wouldn't increase the resolution as you are not adding info
# it can however make sampling rates across each dimension equal

vol_up = ndi.zoom(vol,zoom=2) # the resultant image will have size 512,512 - 
print(vol_up.shape)

# interpolation
# resampling is actually interpolation which is fitting the existing data to a given dimension by adding or removing data points
# eg: nearest neighbour interpolation, b-spline interpolation
# 2d interpolation
im=np.arange(100).reshape([10,10])
zm1 = ndi.zoom(im, zoom=10, order=0) # less smooth
zm2 = ndi.zoom(im, zoom=10, order=1) # more smooth 
zm2 = ndi.zoom(im, zoom=10, order=2) # most smooth - more computation time
voxel_cubic = ndi.zoom(im, zoom=(2,1,1)) # this can be used to make voxels cubic

# comparing images
# how to make sure the automated segmentation of image is correct or not 
# define a metric of similarity - getting all pixel level comparisons to a single point
# Cost functions such as MAE, MSE and SSE are to be minimized
# Objective Functions such as Intersection of Union are to be maximized

# Mean Absolute Error

im1=imageio.imread('OAS1035-v1.dcm')
im2=imageio.imread('OAS1035-v2.dcm')
err = im1 - im2 # subtracting each pixel from other
plt.imshow(err)
abs_err = np.abs(err) # absolute because need to find if images vary in any way without considering the value of change
plt.imshow(abs_err)

mae = np.mean(abs_err)
print(mae)

# we need to minimize this cost func by altering one or both images
# altering
xfm = ndi.shift(im1,shift=(-8,-8)) # translation
xfm = ndi.rotate(xfm,-18,reshape=False) # rotation

# calculation cost
abs_err = np.abs(xfm-im2)
mean_abs_err = np.mean(abs_err)

# one issue with mean absolute error would be that tissues with higher intensity 
# values would contribute more than those with low intensity values
# solution to above problem would be to compare image masks using IOU
# IOU = shared mask values / total mask values

mask1 = im1 > 0 
mask2 = im2 > 0 

intsxn = mask1 & mask2
plt.imshow(intsxn)

union = mask1 | mask2
plt.imshow(union)

iou = intsxn.sum() / union.sum()
print(iou) # 0 - no match, 1 - perfect overlap 

## Normalizing measurements
# assume that df contains OASIS dataset in following format
# df.sample(5)

# to check if male and female brains are equal
brain_m = df.loc[df.sex == 'M', 'brain_vol']
brain_f = df.loc[df.sex == 'F', 'brain_vol']
 
# ttest to test the null hypothesis, null hypothesis = malebrain == female brains
results = ttest_ind(brain_m,brain_f)

# p value corresponds to the probability of null hypothesis being right
# t statistic is a metric which if is more higher means that null hypothesis is not correct
print(results)
# here you get a low p value and high t statistic which shows that null hypothesis isn't right
# i.e female brains are not equal to male 
# should we consider these results? No! we need to first see correlating measurements
# brains are in skulls and skull size is w.r.t the body size
# plot brain and skull vol correlation
df[['brain_vol','skull_vol']].corr()

## now we'll have to see that if there is really a diff in men and women brain actually
# and not because of the body size

# normalizing
# normalize brain vol w.r.t skull size by taking ratio of brain_vol to skull_vol 
df['brain_norm'] = df.brain_vol/df.skull_vol

# now repeat ttest
brain_m = df.loc[df.sex == 'M', 'brain_norm']
brain_f = df.loc[df.sex == 'F', 'brain_norm']
results = ttest_ind(brain_m,brain_f)

print(results)
# results show that size of people drive brain size not gender, hence males 

# confounds omniexist in data science and are pervasive 
# when taking clinical data from different sites we have to make sure that 
# image acquisition is correct and the data is not biased in any way

# Print prevalence of Alzheimer's Disease
print(df.alzheimers.value_counts())

# Print a correlation table
print(df.corr())

# There is a high correlation - nearly 0.7 - between the brain_vol and skull_vol.
# We should be wary of this (and other highly correlated variables) when 
# interpreting results.

# Import independent two-sample t-test
from scipy.stats import ttest_ind

# Select data from "alzheimers" and "typical" groups
brain_alz = df.loc[df.alzheimers == True, 'brain_vol']
brain_typ = df.loc[df.alzheimers == False, 'brain_vol']

# Perform t-test of "alz" > "typ"
results = ttest_ind(brain_alz, brain_typ)
print('t = ', results.statistic)
print('p = ', results.pvalue)

df.boxplot(column='brain_vol',by='alzheimers')
plt.show()

# There is some evidence for decreased brain volume in individuals with Alzheimer's Disease. 
# Since the p-value for this t-test is greater than 0.05, we would not reject the null hypothesis
# that states the two groups are equal.

# Import independent two-sample t-test
from scipy.stats import ttest_ind

# Divide `df.brain_vol` by `df.skull_vol`
df['adj_brain_vol'] = df['brain_vol']/df['skull_vol']

# Select brain measures by Alzheimers group
brain_alz = df.loc[df.alzheimers == True, 'adj_brain_vol']
brain_typ = df.loc[df.alzheimers == False, 'adj_brain_vol']

# Evaluate null hypothesis
results = ttest_ind(brain_alz,brain_typ)
results
