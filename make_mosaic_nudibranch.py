# -*- coding: utf-8 -*-
"""
Created on Sat Aug 20 11:49:37 2022

@author: wlwee
"""

#%%

import os
import glob

import pandas as pd
import numpy as np
import cv2
import random
import math

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib as mpl

import scipy.stats as stats
from skimage import io
from tqdm import tqdm

#%%

def ResizeWithAspectRatio(img, ratio, inter=cv2.INTER_AREA):
    (h, w) = img.shape[:2]
    h = int(ratio*h)
    w = int(ratio*w)
    dim = (w,h)
    return cv2.resize(img, dim, interpolation=inter)

def get_diff_abs (l):
    return l[1]

def calc_diff (s, pixel):
    dR = s[0] - pixel[0] 
    dG = s[1] - pixel[1]
    dB = s[2] - pixel[2]
    dRGB = np.mean(s) - np.mean(pixel)
    D = (dR**2 + dG**2 + dB**2 + dRGB**2)**(1/2)
    return D

#%%

path_img = r'C:/Users/wlwee/Documents/hypothesis-writing/inaturalist_mosaics/DATA/Nudibranch_mosaics/Screenshot 2022-08-21 112751.png'
path_df = r'C:/Users/wlwee/Documents/hypothesis-writing/inaturalist_mosaics/DATA/Nudibranch_mosaics/tdim-75-75_Nudibranch.pickle'
path_mosaic = os.path.join(os.path.dirname(path_img), 'mosaic-' + os.path.basename(path_img))
df = pd.read_pickle(path_df)

#%%

n_colors = 30
resize = True
ratio = .3

img = io.imread(path_img)[:, :, 0:3]
if resize == True:
    img = ResizeWithAspectRatio(img, ratio, inter=cv2.INTER_AREA)
pixels = np.float32(img.reshape(-1, 3))

n_min_sample = 20

use_dom = False

RGB_Avg = df['RGB-Avg'].tolist()
RGB_Avg = [s[0:3] for s in RGB_Avg]
RGB_Avg = np.array(RGB_Avg)
RGB_Dom = np.array(df['RGB-Dom'].tolist())

pixel_choices = []
pbar = tqdm(total = len(pixels), position=0, leave=True)
for pixel in pixels[0:]:
    if use_dom == True:
        diff = [[i, calc_diff(s, pixel)] for i, s in list(zip(range(0, len(RGB_Dom)), RGB_Dom))]
        diff = sorted(diff, key = get_diff_abs)
        rchoice = random.sample(diff[0:n_min_sample], 1)[0]
    else: 
        diff = [[i, calc_diff(s, pixel)]  for i, s in list(zip(range(0, len(RGB_Avg)), RGB_Avg))]
        diff = sorted(diff, key = get_diff_abs)
        rchoice = random.sample(diff[0:n_min_sample], 1)[0]
    pixel_choices.append(rchoice)
    pbar.update(n=1)
pixels = pixels.reshape(img.shape) 

npixels = len(pixel_choices)
nrows = pixels.shape[0]
ncols = pixels.shape[1]

vimgs = []
n = 0
pbar = tqdm(total = npixels, position=0, leave=True)
for i in range(0,nrows):
    if n > npixels-1:
        break
    himgs = []
    for j in range(0, ncols):
        if n > npixels-1:
            break
        idx = pixel_choices[n][0]
        img = df['Image'][idx]
        himgs.append(img)
        n += 1
        pbar.update(n=1)
        
    himg = himgs[0][:,:,0:3]
    for img in himgs[1:]:
        himg = np.hstack([himg, img[:,:,0:3]])
    vimgs.append(himg)
pbar.close()

if len(vimgs) > 1:
    img = vimgs[0]
    for vimg in vimgs[1:]:
        img = np.vstack([img, vimg])
else:
    img = vimgs[0]

img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

cv2.imwrite(path_mosaic, img)





