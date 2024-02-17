# -*- coding: utf-8 -*-
"""
Created on Sat Aug 20 11:10:31 2022

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

path_root_dir = r'C:/Users/wlwee/Documents/hypothesis-writing/inaturalist_mosaics/DATA/Amphipoda'
target_dir = os.path.join(os.path.dirname(path_root_dir), os.path.basename(path_root_dir).split('_')[0] + '_mosaics')
if os.path.exists(target_dir) != True:
    os.mkdir(target_dir)

tdim = (75, 75)
n_colors = 30

path_df = os.path.join(target_dir, f'tdim-{tdim[0]}-{tdim[1]}_{os.path.basename(path_root_dir).split("_")[0]}.pickle' )

path_imgs = glob.glob(os.path.join(path_root_dir, '*.JPG'))

#%%

PIMGS = []
RGB_DOM = []
RGB_AVG = []
zstack = []
pbar = tqdm(total = len(path_imgs), position=0, leave=True)
for path_img in path_imgs[0:]:
    img = io.imread(path_img)[:, :, :]
    nimg = cv2.resize(img, tdim)
    pixels = np.float32(nimg.reshape(-1, 3))
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 200, .1)
    flags = cv2.KMEANS_RANDOM_CENTERS
    _, labels, palette = cv2.kmeans(pixels, n_colors, None, criteria, 10, flags)
    _, counts = np.unique(labels, return_counts=True)
    dominant = palette[np.argmax(counts)]
    average = nimg.mean(axis=0).mean(axis=0).astype('float32')
    RGB_DOM.append(dominant)
    RGB_AVG.append(average)
    PIMGS.append(path_img)
    zstack.append(nimg)
    pbar.update(n=1)
RGB_DOM = np.array(RGB_DOM)
RGB_AVG = np.array(RGB_AVG)

#%%

df = pd.DataFrame(list(zip(PIMGS, RGB_DOM, RGB_AVG, zstack)),
                  columns = ['Image-path', 'RGB-Dom', 'RGB-Avg', 'Image'])
df.to_pickle(path_df)












    
    
    
    
    
    
    
    
    
    
    
    
    
    
    