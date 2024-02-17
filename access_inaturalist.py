# -*- coding: utf-8 -*-
"""
Created on Fri Aug 19 20:57:40 2022

@author: wlwee
"""
#%%
import os
import urllib.request
import pandas as pd
from tqdm import tqdm

#%%

path_df = r'C:/Users/wlwee/Documents/hypothesis-writing/inaturalist_mosaics/DATA/Nudibranch/observations-256094.csv/observations-256094.csv'
target_dir = r'C:/Users/wlwee/Documents/hypothesis-writing/inaturalist_mosaics/DATA/Nudibranch'
df = pd.read_csv(path_df)

#%%

urls = df['image_url'].tolist()
iDs = df['id'].tolist()

#%%
maxl = len(urls)
pbar = tqdm(total = maxl, position=0, leave=True)
for url, iD in list(zip(urls[0:maxl-1], iDs[0:maxl-1])):
    img_path = os.path.join(target_dir, str(iD) + '.jpg')
    if os.path.exists(img_path) != True:
        urllib.request.urlretrieve(url, img_path)
    pbar.update(n=1)
pbar.close()

