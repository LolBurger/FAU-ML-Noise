import numpy as np
import scipy
import scipy.integrate
import pandas as pd
import glob
import os
import h5py
import shutil
import os
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from sklearn.cluster import DBSCAN
from pyevtk.hl import pointsToVTK
from scipy.spatial.distance import cdist
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams.update({'font.size': 22})

fileDir = r'..\dataexp\\'
hffile = fileDir +'totalData.h5'
jj = 'slice_10'
ldf = []
with h5py.File(hffile, 'r') as f:
    slice = [key for key in f.keys()]
    for ss in slice:
        head = [key for key in f[ss][jj].keys()]
        df = pd.DataFrame(data=None, columns=head)
        for c in df.columns:
            df[c] = np.asarray(f[ss][jj][c])
        ldf.append(df)

df_final = pd.DataFrame(data=None, columns=ldf[0].columns)
for dd in ldf:
    df_final = pd.concat([df_final,dd])

df_final = df_final.drop(columns=['X','Y','Z','WD'])
corr = df_final.corr()

xtick= np.arange(0,18)
xlabel = [str(x) for x in range(1,19)]
plt.figure(figsize=(12,12))

def highlight_cell(x,y, ax=None, **kwargs):
    rect = plt.Rectangle((x-.5, y-.5), 1,1, fill=False, **kwargs)
    ax = ax or plt.gca()
    ax.add_patch(rect)
    return rect


aa = plt.imshow(corr.values,cmap='rainbow',vmin=-1, vmax=1)
plt.xticks(xtick,labels=xlabel)
plt.yticks(xtick,labels=xlabel)

for gg in range(len(corr)):
    for hh in range(len(corr)):
        highlight_cell(gg,hh,edgecolor='k',linewidth=0.5)


plt.colorbar(aa)
plt.tight_layout()
plt.savefig('IMAGES/EDAfull.png')
plt.show()